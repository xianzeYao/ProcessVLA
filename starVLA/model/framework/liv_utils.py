from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
import torch

from starVLA.model.framework.signal_common import (
    DEFAULT_LIV_MODEL_PATH,
    DEFAULT_LIV_PYTHON,
    compute_liv_curve_signal,
    compute_liv_signal,
    load_liv_runtime,
)


def _to_pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            finite_values = array[np.isfinite(array)]
            if finite_values.size > 0 and finite_values.min() >= 0.0 and finite_values.max() <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _describe_image_like(image) -> dict:
    if isinstance(image, Image.Image):
        array = np.asarray(image)
    else:
        array = np.asarray(image)
    finite_mask = np.isfinite(array)
    finite_values = array[finite_mask]
    stats = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_count": int(finite_values.size),
    }
    if finite_values.size > 0:
        stats.update(
            {
                "min": float(finite_values.min()),
                "max": float(finite_values.max()),
                "mean": float(finite_values.mean()),
            }
        )
    return stats


class LIVOnlineSubprocessClient:
    def __init__(
        self,
        *,
        liv_python: str,
        repo_root: str,
        device: str,
    ) -> None:
        self.liv_python = str(Path(liv_python).expanduser().resolve())
        self.repo_root = str(Path(repo_root).expanduser().resolve())
        self.device = str(device)
        self.temp_dir = tempfile.TemporaryDirectory(prefix="liv_online_frames_")
        self._frame_index = 0

        runner_path = Path(__file__).with_name("liv_online_subprocess_runner.py")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [
                self.liv_python,
                "-u",
                str(runner_path),
                "--repo-root",
                self.repo_root,
                "--device",
                self.device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.process = process
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("Failed to initialize LIV online subprocess pipes.")
        self.runtime_info = self._read_startup_runtime()

    def _read_startup_runtime(self) -> dict:
        assert self.process is not None and self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if line == "":
                stderr_text = ""
                if self.process.stderr is not None:
                    try:
                        stderr_text = self.process.stderr.read()
                    except Exception:
                        stderr_text = ""
                raise RuntimeError(
                    "LIV online subprocess failed during startup. "
                    f"python={self.liv_python} repo_root={self.repo_root} stderr: {stderr_text}"
                )
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            runtime = response.get("runtime", {})
            if response.get("ready"):
                return runtime
            if "error" in response:
                raise RuntimeError(
                    "LIV online subprocess startup error: "
                    f"{response['error']} runtime={runtime}"
                )

    def close(self) -> None:
        if getattr(self, "process", None) is not None:
            try:
                if self.process.stdin is not None:
                    self.process.stdin.close()
            except Exception:
                pass
            try:
                self.process.terminate()
            except Exception:
                pass
            try:
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        if getattr(self, "temp_dir", None) is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _save_frame(self, frame, stem: str) -> str:
        self._frame_index += 1
        output_path = Path(self.temp_dir.name) / f"{self._frame_index:06d}_{stem}.png"
        _to_pil_rgb(frame).save(output_path)
        return str(output_path)

    def compute_signals(
        self,
        *,
        images,
        instructions,
    ) -> list[float]:
        if len(images) != len(instructions):
            raise ValueError(
                f"LIV subprocess batch size mismatch: {len(images)} images vs {len(instructions)} instructions."
            )
        source_image_stats = [_describe_image_like(image) for image in images]
        image_paths = []
        for batch_idx, image in enumerate(images):
            image_paths.append(self._save_frame(image, f"frame_{batch_idx}"))
        try:
            request = {
                "image_paths": image_paths,
                "instructions": [str(instruction) for instruction in instructions],
            }
            assert self.process is not None and self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if line == "":
                    stderr_text = ""
                    if self.process.stderr is not None:
                        try:
                            stderr_text = self.process.stderr.read()
                        except Exception:
                            stderr_text = ""
                    raise RuntimeError(
                        "LIV online subprocess terminated unexpectedly. "
                        f"stderr: {stderr_text}"
                    )
                line = line.strip()
                if not line:
                    continue
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in response:
                    runtime = response.get("runtime", self.runtime_info)
                    debug = response.get("debug")
                    debug_suffix = f" debug={debug}" if debug is not None else ""
                    raise RuntimeError(
                        "LIV online subprocess error: "
                        f"{response['error']} runtime={runtime} "
                        f"parent_image_stats={source_image_stats}{debug_suffix}"
                    )
                return [float(item) for item in response["signals"]]
        finally:
            for image_path in image_paths:
                try:
                    Path(image_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def compute_signal(
        self,
        *,
        image,
        instruction: str,
    ) -> float:
        signals = self.compute_signals(images=[image], instructions=[instruction])
        return float(signals[0])


__all__ = [
    "DEFAULT_LIV_MODEL_PATH",
    "DEFAULT_LIV_PYTHON",
    "LIVOnlineSubprocessClient",
    "compute_liv_curve_signal",
    "compute_liv_signal",
    "load_liv_runtime",
]
