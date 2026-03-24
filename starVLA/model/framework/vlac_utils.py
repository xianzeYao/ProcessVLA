from __future__ import annotations

from collections import defaultdict, deque
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Sequence

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.signal_common import (
    DEFAULT_VLAC_MODEL_PATH,
    DEFAULT_VLAC_MODEL_TYPE,
    DEFAULT_VLAC_PYTHON,
    _load_vlac_runtime,
    align_vlac_curve,
    compute_vlac_curve_signal,
    read_video,
    run_vlac_web_trajectory_critic,
)
from starVLA.model.framework.signal_common import read_video_fps, select_primary_video_key


def _to_pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def vlac_value_update(prev_value: float, critic: float) -> float:
    prev_value = float(prev_value)
    critic = float(critic)
    return prev_value + (100.0 - prev_value) * critic / 100.0


def _resolve_instruction(dataset, trajectory_id: int) -> str:
    dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
    instructions = dataset.get_language(
        trajectory_id, dataset.modality_keys["language"][0], 0
    )
    instruction = str(instructions[0]).strip() if instructions else ""
    if not instruction:
        raise ValueError(
            f"Empty instruction for trajectory {trajectory_id} in dataset {dataset.dataset_name}"
        )
    return instruction


def _build_reference_dataset_cfg(
    *,
    data_root_dir: str,
    data_mix: str,
    video_backend: str = "torchvision_av",
):
    return OmegaConf.create(
        {
            "data_root_dir": str(Path(data_root_dir).expanduser().resolve()),
            "data_mix": str(data_mix),
            "video_backend": str(video_backend),
        }
    )


class DatasetSeededVLACReferenceResolver:
    def __init__(
        self,
        *,
        data_root_dir: str,
        data_mix: str,
        dataset_name: str | None = None,
        reference_seed: int = 42,
        video_backend: str = "torchvision_av",
    ) -> None:
        self.data_root_dir = str(Path(data_root_dir).expanduser().resolve())
        self.data_mix = str(data_mix)
        self.dataset_name = None if dataset_name is None else str(dataset_name)
        self.reference_seed = int(reference_seed)
        self.video_backend = str(video_backend)
        self.task_to_records_by_dataset = {}

        vla_cfg = _build_reference_dataset_cfg(
            data_root_dir=self.data_root_dir,
            data_mix=self.data_mix,
            video_backend=self.video_backend,
        )
        dataset = get_vla_dataset(data_cfg=vla_cfg, mode="train")

        for single_dataset in dataset.datasets:
            if self.dataset_name is not None and single_dataset.dataset_name != self.dataset_name:
                continue
            video_key = select_primary_video_key(single_dataset.modality_keys["video"])
            video_subkey = video_key.replace("video.", "")
            task_to_records = defaultdict(list)
            iterable = zip(
                single_dataset.trajectory_ids.tolist(),
                single_dataset.trajectory_lengths.tolist(),
            )
            for trajectory_id, _trajectory_length in iterable:
                trajectory_id = int(trajectory_id)
                instruction = _resolve_instruction(single_dataset, trajectory_id)
                video_path = Path(
                    single_dataset.get_video_path(trajectory_id, video_subkey)
                ).expanduser().resolve()
                task_to_records[instruction].append(
                    {
                        "trajectory_id": trajectory_id,
                        "instruction": instruction,
                        "video_path": video_path,
                    }
                )
            self.task_to_records_by_dataset[single_dataset.dataset_name] = task_to_records

        if self.dataset_name is not None and self.dataset_name not in self.task_to_records_by_dataset:
            raise ValueError(
                f"reference_dataset_name={self.dataset_name!r} was not found in data_mix={self.data_mix!r} "
                f"under data_root_dir={self.data_root_dir!r}."
            )
        if not self.task_to_records_by_dataset:
            raise ValueError(
                f"No reference datasets found for data_mix={self.data_mix!r} under data_root_dir={self.data_root_dir!r}."
            )
        if self.dataset_name is None:
            if len(self.task_to_records_by_dataset) != 1:
                available = ", ".join(sorted(self.task_to_records_by_dataset.keys()))
                raise ValueError(
                    "reference_dataset_name must be provided when the configured data_mix contains multiple datasets. "
                    f"Available datasets: {available}"
                )
            self.dataset_name = next(iter(self.task_to_records_by_dataset.keys()))

    def select_reference_video_path(
        self,
        *,
        instruction: str,
        task_id: int | None = None,
        episode_idx: int | None = None,
        dataset_name: str | None = None,
    ) -> str:
        selected_dataset = self.dataset_name if dataset_name is None else str(dataset_name)
        if selected_dataset not in self.task_to_records_by_dataset:
            available = ", ".join(sorted(self.task_to_records_by_dataset.keys()))
            raise ValueError(
                f"Dataset {selected_dataset!r} is not available in reference pool. Available datasets: {available}"
            )
        normalized_instruction = str(instruction).strip()
        candidates = self.task_to_records_by_dataset[selected_dataset].get(
            normalized_instruction, []
        )
        if not candidates:
            raise ValueError(
                "No same-dataset exact-instruction reference candidates found for "
                f"dataset={selected_dataset!r}, instruction={normalized_instruction!r}."
            )
        selector = random.Random(
            f"{self.reference_seed}|{selected_dataset}|{normalized_instruction}|{task_id}|{episode_idx}"
        )
        chosen = selector.choice(candidates)
        return str(Path(chosen["video_path"]).expanduser().resolve())


def compute_vlac_pair_critic(
    *,
    previous_frame,
    current_frame,
    instruction: str,
    reference_frames: Sequence[Image.Image],
    ref_num: int = 6,
    batch_num: int = 5,
    rich: bool = False,
    think: bool = False,
    device: str = "cuda",
) -> float:
    critic = _load_vlac_runtime(str(device))
    image_list = [_to_pil_rgb(previous_frame), _to_pil_rgb(current_frame)]
    ref_list = [_to_pil_rgb(frame) for frame in reference_frames] if reference_frames else None
    critic_list, _value_list = critic.get_trajectory_critic(
        task=str(instruction),
        image_list=image_list,
        ref_image_list=ref_list,
        batch_num=int(batch_num),
        ref_num=int(ref_num),
        think=bool(think),
        skip=1,
        rich=bool(rich),
        reverse_eval=False,
        frame_skip=True,
        addition_scale=1,
        bias=0,
        related_critic=False,
        positive_clip=0,
        negative_clip=0,
    )
    if not critic_list:
        return 0.0
    return float(critic_list[-1])


class VLACOnlineSubprocessClient:
    def __init__(
        self,
        *,
        vlac_python: str,
        model_path: str,
        model_type: str,
        repo_root: str,
        device: str,
    ) -> None:
        self.vlac_python = str(Path(vlac_python).expanduser().resolve())
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.model_type = str(model_type)
        self.repo_root = str(Path(repo_root).expanduser().resolve())
        self.device = str(device)
        self.temp_dir = tempfile.TemporaryDirectory(prefix="vlac_online_frames_")
        self._frame_index = 0
        runner_path = Path(__file__).with_name("vlac_online_subprocess_runner.py")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [
                self.vlac_python,
                "-u",
                str(runner_path),
                "--model-path",
                self.model_path,
                "--model-type",
                self.model_type,
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
            raise RuntimeError("Failed to initialize VLAC online subprocess pipes.")

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

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _save_frame(self, frame, stem: str) -> str:
        self._frame_index += 1
        output_path = (
            Path(self.temp_dir.name)
            / f"{self._frame_index:06d}_{stem}.png"
        )
        _to_pil_rgb(frame).save(output_path)
        return str(output_path)

    def compute_pair_critic(
        self,
        *,
        previous_frame,
        current_frame,
        instruction: str,
        reference_video_path: str,
        ref_num: int,
        batch_num: int,
        rich: bool,
        think: bool,
    ) -> float:
        previous_frame_path = self._save_frame(previous_frame, "prev")
        current_frame_path = self._save_frame(current_frame, "curr")
        request = {
            "previous_frame_path": previous_frame_path,
            "current_frame_path": current_frame_path,
            "instruction": str(instruction),
            "reference_video_path": str(Path(reference_video_path).expanduser().resolve()),
            "ref_num": int(ref_num),
            "batch_num": int(batch_num),
            "rich": bool(rich),
            "think": bool(think),
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
                    "VLAC online subprocess terminated unexpectedly. "
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
                raise RuntimeError(
                    f"VLAC online subprocess error: {response['error']}"
                )
            return float(response["critic"])


class OnlineVLACSignalState:
    def __init__(
        self,
        *,
        instruction: str,
        reference_video_path: str,
        signal_kind: str = "value",
        skip: int = 5,
        frame_skip: bool = False,
        ref_num: int = 6,
        batch_num: int = 5,
        rich: bool = False,
        think: bool = False,
        device: str = "cuda",
        subprocess_client: VLACOnlineSubprocessClient | None = None,
    ) -> None:
        self.instruction = str(instruction)
        self.reference_video_path = str(reference_video_path)
        self.signal_kind = str(signal_kind).lower()
        self.skip = max(int(skip), 1)
        self.frame_skip = bool(frame_skip)
        self.ref_num = int(ref_num)
        self.batch_num = int(batch_num)
        self.rich = bool(rich)
        self.think = bool(think)
        self.device = str(device)
        self.subprocess_client = subprocess_client
        self.frame_buffer = deque(maxlen=self.skip + 1)
        self.prev_value = 0.0
        if self.subprocess_client is None:
            reference_frames_np, _ = read_video(self.reference_video_path)
            self.reference_frames = [_to_pil_rgb(frame) for frame in reference_frames_np]
        else:
            self.reference_frames = None

    def reset(self, instruction: str | None = None) -> None:
        if instruction is not None:
            self.instruction = str(instruction)
        self.frame_buffer.clear()
        self.prev_value = 0.0

    def append_frame(self, frame) -> None:
        self.frame_buffer.append(_to_pil_rgb(frame))

    def compute_current_signal(self) -> float:
        if len(self.frame_buffer) <= self.skip:
            return 0.0

        previous_frame = self.frame_buffer[0]
        current_frame = self.frame_buffer[-1]
        if self.subprocess_client is None:
            critic = compute_vlac_pair_critic(
                previous_frame=previous_frame,
                current_frame=current_frame,
                instruction=self.instruction,
                reference_frames=self.reference_frames,
                ref_num=self.ref_num,
                batch_num=self.batch_num,
                rich=self.rich,
                think=self.think,
                device=self.device,
            )
        else:
            critic = self.subprocess_client.compute_pair_critic(
                previous_frame=previous_frame,
                current_frame=current_frame,
                instruction=self.instruction,
                reference_video_path=self.reference_video_path,
                ref_num=self.ref_num,
                batch_num=self.batch_num,
                rich=self.rich,
                think=self.think,
            )
        if not self.frame_skip:
            critic = critic / float(self.skip)

        if self.signal_kind == "critic":
            return float(critic)
        if self.signal_kind == "value":
            self.prev_value = vlac_value_update(self.prev_value, critic)
            return float(self.prev_value / 100.0)
        raise ValueError(
            f"Unsupported vlac_online signal_kind={self.signal_kind!r}. Supported values are: critic, value."
        )

__all__ = [
    "DEFAULT_VLAC_MODEL_PATH",
    "DEFAULT_VLAC_MODEL_TYPE",
    "DEFAULT_VLAC_PYTHON",
    "DatasetSeededVLACReferenceResolver",
    "OnlineVLACSignalState",
    "VLACOnlineSubprocessClient",
    "align_vlac_curve",
    "compute_vlac_curve_signal",
    "compute_vlac_pair_critic",
    "read_video_fps",
    "run_vlac_web_trajectory_critic",
    "select_primary_video_key",
    "vlac_value_update",
]
