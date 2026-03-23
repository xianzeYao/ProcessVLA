from __future__ import annotations
import matplotlib.pyplot as plt

import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

import imageio.v2 as imageio
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib

matplotlib.use("Agg")


STEP_PATTERN = re.compile(r"step_(\d+)_last_hidden_states_meta\.pt$")

# Fill these defaults for your local machine. Leave a Python entry empty to use the
# current interpreter instead of a dedicated model-specific environment.
DEFAULT_LIV_MODEL_PATH = "/data/yxz/models/VLAC"
DEFAULT_LIV_PYTHON = "/data/yxz/conda/envs/VLAC/bin/python"

DEFAULT_ROBOMETER_MODEL_PATH = "/data/yxz/models/Robo_Meter"
DEFAULT_ROBOMETER_QWEN_SNAPSHOT_PATH = ""
DEFAULT_ROBOMETER_UNSLOTH_SNAPSHOT_PATH = ""
DEFAULT_ROBOMETER_PYTHON = "/data/yxz/models/VLAC"

DEFAULT_ROBODOPAMINE_MODEL_PATH = ""
DEFAULT_ROBODOPAMINE_PYTHON = ""

DEFAULT_VLAC_MODEL_PATH = "/data/yxz/models/VLAC"
DEFAULT_VLAC_MODEL_TYPE = "internvl2"
DEFAULT_VLAC_PYTHON = "/data/yxz/conda/envs/VLAC/bin/python"

DEFAULT_MODEL_PATHS = {
    "liv": DEFAULT_LIV_MODEL_PATH,
    "robometer": DEFAULT_ROBOMETER_MODEL_PATH,
    "robodopamine": DEFAULT_ROBODOPAMINE_MODEL_PATH,
    "vlac": DEFAULT_VLAC_MODEL_PATH,
}

DEFAULT_MODEL_PYTHONS = {
    "liv": DEFAULT_LIV_PYTHON,
    "robometer": DEFAULT_ROBOMETER_PYTHON,
    "robodopamine": DEFAULT_ROBODOPAMINE_PYTHON,
    "vlac": DEFAULT_VLAC_PYTHON,
}


def _normalize_optional_string(value: str | Path | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def get_default_model_path(model_name: str) -> str | None:
    model_key = str(model_name).lower()
    if model_key not in DEFAULT_MODEL_PATHS:
        raise ValueError(f"Unsupported model name: {model_name!r}")
    return _normalize_optional_string(DEFAULT_MODEL_PATHS[model_key])


def get_default_model_python(model_name: str) -> str | None:
    model_key = str(model_name).lower()
    if model_key not in DEFAULT_MODEL_PYTHONS:
        raise ValueError(f"Unsupported model name: {model_name!r}")
    return _normalize_optional_string(DEFAULT_MODEL_PYTHONS[model_key])


def resolve_model_python(
    model_name: str,
    explicit_python: str | Path | None = None,
) -> str | None:
    return _normalize_optional_string(explicit_python) or get_default_model_python(model_name)


def build_signal_save_payload(
    qwen_inputs,
    last_hidden: torch.Tensor,
    base_last_hidden: torch.Tensor,
    signal: torch.Tensor | None,
    batch_images,
    instructions: List[str],
    tokenizer,
    image_token_id: int,
) -> Dict[str, Any]:
    input_ids = qwen_inputs["input_ids"].detach().cpu()
    attention_mask = qwen_inputs["attention_mask"].detach().cpu()
    image_grid_thw = qwen_inputs.get("image_grid_thw")
    image_grid_thw = image_grid_thw.detach().cpu(
    ) if image_grid_thw is not None else None
    image_mask = input_ids == int(image_token_id)
    pad_token_id = tokenizer.pad_token_id

    special_token_ids = {
        "image_token_id": int(image_token_id),
        "pad_token_id": int(pad_token_id) if pad_token_id is not None else None,
        "vision_start_id": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "vision_end_id": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        "im_start_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "im_end_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
    }
    decoded_inputs = tokenizer.batch_decode(
        input_ids, skip_special_tokens=False)
    token_counts = {
        "sequence_length": attention_mask.sum(dim=1),
        "num_image_tokens": image_mask.sum(dim=1),
    }
    if image_grid_thw is not None:
        token_counts["image_grid_thw"] = image_grid_thw

    return {
        "hidden_states_last": last_hidden.detach().cpu(),
        "hidden_states_last_base": base_last_hidden.detach().cpu(),
        "signal": signal.detach().cpu() if signal is not None else None,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_mask": image_mask,
        "special_token_ids": special_token_ids,
        "token_counts": token_counts,
        "decoded_inputs": decoded_inputs,
        "images": batch_images,
        "instructions": instructions,
    }


def build_hidden_state_save_payload(
    qwen_inputs,
    last_hidden: torch.Tensor,
    base_last_hidden: torch.Tensor,
    batch_images,
    instructions: List[str],
    tokenizer,
    image_token_id: int,
    signal: torch.Tensor | None = None,
) -> Dict[str, Any]:
    return build_signal_save_payload(
        qwen_inputs=qwen_inputs,
        last_hidden=last_hidden,
        base_last_hidden=base_last_hidden,
        signal=signal,
        batch_images=batch_images,
        instructions=instructions,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
    )


def compute_signal(
    batch_images,
    instructions: List[str],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = len(batch_images)
    return torch.zeros(batch_size, device=device, dtype=dtype)


def compute_signal_curve(
    video_frames: List[np.ndarray],
    instruction: str,
    signal_name: str = "signal",
) -> np.ndarray:
    del instruction
    del signal_name
    frame_count = len(video_frames)
    if frame_count == 0:
        return np.zeros(0, dtype=np.float32)
    return np.zeros(frame_count, dtype=np.float32)


def default_signal_curve_path(video_path: str | Path, signal_name: str = "signal") -> Path:
    video_path = Path(video_path)
    return video_path.with_name(f"{video_path.stem}_{signal_name}_curve.npz")


def default_signal_overlay_path(video_path: str | Path, stem_suffix: str) -> Path:
    video_path = Path(video_path)
    return video_path.with_name(f"{video_path.stem}_{stem_suffix}.mp4")


def signal_cache_curve_path(
    cache_root: str | Path,
    dataset_name: str,
    trajectory_id: int,
    signal_name: str = "vlac",
) -> Path:
    cache_root = Path(cache_root).expanduser().resolve()
    return cache_root / dataset_name / f"trajectory_{int(trajectory_id)}_{signal_name}.npz"


def select_primary_video_key(video_keys: Sequence[str]) -> str:
    if not video_keys:
        raise ValueError("No video keys are available.")
    for key in video_keys:
        if "wrist" not in key.lower():
            return key
    return str(video_keys[0])


def aggregate_signal_at_step(
    curve: Sequence[float] | np.ndarray,
    base_index: int,
    action_chunk_length: int = 1,
    align_mode: str = "current",
) -> float:
    curve_array = np.asarray(curve, dtype=np.float32).reshape(-1)
    if curve_array.size == 0:
        return 0.0

    align_mode = str(align_mode).lower()
    start = int(np.clip(base_index, 0, curve_array.size - 1))
    action_chunk_length = max(int(action_chunk_length), 1)

    if align_mode in {"current", "base", "base_index"}:
        window = curve_array[start: start + 1]
    elif align_mode in {"chunk_last", "last"}:
        end = min(start + action_chunk_length - 1, curve_array.size - 1)
        window = curve_array[end: end + 1]
    elif align_mode in {"chunk_mean", "mean"}:
        end = min(start + action_chunk_length, curve_array.size)
        window = curve_array[start:end]
    else:
        raise ValueError(
            f"Unsupported signal align_mode={align_mode!r}; "
            "expected 'current', 'chunk_last', or 'chunk_mean'."
        )

    finite_window = window[np.isfinite(window)]
    if finite_window.size == 0:
        return 0.0
    return float(np.mean(finite_window))


def read_video(video_path: str | Path) -> tuple[list[np.ndarray], float]:
    video_path = Path(video_path).expanduser().resolve()
    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data() or {}
        fps = float(meta.get("fps", 10.0))
        frames = [frame for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"Video contains no frames: {video_path}")
    return frames, fps


def read_video_fps(video_path: str | Path) -> float:
    video_path = Path(video_path).expanduser().resolve()
    try:
        reader = imageio.get_reader(str(video_path))
        try:
            meta = reader.get_meta_data() or {}
            fps = meta.get("fps", 10.0)
            return float(fps) if fps is not None else 10.0
        finally:
            reader.close()
    except Exception:
        # Some pyav-backed videos expose broken timing metadata; LIBERO rollouts
        # are commonly authored at 10 FPS, so use that as a safe fallback.
        return 10.0


def _transcode_video_for_vlac(
    video_path: str | Path,
    *,
    temp_root: str | Path,
    stem_suffix: str,
    vlac_python: str | Path | None = None,
) -> Path:
    source_path = Path(video_path).expanduser().resolve()
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None and vlac_python is not None:
        candidate = Path(vlac_python).expanduser().resolve().parent / "ffmpeg"
        if candidate.exists():
            ffmpeg_bin = str(candidate)
    if ffmpeg_bin is None:
        try:
            import imageio_ffmpeg

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None
    if ffmpeg_bin is None:
        raise FileNotFoundError(
            "ffmpeg is required to transcode videos for VLAC, but it was not found. "
            "Install ffmpeg or `pip install imageio-ffmpeg` in the environment running ProcessVLA."
        )

    output_dir = Path(temp_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_path.stem}_{stem_suffix}_h264.mp4"
    command = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def load_instruction_from_hidden_dir(hidden_dir: str | Path) -> str:
    hidden_dir_path = Path(hidden_dir).expanduser().resolve()
    candidates = sorted(hidden_dir_path.glob(
        "step_*_last_hidden_states_meta.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No hidden state metadata found under {hidden_dir_path}")

    payload = torch.load(candidates[0], map_location="cpu", weights_only=False)
    instructions = payload.get("instructions")
    if not instructions:
        raise ValueError(f"No instructions found in {candidates[0]}")
    return str(instructions[0])


def save_signal_curve_npz(
    output_path: str | Path,
    curve: Sequence[float] | np.ndarray,
    signal_name: str,
    instruction: str,
    video_path: str | Path,
    fps: float,
    status: str,
    extra_arrays: Dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    arrays: Dict[str, Any] = dict(
        curve=np.asarray(curve, dtype=np.float32),
        signal_name=str(signal_name),
        instruction=str(instruction),
        video_path=str(Path(video_path).expanduser().resolve()),
        fps=np.asarray([fps], dtype=np.float32),
        status=np.asarray([status], dtype=object),
    )
    if extra_arrays:
        arrays.update(extra_arrays)
    np.savez(output_path, **arrays)
    return output_path


def npz_scalar(payload, key: str, default: str) -> str:
    if key not in payload.files:
        return default
    value = payload[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def normalize_curve(curve: np.ndarray, target_length: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    if curve.size == target_length:
        return curve
    if curve.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    xs_old = np.linspace(0.0, 1.0, num=curve.size)
    xs_new = np.linspace(0.0, 1.0, num=target_length)
    return np.interp(xs_new, xs_old, curve).astype(np.float32)


def align_vlac_curve(
    curve: np.ndarray,
    *,
    target_length: int,
    skip: int,
    frame_skip: bool,
    signal_kind: str,
) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    target_length = int(target_length)
    skip = max(int(skip), 0)

    if curve.size == target_length:
        return curve
    if curve.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if frame_skip:
        return normalize_curve(curve, target_length)

    aligned = np.zeros(target_length, dtype=np.float32)

    if signal_kind == "value" and curve.size == max(target_length - skip, 0):
        aligned[skip:] = curve[: max(target_length - skip, 0)]
        return aligned

    if signal_kind == "critic" and curve.size == max(target_length - skip + 1, 0):
        tail = curve[1:]
        aligned[skip:] = tail[: max(target_length - skip, 0)]
        return aligned

    return normalize_curve(curve, target_length)


def load_signal_curve_npz(curve_path: str | Path, target_length: int) -> Dict[str, Any]:
    curve_path = Path(curve_path).expanduser().resolve()
    payload = np.load(curve_path, allow_pickle=True)
    return {
        "name": npz_scalar(payload, "signal_name", curve_path.stem),
        "curve": normalize_curve(payload["curve"], target_length),
        "instruction": npz_scalar(payload, "instruction", ""),
        "curve_path": curve_path,
    }


def render_signal_overlay_video(
    video_path: str | Path,
    curves: Sequence[Dict[str, Any]],
    instruction: str,
    output_path: str | Path,
    title: str = "Signal Curves",
) -> Path:
    video_path = Path(video_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    frames, fps = read_video(video_path)
    writer = imageio.get_writer(str(output_path), fps=fps)
    try:
        for frame_idx, frame in enumerate(frames):
            writer.append_data(
                render_signal_overlay_frame(
                    frame=frame,
                    curves=curves,
                    frame_idx=frame_idx,
                    instruction=instruction,
                    title=title,
                )
            )
    finally:
        writer.close()
    return output_path


def render_signal_overlay_frame(
    frame: np.ndarray,
    curves: Sequence[Dict[str, Any]],
    frame_idx: int,
    instruction: str,
    title: str,
) -> np.ndarray:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    fig.tight_layout(pad=2.0)

    axes[0].imshow(frame)
    axes[0].axis("off")
    axes[0].set_title("Rollout")

    color_cycle = ["#1f6f8b", "#c53b32", "#6c9a3a", "#7a4ea3", "#e07a1f"]
    all_values = np.concatenate(
        [item["curve"] for item in curves]) if curves else np.zeros(1, dtype=np.float32)
    finite_values = all_values[np.isfinite(all_values)]
    y_min = float(finite_values.min()) if finite_values.size else 0.0
    y_max = float(finite_values.max()) if finite_values.size else 1.0
    if abs(y_max - y_min) < 1e-6:
        y_min -= 0.5
        y_max += 0.5
    for idx, item in enumerate(curves):
        curve = item["curve"]
        color = color_cycle[idx % len(color_cycle)]
        xs = np.arange(curve.shape[0])
        axes[1].plot(xs, curve, color=color, linewidth=2.0, label=item["name"])
        axes[1].scatter([frame_idx], [curve[frame_idx]], color=color, s=20)
    axes[1].axvline(frame_idx, color="#222222", linestyle="--", linewidth=1.2)
    axes[1].set_xlim(0, max(curves[0]["curve"].shape[0] - 1, 1))
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_title(title)
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Signal")
    axes[1].legend(loc="upper right")

    short_instruction = instruction if len(
        instruction) <= 120 else instruction[:117] + "..."
    fig.suptitle(short_instruction, fontsize=10)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(
        height, width, 4)[..., :3]
    plt.close(fig)
    return image


def load_sorted_hidden_signal_entries(hidden_dir: str | Path) -> list[tuple[int, Path]]:
    hidden_dir = Path(hidden_dir).expanduser().resolve()
    entries = []
    for path in hidden_dir.glob("step_*_last_hidden_states_meta.pt"):
        match = STEP_PATTERN.search(path.name)
        if match is None:
            continue
        entries.append((int(match.group(1)), path))
    entries.sort(key=lambda item: item[0])
    if not entries:
        raise FileNotFoundError(
            f"No hidden state metadata found under {hidden_dir}")
    return entries


def staircase_expand(
    steps: np.ndarray,
    values: np.ndarray,
    frame_count: int,
) -> np.ndarray:
    expanded = np.full(frame_count, np.nan, dtype=np.float32)
    if len(steps) == 0 or len(values) == 0:
        return expanded
    for idx, step in enumerate(steps):
        start = int(step)
        end = int(steps[idx + 1]) if idx + 1 < len(steps) else frame_count
        if start >= frame_count:
            continue
        expanded[start:min(end, frame_count)] = values[idx]
    first_valid = int(steps[0])
    if first_valid > 0:
        expanded[:first_valid] = values[0]
    return expanded


def build_online_signal_curve(
    hidden_dir: str | Path,
    frame_count: int,
) -> tuple[np.ndarray, str]:
    entries = load_sorted_hidden_signal_entries(hidden_dir)
    steps = []
    values = []
    instruction = ""
    for step, path in entries:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        signal = payload.get("signal")
        if signal is None:
            value = np.nan
        else:
            signal_tensor = torch.as_tensor(signal).reshape(-1)
            value = float(signal_tensor[0].item()
                          ) if signal_tensor.numel() else np.nan
        steps.append(step)
        values.append(value)
        if not instruction:
            instructions = payload.get("instructions") or []
            if instructions:
                instruction = str(instructions[0])
    curve = staircase_expand(
        np.asarray(steps, dtype=np.int32),
        np.asarray(values, dtype=np.float32),
        frame_count,
    )
    return curve, instruction


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_repo_import_path(repo_name: str) -> None:
    repo_root = _workspace_root() / repo_name
    if not repo_root.exists():
        raise ImportError(
            f"Local repo '{repo_name}' not found under {_workspace_root()}.")
    path_str = str(repo_root)
    if path_str not in os.sys.path:
        os.sys.path.insert(0, path_str)


@lru_cache(maxsize=4)
def _load_vlac_runtime(device_name: str):
    if not DEFAULT_VLAC_MODEL_PATH:
        raise ValueError(
            "VLAC selected but no default model path is configured. "
            "Edit DEFAULT_VLAC_MODEL_PATH in signal_utils.py."
        )

    _ensure_repo_import_path("VLAC")
    from evo_vlac import GAC_model

    critic = GAC_model(tag="critic")
    critic.init_model(
        model_path=DEFAULT_VLAC_MODEL_PATH,
        model_type=DEFAULT_VLAC_MODEL_TYPE,
        device_map=device_name,
    )
    critic.temperature = 0.5
    critic.top_k = 1
    critic.set_config()
    critic.set_system_prompt()
    return critic


def run_vlac_web_trajectory_critic(
    *,
    video_path: str | Path,
    instruction: str,
    fps: float,
    reference_video_path: str | Path | None = None,
    ref_num: int = 6,
    batch_num: int = 5,
    skip: int = 5,
    rich: bool = False,
    frame_skip: bool = False,
    think: bool = False,
    device: torch.device | None = None,
    done_flag: bool = False,
    in_context_done: bool = False,
    done_threshold: float = 0.9,
    output_path: str | Path | None = None,
    video_output: bool = False,
    vlac_python: str | Path | None = None,
) -> tuple[str | None, list[float], list[float], list[float] | None]:
    if device is None:
        raise ValueError("'device' is required for VLAC curve computation.")

    video_path = Path(video_path).expanduser().resolve()
    resolved_reference = None if reference_video_path is None else Path(
        reference_video_path).expanduser().resolve()
    resolved_output_path = video_path.parent if output_path is None else Path(
        output_path).expanduser().resolve()
    resolved_vlac_python = resolve_model_python("vlac", vlac_python)
    with ExitStack() as stack:
        if resolved_vlac_python is None:
            transcode_root = Path(
                stack.enter_context(tempfile.TemporaryDirectory(
                    prefix="vlac_video_transcode_"))
            )
            prepared_video_path = _transcode_video_for_vlac(
                video_path,
                temp_root=transcode_root,
                stem_suffix="main",
                vlac_python=resolved_vlac_python,
            )
            prepared_reference = None
            if resolved_reference is not None:
                prepared_reference = _transcode_video_for_vlac(
                    resolved_reference,
                    temp_root=transcode_root,
                    stem_suffix="reference",
                    vlac_python=resolved_vlac_python,
                )
            critic = _load_vlac_runtime(str(device))
            return critic.web_trajectory_critic(
                task_description=instruction,
                main_video_path=str(prepared_video_path),
                reference_video_path=None if prepared_reference is None else str(
                    prepared_reference),
                batch_num=int(batch_num),
                ref_num=int(ref_num),
                think=bool(think),
                skip=int(skip),
                rich=bool(rich),
                reverse_eval=False,
                output_path=str(resolved_output_path),
                fps=float(fps),
                frame_skip=bool(frame_skip),
                done_flag=bool(done_flag),
                in_context_done=bool(in_context_done),
                done_threshold=float(done_threshold),
                video_output=bool(video_output),
            )

        helper_script = Path(__file__).with_name("vlac_subprocess_runner.py")
        resolved_python = str(
            Path(resolved_vlac_python).expanduser().resolve())
        tmpdir = stack.enter_context(
            tempfile.TemporaryDirectory(prefix="vlac_subprocess_"))
        request_path = Path(tmpdir) / "request.json"
        response_path = Path(tmpdir) / "response.npz"
        request_payload = {
            "instruction": str(instruction),
            "video_path": str(video_path),
            "reference_video_path": None if resolved_reference is None else str(resolved_reference),
            "fps": float(fps),
            "ref_num": int(ref_num),
            "batch_num": int(batch_num),
            "skip": int(skip),
            "rich": bool(rich),
            "frame_skip": bool(frame_skip),
            "think": bool(think),
            "device": str(device),
            "done_flag": bool(done_flag),
            "in_context_done": bool(in_context_done),
            "done_threshold": float(done_threshold),
            "output_path": str(resolved_output_path),
            "video_output": bool(video_output),
            "model_path": str(DEFAULT_VLAC_MODEL_PATH),
            "model_type": str(DEFAULT_VLAC_MODEL_TYPE),
            "transcode_inputs": True,
        }
        request_path.write_text(json.dumps(
            request_payload, ensure_ascii=False))
        subprocess.run(
            [resolved_python, str(helper_script), str(
                request_path), str(response_path)],
            check=True,
        )
        payload = np.load(response_path, allow_pickle=True)
        result_video_path = str(
            payload["result_video_path"][0]) if payload["result_video_path"].size else ""
        value_list = payload["value_list"].astype(np.float32).tolist()
        critic_list = payload["critic_list"].astype(np.float32).tolist()
        done_list_raw = payload["done_list"].astype(np.float32)
        done_list = None if done_list_raw.size == 0 else done_list_raw.tolist()
        return (result_video_path or None), value_list, critic_list, done_list


def compute_vlac_curve_signal(
    video_path: str | Path,
    instruction: str,
    fps: float,
    *,
    frame_count: int,
    reference_video_path: str | Path | None = None,
    signal_kind: str = "value",
    ref_num: int = 6,
    batch_num: int = 5,
    skip: int = 5,
    rich: bool = False,
    frame_skip: bool = False,
    think: bool = False,
    device: torch.device | None = None,
    return_raw_curve: bool = False,
    vlac_python: str | Path | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if device is None:
        raise ValueError("'device' is required for VLAC curve computation.")

    video_path = Path(video_path).expanduser().resolve()
    resolved_reference = None if reference_video_path is None else Path(
        reference_video_path).expanduser().resolve()
    need_done = signal_kind == "done"
    _, value_list, critic_list, done_list = run_vlac_web_trajectory_critic(
        video_path=video_path,
        instruction=instruction,
        fps=float(fps),
        reference_video_path=resolved_reference,
        ref_num=int(ref_num),
        batch_num=int(batch_num),
        skip=int(skip),
        rich=bool(rich),
        frame_skip=bool(frame_skip),
        think=bool(think),
        device=device,
        done_flag=need_done,
        in_context_done=resolved_reference is not None,
        done_threshold=0.9,
        output_path=video_path.parent,
        video_output=False,
        vlac_python=vlac_python,
    )

    if signal_kind == "value":
        raw_curve = np.asarray(value_list, dtype=np.float32)
        if raw_curve.size > 0 and float(np.nanmax(raw_curve)) > 1.0:
            raw_curve = raw_curve / 100.0
    elif signal_kind == "critic":
        raw_curve = np.asarray([0.0] + [float(v)
                               for v in critic_list], dtype=np.float32)
    elif signal_kind == "done":
        raw_curve = np.full(frame_count, np.nan, dtype=np.float32) if done_list is None else np.asarray(
            done_list, dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported VLAC signal_kind={signal_kind!r}; expected 'value', 'critic', or 'done'."
        )

    normalized_curve = align_vlac_curve(
        raw_curve,
        target_length=frame_count,
        skip=int(skip),
        frame_skip=bool(frame_skip),
        signal_kind=signal_kind,
    )
    if return_raw_curve:
        return normalized_curve, raw_curve
    return normalized_curve
