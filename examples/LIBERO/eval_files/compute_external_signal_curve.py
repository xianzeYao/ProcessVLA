import argparse
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import re
import sys
from typing import Optional

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import torch

try:
    import av
except ImportError:
    av = None


REPO_ROOT = Path(__file__).resolve().parents[3]
STEP_PATTERN = re.compile(r"step_(\d+)_last_hidden_states_meta\.pt$")
ROLLOUT_VIDEO_PATTERN = re.compile(r"^rollout_(.+)_episode(\d+)_")
SUPPORTED_EXTERNAL_MODELS = ("liv", "robometer", "vlac", "robodopamine")


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load module {module_name!r} from {module_path}."
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_repo_import_path(repo_name: str, src_subdir: Optional[str] = None) -> None:
    workspace_root = Path(__file__).resolve().parents[4]
    repo_root = workspace_root / repo_name
    if not repo_root.exists():
        raise ImportError(
            f"Local repo '{repo_name}' not found under {workspace_root}."
        )
    import_path = repo_root / src_subdir if src_subdir is not None else repo_root
    path_str = str(import_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@lru_cache(maxsize=1)
def load_signal_utils_module():
    module_path = REPO_ROOT / "starVLA" / "model" / "framework" / "signal_utils.py"
    return load_module_from_path("critic4vla_signal_utils_external", module_path)


def default_curve_path(video_path: Path, model_name: str) -> Path:
    return video_path.with_name(f"{video_path.stem}_{model_name}_curve.npz")


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


def sort_hidden_files(hidden_dir: Path) -> list[tuple[int, Path]]:
    hidden_files = []
    for path in hidden_dir.glob("step_*_last_hidden_states_meta.pt"):
        match = STEP_PATTERN.search(path.name)
        if match is None:
            continue
        hidden_files.append((int(match.group(1)), path))
    hidden_files.sort(key=lambda item: item[0])
    if not hidden_files:
        raise FileNotFoundError(
            f"No hidden state files found under {hidden_dir}")
    return hidden_files


def infer_hidden_dir_from_video(video_path: Path) -> Optional[Path]:
    match = ROLLOUT_VIDEO_PATTERN.match(video_path.stem)
    if match is None:
        return None

    task_slug = match.group(1)
    episode_idx = int(match.group(2))
    candidate_roots = []
    for ancestor in video_path.parents:
        hidden_states_dir = ancestor / "hidden_states"
        if hidden_states_dir.is_dir():
            candidate_roots.append(hidden_states_dir)

    for hidden_states_dir in candidate_roots:
        exact_dir = hidden_states_dir / f"{task_slug}_episode_{episode_idx}"
        if exact_dir.is_dir():
            return exact_dir

        fallback_dirs = sorted(
            hidden_states_dir.glob(f"*_episode_{episode_idx}"))
        if len(fallback_dirs) == 1:
            return fallback_dirs[0]

    return None


def resolve_hidden_dir(hidden_dir: Optional[str], video_path: Path) -> Optional[Path]:
    if hidden_dir is not None:
        return Path(hidden_dir).expanduser().resolve()
    inferred = infer_hidden_dir_from_video(video_path)
    return None if inferred is None else inferred.resolve()


def load_instruction(instruction: Optional[str], hidden_dir: Optional[str], video_path: Path) -> str:
    if instruction is not None:
        return str(instruction)
    hidden_dir_path = resolve_hidden_dir(hidden_dir, video_path)
    if hidden_dir_path is None:
        raise ValueError(
            "Unable to resolve instruction automatically. Provide --instruction or a resolvable --hidden-dir."
        )
    hidden_files = sort_hidden_files(hidden_dir_path)
    payload = torch.load(hidden_files[0][1],
                         map_location="cpu", weights_only=False)
    instructions = payload.get("instructions")
    if not instructions:
        raise ValueError(f"No instructions found in {hidden_files[0][1]}")
    return str(instructions[0])


def read_video(video_path: Path) -> tuple[list[np.ndarray], float]:
    if av is not None:
        container = None
        try:
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]
            fps = 10.0
            if video_stream.average_rate is not None:
                fps = float(video_stream.average_rate)
            elif video_stream.base_rate is not None:
                fps = float(video_stream.base_rate)
            frames = [frame.to_ndarray(format="rgb24")
                      for frame in container.decode(video=0)]
            if not frames:
                raise ValueError(f"Video contains no frames: {video_path}")
            print(
                f"[INFO] Loaded video: {video_path} | frames={len(frames)} | fps={fps:.3f}")
            return frames, fps
        except Exception:
            pass
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

    if imageio is None and cv2 is None:
        raise RuntimeError(
            f"Failed to read video {video_path}: neither PyAV, imageio, nor OpenCV is available in this environment."
        )

    last_error = None
    for format_name in ("FFMPEG", None):
        fps = 10.0
        reader = None
        try:
            reader = imageio.get_reader(video_path, format=format_name)
            try:
                meta = reader.get_meta_data()
                fps_value = meta.get("fps", 10.0) if isinstance(
                    meta, dict) else 10.0
                if fps_value is not None:
                    fps = float(fps_value)
            except Exception:
                fps = 10.0
            frames = [frame for frame in reader]
            if not frames:
                raise ValueError(f"Video contains no frames: {video_path}")
            print(
                f"[INFO] Loaded video: {video_path} | frames={len(frames)} | fps={fps:.3f}")
            return frames, fps
        except Exception as exc:
            last_error = exc
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
    if cv2 is not None:
        capture = cv2.VideoCapture(str(video_path))
        if capture.isOpened():
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 10.0)
                frames = []
                while True:
                    ok, frame_bgr = capture.read()
                    if not ok or frame_bgr is None:
                        break
                    frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                if frames:
                    print(
                        f"[INFO] Loaded video: {video_path} | frames={len(frames)} | fps={fps:.3f}")
                    return frames, fps
            finally:
                capture.release()
    raise RuntimeError(f"Failed to read video {video_path}: {last_error}")


def pad_or_trim_curve(curve: np.ndarray, target_length: int) -> np.ndarray:
    array = np.asarray(curve, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return np.zeros(target_length, dtype=np.float32)
    if array.shape[0] < target_length:
        return np.pad(array, (0, target_length - array.shape[0]), mode="edge")
    if array.shape[0] > target_length:
        return array[:target_length]
    return array


def compute_robometer_curve_signal_official(
    video_frames: np.ndarray,
    instruction: str,
    *,
    model_path: str | Path,
    device: torch.device,
) -> np.ndarray:
    if not str(model_path).strip():
        raise ValueError(
            "Robometer model path is not configured. Edit DEFAULT_ROBOMETER_MODEL_PATH in signal_utils.py."
        )
    ensure_repo_import_path("robometer")
    from robometer.models.rbm import RBM

    if not hasattr(RBM, "all_tied_weights_keys") or not isinstance(getattr(RBM, "all_tied_weights_keys"), dict):
        RBM.all_tied_weights_keys = {}
    if not hasattr(RBM, "_tied_weights_keys"):
        RBM._tied_weights_keys = []

    script_path = Path(__file__).resolve(
    ).parents[4] / "robometer" / "scripts" / "example_inference_local.py"
    module = load_module_from_path(
        "robometer_example_inference_local", script_path)
    rewards, _success_probs = module.compute_rewards_per_frame_local(
        model_path=str(Path(model_path).expanduser().resolve()),
        video_frames=video_frames,
        task=instruction,
        device=device,
    )
    return pad_or_trim_curve(rewards, video_frames.shape[0])


def sample_robometer_video_official(
    video_path: Path,
    *,
    fps: float = 1.0,
    max_frames: int = 64,
) -> tuple[np.ndarray, np.ndarray, int]:
    ensure_repo_import_path("robometer")
    import decord

    vr = decord.VideoReader(str(video_path), num_threads=1)
    total_frames = len(vr)
    if total_frames <= 0:
        raise RuntimeError(f"Video contains no frames: {video_path}")

    try:
        native_fps = float(vr.get_avg_fps())
    except Exception:
        native_fps = 1.0

    effective_fps = fps if fps > 0 else (native_fps if native_fps > 0 else 1.0)
    if native_fps > 0:
        desired_frames = int(
            round(total_frames * (effective_fps / native_fps)))
    else:
        desired_frames = total_frames

    desired_frames = max(1, min(desired_frames, total_frames, max_frames))

    if desired_frames == total_frames:
        frame_indices = np.arange(total_frames, dtype=np.int32)
    else:
        frame_indices = np.linspace(
            0, total_frames - 1, desired_frames, dtype=int).astype(np.int32)

    frames_array = vr.get_batch(
        frame_indices.tolist()).asnumpy().astype(np.uint8)
    print(
        f"[INFO] Robometer sampling: {video_path} | total_frames={total_frames} | "
        f"native_fps={native_fps:.3f} | sampled_frames={frames_array.shape[0]} | target_fps={effective_fps:.3f}"
    )
    return frames_array, frame_indices, total_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute a full offline signal curve and save it as a sidecar .npz."
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=SUPPORTED_EXTERNAL_MODELS, help="External model to run.")
    parser.add_argument("--video-path", type=str,
                        required=True, help="Path to rollout mp4.")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Task instruction. If omitted, infer from --hidden-dir or --video-path.")
    parser.add_argument("--hidden-dir", type=str, default=None,
                        help="Optional hidden_states episode directory for auto-loading the instruction.")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Optional sidecar .npz path. Defaults to <video>_<model>_curve.npz.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                        else "cpu", help="Inference device for models that use torch directly.")
    parser.add_argument("--reference-video-path", type=str,
                        default=None, help="Reference/demo video required by VLAC.")
    parser.add_argument("--vlac-ref-num", type=int, default=6,
                        help="VLAC reference count.")
    parser.add_argument("--vlac-batch-num", type=int, default=5,
                        help="VLAC batch count.")
    parser.add_argument("--vlac-skip", type=int, default=5,
                        help="VLAC temporal skip.")
    parser.add_argument("--vlac-rich", action="store_true",
                        help="Enable VLAC rich mode.")
    parser.add_argument("--vlac-frame-skip", action="store_true",
                        help="Enable VLAC frame-skip mode.")
    parser.add_argument("--vlac-think", action="store_true",
                        help="Enable VLAC think mode.")
    parser.add_argument("--wrist-video-path", type=str, default=None,
                        help="Optional wrist-view rollout for Robo-Dopamine.")
    parser.add_argument("--goal-image-path", type=str, default=None,
                        help="Optional goal image for Robo-Dopamine.")
    parser.add_argument(
        "--robodopamine-mode",
        type=str,
        default="incremental",
        choices=["incremental", "forward", "backward"],
        help="Robo-Dopamine evaluation mode.",
    )
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size for models that expose it.")
    return parser.parse_args()


def save_curve_npz(
    output_path: Path,
    instruction: str,
    steps: np.ndarray,
    values: np.ndarray,
    expanded: np.ndarray,
) -> None:
    np.savez(
        output_path,
        instruction=np.asarray([instruction]),
        steps=np.asarray(steps, dtype=np.int32),
        values=np.asarray(values, dtype=np.float32),
        expanded=np.asarray(expanded, dtype=np.float32),
    )


def main():
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Rollout video not found: {video_path}")

    instruction = load_instruction(
        args.instruction, args.hidden_dir, video_path)
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else default_curve_path(video_path, args.model)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    signal_utils = load_signal_utils_module()

    print(f"[INFO] Computing offline curve: model={args.model}")
    if args.model == "liv":
        primary_frames, _fps = read_video(video_path)
        steps = np.arange(len(primary_frames), dtype=np.int32)
        values = signal_utils.compute_liv_curve_signal(
            primary_frames=primary_frames,
            instruction=instruction,
            device=torch.device(args.device),
            frame_indices=steps,
            batch_size=int(args.batch_size),
        )
        expanded = np.asarray(values, dtype=np.float32)
    elif args.model == "robometer":
        primary_frames, _fps = read_video(video_path)
        values = signal_utils.compute_robometer_curve_signal(
            video_frames=primary_frames,
            instruction=instruction,
            device=torch.device(args.device),
        )
        steps = np.arange(len(values), dtype=np.int32)
        expanded = np.asarray(values, dtype=np.float32)
    elif args.model == "vlac":
        if args.reference_video_path is None:
            raise ValueError("--reference-video-path is required for vlac.")
        primary_frames, fps = read_video(video_path)
        values = signal_utils.compute_vlac_curve_signal(
            video_path=video_path,
            instruction=instruction,
            fps=fps,
            frame_count=len(primary_frames),
            reference_video_path=Path(
                args.reference_video_path).expanduser().resolve(),
            ref_num=int(args.vlac_ref_num),
            batch_num=int(args.vlac_batch_num),
            skip=int(args.vlac_skip),
            rich=bool(args.vlac_rich),
            frame_skip=bool(args.vlac_frame_skip),
            think=bool(args.vlac_think),
            device=torch.device(args.device),
        )
        steps = np.arange(len(values), dtype=np.int32)
        expanded = np.asarray(values, dtype=np.float32)
    elif args.model == "robodopamine":
        values = signal_utils.compute_robodopamine_curve_signal(
            video_path=video_path,
            instruction=instruction,
            wrist_video_path=(
                Path(args.wrist_video_path).expanduser().resolve()
                if args.wrist_video_path is not None
                else None
            ),
            goal_image_path=(
                Path(args.goal_image_path).expanduser().resolve()
                if args.goal_image_path is not None
                else None
            ),
            eval_mode=args.robodopamine_mode,
            batch_size=int(args.batch_size),
        )
        steps = np.arange(len(values), dtype=np.int32)
        expanded = np.asarray(values, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported offline model: {args.model}")

    save_curve_npz(output_path, instruction, steps, values, expanded)
    print("[OK] Offline signal curve complete")
    print(f"  Model: {args.model}")
    print(f"  Video source: {video_path}")
    print(f"  Instruction: {instruction}")
    print(f"  Curve file: {output_path}")


if __name__ == "__main__":
    main()
