from starVLA.model.framework.signal_utils import compute_liv_signal
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import argparse
import re
from pathlib import Path
from typing import Optional

import imageio
import matplotlib
matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams["axes.unicode_minus"] = False

try:
    import cv2 as cv
except ImportError:
    cv = None

try:
    import av
except ImportError:
    av = None


STEP_PATTERN = re.compile(r"step_(\d+)_last_hidden_states_meta\.pt$")


def is_overlay_video(path: Path) -> bool:
    return path.stem.endswith("_liv_overlay")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize LIV curves alongside the corresponding LIBERO rollout video."
    )
    parser.add_argument(
        "--hidden-dir",
        type=str,
        required=True,
        help="Episode hidden_states directory, e.g. .../hidden_states/<task>_episode_<idx>",
    )
    parser.add_argument(
        "--liv-source",
        type=str,
        default="both",
        choices=["saved", "offline", "both"],
        help="Which LIV source to visualize.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output mp4 path. Defaults to the source video path with a _liv_overlay suffix.",
    )
    parser.add_argument(
        "--task-suite",
        type=str,
        default=None,
        help="Optional task suite subdirectory under results/. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for offline LIV computation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for offline LIV computation.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=224,
        help="Width used before offline LIV computation.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=224,
        help="Height used before offline LIV computation.",
    )
    return parser.parse_args()


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


def discover_episode_dirs(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    hidden_states_dir = input_dir / "hidden_states"
    if hidden_states_dir.is_dir():
        input_dir = hidden_states_dir

    if list(input_dir.glob("step_*_last_hidden_states_meta.pt")):
        return [input_dir]

    episode_dirs = []
    for child in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        if list(child.glob("step_*_last_hidden_states_meta.pt")):
            episode_dirs.append(child)

    if not episode_dirs:
        raise FileNotFoundError(
            f"No episode directories with hidden state files found under {input_dir}"
        )
    return episode_dirs


def read_first_payload(hidden_files: list[tuple[int, Path]]) -> dict:
    return torch.load(hidden_files[0][1], map_location="cpu", weights_only=False)


def infer_episode_idx(hidden_dir: Path) -> int:
    match = re.search(r"_episode_(\d+)$", hidden_dir.name)
    if match is None:
        raise ValueError(
            f"Could not infer episode index from hidden directory name: {hidden_dir.name}"
        )
    return int(match.group(1))


def locate_video(
    hidden_dir: Path,
    instruction: str,
    episode_idx: int,
    task_suite: Optional[str],
) -> Path:
    run_root = hidden_dir.parent.parent
    results_root = run_root / "results"
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    if task_suite is not None:
        search_root = results_root / task_suite
        if not search_root.exists():
            raise FileNotFoundError(
                f"Task suite directory not found: {search_root}")
    else:
        task_suite_dirs = [
            path for path in results_root.iterdir() if path.is_dir()]
        if len(task_suite_dirs) != 1:
            raise ValueError(
                f"Could not auto-detect task suite under {results_root}; found {[p.name for p in task_suite_dirs]}"
            )
        search_root = task_suite_dirs[0]

    task_segment = instruction.replace(" ", "_")
    exact_matches = sorted(
        path for path in search_root.glob(f"rollout_{task_segment}_episode{episode_idx}_*.mp4")
        if not is_overlay_video(path)
    )
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(f"Multiple matching videos found: {exact_matches}")

    fallback_matches = sorted(
        path for path in search_root.glob(f"*episode{episode_idx}_*.mp4")
        if not is_overlay_video(path)
    )
    if len(fallback_matches) == 1:
        return fallback_matches[0]
    if len(fallback_matches) > 1:
        raise ValueError(
            f"Could not uniquely resolve video for episode {episode_idx}; candidates: {fallback_matches}"
        )
    raise FileNotFoundError(
        f"No rollout video found under {search_root} for episode {episode_idx}"
    )


def read_video(video_path: Path) -> tuple[list[np.ndarray], float]:
    # LIBERO rollout videos are originally written by imageio.mimwrite(..., fps=10).
    # This post-processing script prefers direct PyAV decoding because imageio's pyav wrapper
    # has been unstable on some environments for these mp4 files.
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

            frames = []
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
            if not frames:
                raise ValueError(f"Video contains no frames: {video_path}")
            return frames, fps
        except Exception:
            pass
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

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
                # LIBERO eval videos are written at 10 fps, so this is a safe fallback.
                fps = 10.0
            frames = [frame for frame in reader]
            if not frames:
                raise ValueError(f"Video contains no frames: {video_path}")
            return frames, fps
        except Exception as exc:
            last_error = exc
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
    raise RuntimeError(f"Failed to read video {video_path}: {last_error}")


def tensor_to_scalar(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return float(arr.reshape(-1)[0])
    return float(value)


def load_saved_signal(hidden_files: list[tuple[int, Path]]) -> tuple[np.ndarray, np.ndarray]:
    steps = []
    values = []
    for step, path in hidden_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        signal_value = tensor_to_scalar(payload.get("signal"))
        if signal_value is None:
            signal_value = tensor_to_scalar(payload.get("liv_signal"))
        if signal_value is None:
            continue
        steps.append(step)
        values.append(signal_value)
    return np.asarray(steps, dtype=np.int32), np.asarray(values, dtype=np.float32)


def staircase_expand(
    steps: np.ndarray,
    values: np.ndarray,
    frame_count: int,
) -> Optional[np.ndarray]:
    if len(steps) == 0 or len(values) == 0:
        return None
    expanded = np.full(frame_count, np.nan, dtype=np.float32)
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


def compute_offline_liv(
    frames: list[np.ndarray],
    instruction: str,
    resize_width: int,
    resize_height: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    all_values = []
    torch_device = torch.device(device)
    # Match the online LIBERO client as closely as possible:
    # 1. resize with OpenCV INTER_AREA when available
    # 2. cast saved values to bf16 on CUDA, which is what online inference commonly uses
    signal_dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
    for start in range(0, len(frames), batch_size):
        chunk = frames[start:start + batch_size]
        batch_images = []
        for frame in chunk:
            if cv is not None:
                resized = cv.resize(
                    frame,
                    (resize_width, resize_height),
                    interpolation=cv.INTER_AREA,
                )
                pil = Image.fromarray(resized)
            else:
                pil = Image.fromarray(frame).resize(
                    (resize_width, resize_height))
            batch_images.append([pil])
        liv_values = compute_liv_signal(
            batch_images=batch_images,
            instructions=[instruction] * len(batch_images),
            device=torch_device,
            dtype=signal_dtype,
        )
        all_values.append(liv_values.detach().cpu().float().numpy())
    return np.concatenate(all_values, axis=0).astype(np.float32)


def resolve_ylim(*arrays: Optional[np.ndarray]) -> tuple[float, float]:
    valid = []
    for arr in arrays:
        if arr is None:
            continue
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            valid.append(arr)
    if not valid:
        return -1.0, 1.0
    merged = np.concatenate(valid)
    low = float(merged.min())
    high = float(merged.max())
    if abs(high - low) < 1e-6:
        margin = 0.1 if abs(low) < 1e-6 else abs(low) * 0.1
        return low - margin, high + margin
    margin = (high - low) * 0.1
    return low - margin, high + margin


def render_plot_frame(
    frame_idx: int,
    frame_count: int,
    offline_values: Optional[np.ndarray],
    saved_steps: np.ndarray,
    saved_values: np.ndarray,
    saved_staircase: Optional[np.ndarray],
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    dpi = 100
    fig = plt.figure(figsize=(panel_width / dpi, panel_height / dpi), dpi=dpi)
    fig.patch.set_facecolor("#f6f3ec")
    ax = fig.add_axes([0.14, 0.16, 0.82, 0.70])
    ax.set_facecolor("#fffdf8")
    x = np.arange(frame_count, dtype=np.int32)

    ylim_low, ylim_high = resolve_ylim(offline_values, saved_staircase)

    if offline_values is not None:
        ax.plot(x, offline_values, color="#e07a1f",
                linewidth=2.0, label="offline LIV")
        current_offline = float(offline_values[frame_idx])
    else:
        current_offline = None

    if saved_staircase is not None:
        ax.step(x, saved_staircase, where="post",
                color="#1f77b4", linewidth=2.0, label="saved LIV")
        ax.scatter(saved_steps, saved_values, color="#1f77b4", s=12, zorder=3)
        current_saved = float(saved_staircase[frame_idx])
    else:
        current_saved = None

    ax.axvline(frame_idx, color="#222222", linestyle="--", linewidth=1.1)
    ax.set_xlim(0, max(frame_count - 1, 1))
    ax.set_ylim(ylim_low, ylim_high)
    ax.set_xlabel("Frame", fontsize=10, labelpad=3)
    ax.set_ylabel("LIV", fontsize=10, labelpad=3)
    ax.set_title("LIV Progress", fontsize=13, pad=4)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.18, color="#7f7f7f")
    for spine in ax.spines.values():
        spine.set_color("#3a3a3a")
        spine.set_linewidth(1.0)
    if offline_values is not None or saved_staircase is not None:
        ax.legend(loc="lower right", fontsize=7, framealpha=0.92)

    text_lines = [f"frame={frame_idx}"]
    if current_offline is not None:
        text_lines.append(f"offline={current_offline:.4f}")
    if current_saved is not None:
        text_lines.append(f"saved={current_saved:.4f}")
    if text_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.35",
                  "facecolor": "white", "alpha": 0.88},
        )

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    plot_image = np.frombuffer(fig.canvas.buffer_rgba(
    ), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    plt.close(fig)
    return plot_image


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    pil = Image.fromarray(image)
    width = max(1, int(round(pil.width * (height / pil.height))))
    return np.asarray(pil.resize((width, height)))


def infer_rollout_status(video_path: Path) -> Optional[str]:
    stem = video_path.stem.lower()
    if stem.endswith("_success"):
        return "success"
    if stem.endswith("_failure"):
        return "failure"
    return None


def get_status_font(target_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_size = max(12, int(round(target_height * 0.038)))
    try:
        font_path = matplotlib.font_manager.findfont("DejaVu Sans")
        return ImageFont.truetype(font_path, size=font_size)
    except Exception:
        return ImageFont.load_default()


def annotate_video_frame(frame_rgb: np.ndarray, status: Optional[str]) -> np.ndarray:
    if status not in {"success", "failure"}:
        return frame_rgb

    color = "#18864b" if status == "success" else "#c53b32"
    text = status.upper()
    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image)
    font = get_status_font(image.height)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w = right - left
    text_h = bottom - top
    pad_x = max(6, text_w // 6)
    pad_y = max(4, text_h // 5)
    margin = max(5, image.height // 44)
    x1 = image.width - text_w - 2 * pad_x - margin
    y1 = margin
    x2 = image.width - margin
    y2 = y1 + text_h + 2 * pad_y
    draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=color)
    draw.text((x1 + pad_x, y1 + pad_y - 1), text, fill="white", font=font)
    return np.asarray(image)


def compose_output_frame(
    plot_frame: np.ndarray,
    video_frame: np.ndarray,
    outer_padding: int = 8,
    panel_gap: int = 10,
) -> np.ndarray:
    canvas_height = max(
        plot_frame.shape[0], video_frame.shape[0]) + outer_padding * 2
    canvas_width = plot_frame.shape[1] + \
        video_frame.shape[1] + panel_gap + outer_padding * 2
    canvas = np.full((canvas_height, canvas_width, 3), 244, dtype=np.uint8)

    plot_y = outer_padding + (canvas_height - 2 *
                              outer_padding - plot_frame.shape[0]) // 2
    video_y = outer_padding + \
        (canvas_height - 2 * outer_padding - video_frame.shape[0]) // 2
    plot_x = outer_padding
    video_x = plot_x + plot_frame.shape[1] + panel_gap

    canvas[plot_y:plot_y + plot_frame.shape[0],
           plot_x:plot_x + plot_frame.shape[1]] = plot_frame
    canvas[video_y:video_y + video_frame.shape[0],
           video_x:video_x + video_frame.shape[1]] = video_frame

    separator_x = plot_x + plot_frame.shape[1] + panel_gap // 2
    canvas[outer_padding:canvas_height - outer_padding,
           separator_x:separator_x + 1] = 223
    return canvas


def create_video_stream_av(container, fps: float):
    # Keep the output codec setup aligned with the repo's existing PyAV video writer pattern.
    # This is only for the visualization overlay video; the original LIBERO rollout mp4s are still
    # produced by eval_libero.py via imageio.mimwrite.
    stream = container.add_stream("h264", rate=max(1, int(round(fps))))
    codec_context = stream.codec_context
    codec_context.pix_fmt = "yuv420p"
    codec_context.options = {"crf": "18", "profile:v": "high"}
    return stream


def write_video_frame_av(container, stream, frame_rgb: np.ndarray) -> None:
    if stream.width == 0 or stream.height == 0:
        h, w, _ = frame_rgb.shape
        stream.width = w
        stream.height = h
    video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
    for packet in stream.encode(video_frame):
        container.mux(packet)


def process_episode(hidden_dir: Path, args, allow_output_override: bool) -> Path:
    hidden_files = sort_hidden_files(hidden_dir)
    first_payload = read_first_payload(hidden_files)

    instructions = first_payload.get("instructions")
    if not instructions:
        raise ValueError(f"No instructions found in {hidden_files[0][1]}")
    instruction = str(instructions[0])
    episode_idx = infer_episode_idx(hidden_dir)

    video_path = locate_video(
        hidden_dir=hidden_dir,
        instruction=instruction,
        episode_idx=episode_idx,
        task_suite=args.task_suite,
    )
    frames, fps = read_video(video_path)
    rollout_status = infer_rollout_status(video_path)

    saved_steps = np.asarray([], dtype=np.int32)
    saved_values = np.asarray([], dtype=np.float32)
    saved_staircase = None
    if args.liv_source in {"saved", "both"}:
        saved_steps, saved_values = load_saved_signal(hidden_files)
        if len(saved_values) == 0 and args.liv_source == "saved":
            raise ValueError(
                f"No saved signal values found in {hidden_dir}; cannot visualize liv_source=saved"
            )
        if len(saved_values) > 0:
            saved_staircase = staircase_expand(
                saved_steps, saved_values, len(frames))

    offline_values = None
    if args.liv_source in {"offline", "both"}:
        offline_values = compute_offline_liv(
            frames=frames,
            instruction=instruction,
            resize_width=args.resize_width,
            resize_height=args.resize_height,
            batch_size=args.batch_size,
            device=args.device,
        )

    if offline_values is None and saved_staircase is None:
        raise ValueError("No LIV values available to visualize.")

    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None and allow_output_override
        else video_path.with_name(f"{video_path.stem}_liv_overlay.mp4")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_height = frames[0].shape[0]
    video_width = frames[0].shape[1]
    # Keep the combined video from becoming too wide, otherwise the player scales
    # the whole frame down and the rollout panel looks smaller on screen.
    plot_width = int(video_width * 1.50)

    # Prefer direct PyAV writing for the overlay video to avoid imageio writer issues observed
    # in this environment. Fall back to imageio only when PyAV is unavailable.
    if av is not None:
        container = av.open(str(output_path), mode="w")
        stream = create_video_stream_av(container, fps)
        try:
            for frame_idx, frame in enumerate(frames):
                plot_frame = render_plot_frame(
                    frame_idx=frame_idx,
                    frame_count=len(frames),
                    offline_values=offline_values,
                    saved_steps=saved_steps,
                    saved_values=saved_values,
                    saved_staircase=saved_staircase,
                    panel_width=plot_width,
                    panel_height=video_height,
                )
                right_frame = annotate_video_frame(
                    frame,
                    rollout_status,
                )
                combined = compose_output_frame(plot_frame, right_frame)
                write_video_frame_av(container, stream, combined)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
    else:
        writer = imageio.get_writer(output_path, fps=fps)
        try:
            for frame_idx, frame in enumerate(frames):
                plot_frame = render_plot_frame(
                    frame_idx=frame_idx,
                    frame_count=len(frames),
                    offline_values=offline_values,
                    saved_steps=saved_steps,
                    saved_values=saved_values,
                    saved_staircase=saved_staircase,
                    panel_width=plot_width,
                    panel_height=video_height,
                )
                right_frame = annotate_video_frame(
                    frame,
                    rollout_status,
                )
                combined = compose_output_frame(plot_frame, right_frame)
                writer.append_data(combined)
        finally:
            writer.close()

    print(f"[OK] {hidden_dir.name}")
    print(f"  Video source: {video_path}")
    print(f"  Output written to: {output_path}")
    return output_path


def main():
    args = parse_args()

    input_dir = Path(args.hidden_dir).expanduser().resolve()
    episode_dirs = discover_episode_dirs(input_dir)
    allow_output_override = len(episode_dirs) == 1

    if args.output_path is not None and not allow_output_override:
        raise ValueError(
            "--output-path can only be used when --hidden-dir points to a single episode directory."
        )

    for idx, hidden_dir in enumerate(episode_dirs, start=1):
        print(f"[{idx}/{len(episode_dirs)}] Processing {hidden_dir.name}")
        process_episode(
            hidden_dir=hidden_dir,
            args=args,
            allow_output_override=allow_output_override,
        )


if __name__ == "__main__":
    main()
