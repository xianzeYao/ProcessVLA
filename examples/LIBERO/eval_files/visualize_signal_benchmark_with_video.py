from starVLA.model.framework.signal_utils import (
    compute_liv_curve_signal,
)
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
from functools import lru_cache
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None
try:
    import cv2
except ImportError:
    cv2 = None


try:
    import av
except ImportError:
    av = None


SUPPORTED_MODELS = ("liv", "robometer", "vlac", "robodopamine")
MODEL_COLORS = {
    "liv": "#e07a1f",
    "robometer": "#c53b32",
    "vlac": "#7a4ea3",
    "robodopamine": "#1f6f8b",
}
STEP_PATTERN = re.compile(r"step_(\d+)_last_hidden_states_meta\.pt$")
ROLLOUT_VIDEO_PATTERN = re.compile(r"^rollout_(.+)_episode(\d+)_")
ROLLOUT_MAIN_VIDEO_PATTERN = re.compile(
    r"^(rollout_.+)_episode(\d+)_(success|failure)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize offline multi-model signal curves alongside the original rollout video."
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to the original rollout mp4 video.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Task instruction text. If omitted, the script will use --hidden-dir or infer it from --video-path.",
    )
    parser.add_argument(
        "--hidden-dir",
        type=str,
        default=None,
        help="Optional hidden_states episode directory for auto-loading the instruction.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["liv"],
        help="Models to visualize. Use 'all' or any subset of: liv robometer vlac robodopamine.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output mp4 path. Defaults to <video>_signals_overlay.mp4.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for model inference.",
    )
    parser.add_argument(
        "--plot-layout",
        type=str,
        default="sameplot",
        choices=["sameplot", "subplot"],
        help="Plot all model curves on one axis ('sameplot') or render one subplot per model ('subplot').",
    )
    return parser.parse_args()


@lru_cache(maxsize=1)
def ensure_matplotlib_runtime():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams["axes.unicode_minus"] = False
    return matplotlib, plt, FormatStrFormatter


def normalize_models(raw_models: Sequence[str]) -> List[str]:
    parsed = []
    for item in raw_models:
        for piece in str(item).split(","):
            piece = piece.strip().lower()
            if piece:
                parsed.append(piece)
    if not parsed:
        raise ValueError("At least one model must be provided via --models.")
    if "all" in parsed:
        return list(SUPPORTED_MODELS)
    invalid = sorted(set(parsed) - set(SUPPORTED_MODELS))
    if invalid:
        raise ValueError(
            f"Unsupported models: {invalid}. Supported: {SUPPORTED_MODELS}.")
    deduped = []
    for model in parsed:
        if model not in deduped:
            deduped.append(model)
    return deduped


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load module {module_name!r} from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_repo_import_path(repo_name: str, src_subdir: Optional[str] = None) -> None:
    workspace_root = Path(__file__).resolve().parents[4]
    repo_root = workspace_root / repo_name
    if not repo_root.exists():
        raise ImportError(
            f"Local repo '{repo_name}' not found under {workspace_root}.")
    import_path = repo_root / src_subdir if src_subdir is not None else repo_root
    path_str = str(import_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


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


def align_auxiliary_frames(
    primary_frames: list[np.ndarray],
    auxiliary_frames: Optional[list[np.ndarray]],
    name: str,
) -> Optional[list[np.ndarray]]:
    if auxiliary_frames is None:
        return None
    if len(auxiliary_frames) == len(primary_frames):
        return auxiliary_frames
    new_length = min(len(primary_frames), len(auxiliary_frames))
    if new_length <= 0:
        raise ValueError(f"{name} contains no frames.")
    print(
        f"[WARN] Frame count mismatch for {name}: primary={len(primary_frames)} auxiliary={len(auxiliary_frames)}. "
        f"Truncating both to {new_length}."
    )
    del primary_frames[new_length:]
    return auxiliary_frames[:new_length]


def infer_auxiliary_video_paths(video_path: Path) -> list[tuple[str, Path]]:
    match = ROLLOUT_MAIN_VIDEO_PATTERN.match(video_path.stem)
    if match is None:
        return []

    base_prefix, episode_idx, status = match.groups()
    auxiliary_paths: list[tuple[str, Path]] = []
    pattern = f"{base_prefix}_*_episode{episode_idx}_{status}.mp4"
    suffix = f"_episode{episode_idx}_{status}"
    for candidate in sorted(video_path.parent.glob(pattern)):
        if candidate == video_path:
            continue
        if candidate.stem.endswith("_signals_overlay"):
            continue
        raw_label = candidate.stem[len(base_prefix) + 1: -len(suffix)]
        if not raw_label:
            continue
        auxiliary_paths.append((raw_label, candidate))
    return auxiliary_paths


def humanize_camera_label(raw_label: str) -> str:
    words = [part for part in str(raw_label).replace(
        "-", "_").split("_") if part]
    if not words:
        return "Auxiliary View"
    label = " ".join(word.upper() if len(word) <= 2 else word.capitalize()
                     for word in words)
    if "view" not in label.lower():
        label = f"{label} View"
    return label


def truncate_video_streams(
    primary_frames: list[np.ndarray],
    auxiliary_streams: list[tuple[str, list[np.ndarray], Optional[str]]],
) -> tuple[list[np.ndarray], list[tuple[str, list[np.ndarray], Optional[str]]]]:
    if not auxiliary_streams:
        return primary_frames, auxiliary_streams

    target_length = min(
        [len(primary_frames)] + [len(frames)
                                 for _, frames, _ in auxiliary_streams]
    )
    if target_length <= 0:
        raise ValueError("Auxiliary video alignment produced zero frames.")

    if len(primary_frames) != target_length or any(len(frames) != target_length for _, frames, _ in auxiliary_streams):
        print(
            f"[WARN] Video frame count mismatch across views. Truncating all streams to {target_length} frames."
        )
    primary_frames = primary_frames[:target_length]
    auxiliary_streams = [
        (label, frames[:target_length], status)
        for label, frames, status in auxiliary_streams
    ]
    return primary_frames, auxiliary_streams


def infer_rollout_status(video_path: Path) -> Optional[str]:
    stem = video_path.stem.lower()
    if stem.endswith("_success"):
        return "success"
    if stem.endswith("_failure"):
        return "failure"
    return None


def get_status_font(target_height: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    matplotlib, _, _ = ensure_matplotlib_runtime()
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


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        left, top, right, bottom = draw.textbbox((0, 0), candidate, font=font)
        if right - left <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_camera_panel(
    video_frame: np.ndarray,
    title: str,
    instruction: Optional[str] = None,
    *,
    footer_height: Optional[int] = None,
) -> np.ndarray:
    image = Image.fromarray(video_frame)
    width, height = image.size
    if instruction:
        footer_height = footer_height or max(54, int(round(height * 0.18)))
    else:
        footer_height = footer_height or max(26, int(round(height * 0.10)))
    panel = Image.new("RGB", (width, height + footer_height),
                      color=(244, 244, 244))
    panel.paste(image, (0, 0))

    draw = ImageDraw.Draw(panel)
    label_font = get_status_font(footer_height)
    body_font = get_status_font(max(footer_height - 10, 24))
    padding_x = max(10, width // 36)
    padding_y = max(7, footer_height // 8)
    footer_y = height

    draw.rectangle((0, footer_y, width, height + footer_height),
                   fill=(248, 246, 241))
    draw.line((0, footer_y, width, footer_y), fill=(223, 223, 223), width=1)
    draw.text((padding_x, footer_y + padding_y), title,
              fill=(56, 56, 56), font=label_font)

    if instruction:
        label_bottom = draw.textbbox(
            (padding_x, footer_y + padding_y), title, font=label_font)[3]
        available_width = width - padding_x * 2
        wrapped_lines = wrap_text_to_width(
            draw, instruction, body_font, available_width)
        text_y = label_bottom + max(4, padding_y // 2)
        sample_bbox = draw.textbbox((0, 0), "Ag", font=body_font)
        line_height = max(14, sample_bbox[3] - sample_bbox[1] + 2)
        max_lines = max(1, (height + footer_height -
                        text_y - padding_y) // line_height)
        all_lines = wrapped_lines
        wrapped_lines = wrapped_lines[:max_lines]
        if wrapped_lines and len(wrapped_lines) < len(all_lines):
            wrapped_lines[-1] = wrapped_lines[-1].rstrip(". ") + "..."
        for line in wrapped_lines:
            draw.text((padding_x, text_y), line,
                      fill=(28, 28, 28), font=body_font)
            text_y += line_height

    return np.asarray(panel)


def render_camera_stack_panel(
    primary_frame: np.ndarray,
    instruction: str,
    auxiliary_frames: Sequence[tuple[str, np.ndarray]],
) -> np.ndarray:
    panels = [
        render_camera_panel(
            primary_frame,
            title="Primary View",
            instruction=instruction,
        )
    ]
    for label, frame in auxiliary_frames:
        panels.append(
            render_camera_panel(
                frame,
                title=label,
                instruction=None,
            )
        )

    panel_gap = 8
    width = max(panel.shape[1] for panel in panels)
    height = sum(panel.shape[0] for panel in panels) + \
        panel_gap * max(0, len(panels) - 1)
    stack = np.full((height, width, 3), 244, dtype=np.uint8)

    cursor_y = 0
    for panel in panels:
        x = (width - panel.shape[1]) // 2
        stack[cursor_y:cursor_y + panel.shape[0], x:x + panel.shape[1]] = panel
        cursor_y += panel.shape[0] + panel_gap
    return stack


def compose_output_frame(
    plot_frame: np.ndarray,
    video_panel: np.ndarray,
    outer_padding: int = 8,
    panel_gap: int = 10,
) -> np.ndarray:
    canvas_height = max(
        plot_frame.shape[0], video_panel.shape[0]) + outer_padding * 2
    canvas_width = plot_frame.shape[1] + \
        video_panel.shape[1] + panel_gap + outer_padding * 2
    canvas = np.full((canvas_height, canvas_width, 3), 244, dtype=np.uint8)

    plot_y = outer_padding + (canvas_height - 2 *
                              outer_padding - plot_frame.shape[0]) // 2
    video_y = outer_padding + \
        (canvas_height - 2 * outer_padding - video_panel.shape[0]) // 2
    plot_x = outer_padding
    video_x = plot_x + plot_frame.shape[1] + panel_gap

    canvas[plot_y:plot_y + plot_frame.shape[0],
           plot_x:plot_x + plot_frame.shape[1]] = plot_frame
    canvas[video_y:video_y + video_panel.shape[0],
           video_x:video_x + video_panel.shape[1]] = video_panel

    separator_x = plot_x + plot_frame.shape[1] + panel_gap // 2
    canvas[outer_padding:canvas_height - outer_padding,
           separator_x:separator_x + 1] = 223
    return canvas


def create_video_stream_av(container, fps: float):
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


def default_curve_path(video_path: Path, model_name: str) -> Path:
    return video_path.with_name(f"{video_path.stem}_{model_name}_curve.npz")


def load_external_curve(
    curve_path: Path,
    frame_count: int,
    sampled_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not curve_path.exists():
        raise FileNotFoundError(
            f"Missing precomputed curve for external model: {curve_path}")
    payload = np.load(curve_path)
    if "expanded" not in payload:
        raise ValueError(
            f"Curve file must contain 'expanded': {curve_path}")
    expanded = np.asarray(payload["expanded"], dtype=np.float32)
    if expanded.shape[0] < frame_count:
        expanded = np.pad(expanded, (0, frame_count -
                          expanded.shape[0]), mode="edge")
    elif expanded.shape[0] > frame_count:
        expanded = expanded[:frame_count]

    if "steps" in payload and "values" in payload:
        steps = np.asarray(payload["steps"], dtype=np.int32)
        values = np.asarray(payload["values"], dtype=np.float32)
    else:
        steps = sampled_indices.copy()
        values = expanded[sampled_indices].astype(np.float32)
    return steps, expanded.astype(np.float32), values.astype(np.float32)


def select_frame_indices(frame_count: int, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("'compute_stride' must be positive.")
    indices = list(range(0, frame_count, stride))
    if not indices or indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return np.asarray(sorted(set(indices)), dtype=np.int32)


def resolve_ylim(curves: Dict[str, np.ndarray]) -> tuple[float, float]:
    valid = []
    for values in curves.values():
        arr = values[np.isfinite(values)]
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


def style_plot_axis(
    ax,
    frame_idx: int,
    frame_count: int,
    format_str_formatter,
    *,
    ylim_low: float,
    ylim_high: float,
    show_xlabel: bool,
    ylabel: str = "Signal",
    title: Optional[str] = None,
) -> None:
    ax.axvline(frame_idx, color="#222222", linestyle="--", linewidth=1.1)
    ax.set_xlim(0, max(frame_count - 1, 1))
    ax.set_ylim(ylim_low, ylim_high)
    if show_xlabel:
        ax.set_xlabel("Frame", fontsize=10, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=3)
    if title is not None:
        ax.set_title(title, fontsize=11, pad=4)
    ax.yaxis.set_major_formatter(format_str_formatter("%.3f"))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.18, color="#7f7f7f")
    for spine in ax.spines.values():
        spine.set_color("#3a3a3a")
        spine.set_linewidth(1.0)


def render_plot_frame(
    frame_idx: int,
    frame_count: int,
    expanded_curves: Dict[str, np.ndarray],
    sampled_curves: Dict[str, tuple[np.ndarray, np.ndarray]],
    panel_width: int,
    panel_height: int,
    plot_layout: str = "sameplot",
) -> np.ndarray:
    _, plt, FormatStrFormatter = ensure_matplotlib_runtime()
    dpi = 100
    x = np.arange(frame_count, dtype=np.int32)
    fig = plt.figure(figsize=(panel_width / dpi, panel_height / dpi), dpi=dpi)
    fig.patch.set_facecolor("#f6f3ec")
    text_lines = [f"frame={frame_idx}"]

    if plot_layout == "subplot":
        model_items = list(expanded_curves.items())
        axes = fig.subplots(
            len(model_items),
            1,
            sharex=True,
            squeeze=False,
            gridspec_kw={"left": 0.12, "right": 0.96,
                         "top": 0.88, "bottom": 0.08, "hspace": 0.32},
        )[:, 0]
        fig.suptitle("Offline Signal Benchmark", fontsize=13, y=0.96)
        for axis_idx, (ax, (model_name, values)) in enumerate(zip(axes, model_items)):
            ax.set_facecolor("#fffdf8")
            color = MODEL_COLORS.get(model_name, "#333333")
            sampled_steps, sampled_values = sampled_curves[model_name]
            ax.plot(x, values, color=color, linewidth=2.0)
            if len(sampled_steps) > 1:
                ax.scatter(sampled_steps, sampled_values,
                           color=color, s=10, zorder=3, alpha=0.85)
            current = values[frame_idx]
            if np.isfinite(current):
                text_lines.append(f"{model_name}={float(current):.4f}")
            ylim_low, ylim_high = resolve_ylim({model_name: values})
            style_plot_axis(
                ax,
                frame_idx,
                frame_count,
                FormatStrFormatter,
                ylim_low=ylim_low,
                ylim_high=ylim_high,
                show_xlabel=axis_idx == len(model_items) - 1,
                title=model_name,
            )
        axes[0].text(
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.35",
                  "facecolor": "white", "alpha": 0.88},
        )
    else:
        ax = fig.add_axes([0.12, 0.16, 0.84, 0.70])
        ax.set_facecolor("#fffdf8")
        ylim_low, ylim_high = resolve_ylim(expanded_curves)

        for model_name, values in expanded_curves.items():
            color = MODEL_COLORS.get(model_name, "#333333")
            ax.plot(x, values, color=color, linewidth=2.0, label=model_name)
            sampled_steps, sampled_values = sampled_curves[model_name]
            if len(sampled_steps) > 1:
                ax.scatter(sampled_steps, sampled_values,
                           color=color, s=10, zorder=3, alpha=0.85)
            current = values[frame_idx]
            if np.isfinite(current):
                text_lines.append(f"{model_name}={float(current):.4f}")

        style_plot_axis(
            ax,
            frame_idx,
            frame_count,
            FormatStrFormatter,
            ylim_low=ylim_low,
            ylim_high=ylim_high,
            show_xlabel=True,
            title="Offline Signal Benchmark",
        )
        ax.legend(loc="lower right", fontsize=7, framealpha=0.92)
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


def save_signals_npz(
    output_path: Path,
    instruction: str,
    sampled_curves: Dict[str, tuple[np.ndarray, np.ndarray]],
    expanded_curves: Dict[str, np.ndarray],
) -> None:
    arrays: Dict[str, Any] = {"instruction": np.asarray([instruction])}
    for model_name, (steps, values) in sampled_curves.items():
        arrays[f"{model_name}_steps"] = steps.astype(np.int32)
        arrays[f"{model_name}_values"] = values.astype(np.float32)
        arrays[f"{model_name}_expanded"] = expanded_curves[model_name].astype(
            np.float32)
    np.savez(output_path, **arrays)


def main():
    args = parse_args()
    models = normalize_models(args.models)

    video_path = Path(args.video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Rollout video not found: {video_path}")

    instruction = load_instruction(
        args.instruction, args.hidden_dir, video_path)
    primary_frames, fps = read_video(video_path)
    auxiliary_streams: list[tuple[str, list[np.ndarray], Optional[str]]] = []
    for raw_label, auxiliary_path in infer_auxiliary_video_paths(video_path):
        auxiliary_frames, _aux_fps = read_video(auxiliary_path)
        auxiliary_streams.append(
            (
                humanize_camera_label(raw_label),
                auxiliary_frames,
                infer_rollout_status(auxiliary_path),
            )
        )
    primary_frames, auxiliary_streams = truncate_video_streams(
        primary_frames, auxiliary_streams)
    device = torch.device(args.device)
    rollout_status = infer_rollout_status(video_path)
    compute_stride = 1
    batch_size = 10
    plot_layout = args.plot_layout
    sampled_indices = select_frame_indices(
        len(primary_frames), compute_stride)

    sampled_curves: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expanded_curves: Dict[str, np.ndarray] = {}

    for model_name in models:
        if model_name != "liv":
            continue

        curve_path = default_curve_path(video_path, model_name)
        if curve_path.exists():
            print(f"[INFO] Loading precomputed curve for {model_name}...")
            steps, expanded, values = load_external_curve(
                curve_path=curve_path,
                frame_count=len(primary_frames),
                sampled_indices=sampled_indices,
            )
        else:
            print(f"[INFO] Computing {model_name}...")
            values = compute_liv_curve_signal(
                primary_frames=primary_frames,
                instruction=instruction,
                device=device,
                frame_indices=sampled_indices,
                batch_size=batch_size,
            )
            steps = sampled_indices.copy()
            expanded = staircase_expand(steps, values, len(primary_frames))

        sampled_curves[model_name] = (
            np.asarray(steps, dtype=np.int32),
            np.asarray(values, dtype=np.float32),
        )
        expanded_curves[model_name] = np.asarray(expanded, dtype=np.float32)

    for model_name in models:
        if model_name == "liv":
            continue
        print(f"[INFO] Loading external curve for {model_name}...")
        curve_path = default_curve_path(video_path, model_name)
        steps, expanded, values = load_external_curve(
            curve_path=curve_path,
            frame_count=len(primary_frames),
            sampled_indices=sampled_indices,
        )
        sampled_curves[model_name] = (steps, values)
        expanded_curves[model_name] = expanded

    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else video_path.with_name(f"{video_path.stem}_signals_overlay.mp4")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    signals_path = output_path.with_suffix(".npz")
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    save_signals_npz(signals_path, instruction,
                     sampled_curves, expanded_curves)

    video_height = primary_frames[0].shape[0]
    video_width = primary_frames[0].shape[1]
    plot_width = int(video_width * 1.55)
    if plot_layout == "subplot":
        plot_height = max(video_height, int(
            video_height * max(1.0, 0.68 * len(models))))
    else:
        plot_height = video_height

    if av is not None:
        container = av.open(str(output_path), mode="w")
        stream = create_video_stream_av(container, fps)
        try:
            for frame_idx, frame in enumerate(primary_frames):
                plot_frame = render_plot_frame(
                    frame_idx=frame_idx,
                    frame_count=len(primary_frames),
                    expanded_curves=expanded_curves,
                    sampled_curves=sampled_curves,
                    panel_width=plot_width,
                    panel_height=plot_height,
                    plot_layout=plot_layout,
                )
                auxiliary_panels = [
                    (
                        label,
                        annotate_video_frame(
                            frames[frame_idx], status or rollout_status),
                    )
                    for label, frames, status in auxiliary_streams
                ]
                right_frame = annotate_video_frame(frame, rollout_status)
                right_panel = render_camera_stack_panel(
                    right_frame, instruction, auxiliary_panels)
                combined = compose_output_frame(plot_frame, right_panel)
                write_video_frame_av(container, stream, combined)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
    else:
        if imageio is None:
            raise RuntimeError(
                "PyAV is unavailable and imageio is not installed, cannot write overlay video.")
        writer = imageio.get_writer(output_path, fps=fps)
        try:
            for frame_idx, frame in enumerate(primary_frames):
                plot_frame = render_plot_frame(
                    frame_idx=frame_idx,
                    frame_count=len(primary_frames),
                    expanded_curves=expanded_curves,
                    sampled_curves=sampled_curves,
                    panel_width=plot_width,
                    panel_height=plot_height,
                    plot_layout=plot_layout,
                )
                auxiliary_panels = [
                    (
                        label,
                        annotate_video_frame(
                            frames[frame_idx], status or rollout_status),
                    )
                    for label, frames, status in auxiliary_streams
                ]
                right_frame = annotate_video_frame(frame, rollout_status)
                right_panel = render_camera_stack_panel(
                    right_frame, instruction, auxiliary_panels)
                combined = compose_output_frame(plot_frame, right_panel)
                writer.append_data(combined)
        finally:
            writer.close()

    print("[OK] Signal visualization complete")
    print(f"  Video source: {video_path}")
    print(f"  Instruction: {instruction}")
    print(f"  Models: {models}")
    print(f"  Plot layout: {plot_layout}")
    print(f"  Overlay video: {output_path}")
    print(f"  Signal dump: {signals_path}")


if __name__ == "__main__":
    main()
