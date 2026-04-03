import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

try:
    import av
except ImportError:
    av = None


ANSI_ESCAPE_PATTERN = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
EPISODE_LOG_PATTERN = re.compile(r"task(\d+)_ep(\d+)\.log$")
SUCCESS_PATTERN = re.compile(r"Success:\s*(True|False)")
VLAC_STEP_PATTERN = re.compile(
    r"\*\*\*\s*vlac_online step=(\d+)\s+critic=([+-]?\d+(?:\.\d+)?)\s+signal=([+-]?\d+(?:\.\d+)?)\s+reference=(.*?)\s+\*\*\*"
)
ROLLOUT_MAIN_VIDEO_PATTERN = re.compile(
    r"^(rollout_.+)_episode(\d+)_(success|failure)$"
)


def ensure_matplotlib_runtime():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams["axes.unicode_minus"] = False
    return matplotlib, plt, FormatStrFormatter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize chunk-level VLAC online signal curves alongside the corresponding LIBERO rollout videos."
    )
    parser.add_argument(
        "--log-path",
        type=str,
        required=True,
        help="Path to an isolated episode log, e.g. .../_episode_logs/task000_ep016.log",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output mp4 path. Defaults to <main_rollout>_vlac_online_overlay.mp4.",
    )
    parser.add_argument(
        "--main-video-path",
        type=str,
        default=None,
        help="Optional override for the main rollout video path.",
    )
    parser.add_argument(
        "--wrist-video-path",
        type=str,
        default=None,
        help="Optional override for the wrist rollout video path.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional override for the task instruction. Useful when the log's wrapped Task line is hard to parse.",
    )
    parser.add_argument(
        "--plot-width-scale",
        type=float,
        default=1.55,
        help="Left plot panel width relative to the main video width.",
    )
    return parser.parse_args()


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_PATTERN.sub("", text)


def read_text_lines(path: Path) -> List[str]:
    return [strip_ansi(line.rstrip("\n")) for line in path.read_text(encoding="utf-8").splitlines()]


def parse_episode_identifiers(log_path: Path) -> Tuple[int, int]:
    match = EPISODE_LOG_PATTERN.fullmatch(log_path.name)
    if match is None:
        raise ValueError(
            f"Expected log filename like task000_ep016.log, got: {log_path.name}"
        )
    return int(match.group(1)), int(match.group(2))


def parse_success_flag(lines: Sequence[str]) -> Optional[bool]:
    for line in lines:
        match = SUCCESS_PATTERN.search(line)
        if match is not None:
            return match.group(1) == "True"
    return None


def is_task_continuation_line(text: str) -> bool:
    if not text:
        return False
    blockers = (
        "Starting episode",
        "About to env.step",
        "Success:",
        "Episode end reason",
        "Current task success rate",
        "Current total success rate",
        "Total success rate",
        "Total episodes",
        "***",
        "[",
        "0%|",
        "100%|",
    )
    if any(token in text for token in blockers):
        return False
    if "INFO" in text and "eval_libero.py" in text:
        return False
    if "WARNING" in text:
        return False
    return True


def parse_instruction(lines: Sequence[str]) -> Optional[str]:
    for idx, line in enumerate(lines):
        if "Task suite:" in line:
            continue
        if "Task:" not in line:
            continue
        task_text = line.split("Task:", 1)[1].strip()
        parts = [task_text] if task_text else []
        cursor = idx + 1
        while cursor < len(lines):
            stripped = lines[cursor].strip()
            if not is_task_continuation_line(stripped):
                break
            parts.append(stripped)
            cursor += 1
        instruction = " ".join(part for part in parts if part)
        instruction = re.sub(r"\s+", " ", instruction).strip()
        if instruction:
            return instruction
    return None


def parse_vlac_entries(lines: Sequence[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for line in lines:
        match = VLAC_STEP_PATTERN.search(line)
        if match is None:
            continue
        entries.append(
            {
                "step": int(match.group(1)),
                "critic": float(match.group(2)),
                "signal": float(match.group(3)),
                "reference_video_path": match.group(4).strip(),
            }
        )
    if not entries:
        raise ValueError("No VLAC online step entries found in the log.")
    return entries


def infer_rollout_status(video_path: Path) -> Optional[str]:
    stem = video_path.stem.lower()
    if stem.endswith("_success"):
        return "success"
    if stem.endswith("_failure"):
        return "failure"
    return None


def locate_rollout_videos(
    *,
    log_path: Path,
    instruction: str,
    episode_idx: int,
    success_flag: Optional[bool],
    main_override: Optional[Path],
    wrist_override: Optional[Path],
) -> Tuple[Path, Optional[Path]]:
    if main_override is not None and not main_override.exists():
        raise FileNotFoundError(f"Main video override not found: {main_override}")
    if wrist_override is not None and not wrist_override.exists():
        raise FileNotFoundError(f"Wrist video override not found: {wrist_override}")
    if main_override is not None:
        return main_override, wrist_override

    rollout_root = log_path.parent.parent
    if not rollout_root.exists():
        raise FileNotFoundError(f"Rollout root not found: {rollout_root}")

    task_segment = instruction.replace(" ", "_")
    status_candidates = []
    if success_flag is True:
        status_candidates.append("success")
    elif success_flag is False:
        status_candidates.append("failure")
    status_candidates.extend(["success", "failure"])

    main_candidates: List[Path] = []
    wrist_candidates: List[Path] = []
    for status in list(dict.fromkeys(status_candidates)):
        exact_main = sorted(
            path
            for path in rollout_root.glob(
                f"rollout_{task_segment}_episode{episode_idx}_{status}.mp4"
            )
            if "_wrist_" not in path.name
        )
        exact_wrist = sorted(
            rollout_root.glob(
                f"rollout_{task_segment}_wrist_episode{episode_idx}_{status}.mp4"
            )
        )
        if len(exact_main) > 1:
            raise ValueError(f"Multiple main rollout candidates found: {exact_main}")
        if len(exact_wrist) > 1:
            raise ValueError(f"Multiple wrist rollout candidates found: {exact_wrist}")
        if len(exact_main) == 1:
            return exact_main[0], exact_wrist[0] if exact_wrist else None
        main_candidates.extend(exact_main)
        wrist_candidates.extend(exact_wrist)

    if not main_candidates:
        fallback_main = sorted(
            path
            for path in rollout_root.glob(f"rollout_*_episode{episode_idx}_*.mp4")
            if "_wrist_" not in path.name
        )
        if len(fallback_main) == 1:
            main_candidates = fallback_main
        else:
            raise FileNotFoundError(
                "Unable to uniquely locate the main rollout video. "
                f"Looked under {rollout_root} for task={instruction!r}, episode_idx={episode_idx}."
            )
    if len(main_candidates) != 1:
        raise ValueError(f"Multiple main rollout candidates found: {main_candidates}")

    main_video_path = main_candidates[0]

    if wrist_override is not None:
        return main_video_path, wrist_override
    wrist_candidates = sorted(dict.fromkeys(wrist_candidates))
    if wrist_candidates:
        if len(wrist_candidates) != 1:
            raise ValueError(f"Multiple wrist rollout candidates found: {wrist_candidates}")
        return main_video_path, wrist_candidates[0]

    derived_wrist = main_video_path.with_name(
        main_video_path.name.replace(
            f"_episode{episode_idx}_", f"_wrist_episode{episode_idx}_", 1
        )
    )
    if derived_wrist.exists():
        return main_video_path, derived_wrist
    return main_video_path, None


def read_video(video_path: Path) -> Tuple[List[np.ndarray], float]:
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
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
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

    if imageio is None:
        raise RuntimeError(
            f"Failed to read video {video_path}: neither PyAV nor imageio is available."
        )

    last_error = None
    for format_name in ("FFMPEG", None):
        fps = 10.0
        reader = None
        try:
            reader = imageio.get_reader(video_path, format=format_name)
            try:
                meta = reader.get_meta_data()
                fps_value = meta.get("fps", 10.0) if isinstance(meta, dict) else 10.0
                if fps_value is not None:
                    fps = float(fps_value)
            except Exception:
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


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    pil = Image.fromarray(image)
    height = max(1, int(round(pil.height * (width / pil.width))))
    return np.asarray(pil.resize((width, height)))


def get_status_font(target_height: int) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    matplotlib, _, _ = ensure_matplotlib_runtime()
    font_size = max(12, int(round(target_height * 0.038)))
    try:
        font_path = matplotlib.font_manager.findfont("DejaVu Sans")
        return ImageFont.truetype(font_path, size=font_size)
    except Exception:
        return ImageFont.load_default()


def wrap_text_to_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont],
    max_width: int,
) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: List[str] = []
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
    panel = Image.new("RGB", (width, height + footer_height), color=(244, 244, 244))
    panel.paste(image, (0, 0))

    draw = ImageDraw.Draw(panel)
    label_font = get_status_font(footer_height)
    body_font = get_status_font(max(footer_height - 10, 24))
    padding_x = max(10, width // 36)
    padding_y = max(7, footer_height // 8)
    footer_y = height

    draw.rectangle((0, footer_y, width, height + footer_height), fill=(248, 246, 241))
    draw.line((0, footer_y, width, footer_y), fill=(223, 223, 223), width=1)
    draw.text((padding_x, footer_y + padding_y), title, fill=(56, 56, 56), font=label_font)

    if instruction:
        label_bottom = draw.textbbox((padding_x, footer_y + padding_y), title, font=label_font)[3]
        available_width = width - padding_x * 2
        wrapped_lines = wrap_text_to_width(draw, instruction, body_font, available_width)
        text_y = label_bottom + max(4, padding_y // 2)
        sample_bbox = draw.textbbox((0, 0), "Ag", font=body_font)
        line_height = max(14, sample_bbox[3] - sample_bbox[1] + 2)
        max_lines = max(1, (height + footer_height - text_y - padding_y) // line_height)
        all_lines = wrapped_lines
        wrapped_lines = wrapped_lines[:max_lines]
        if wrapped_lines and len(wrapped_lines) < len(all_lines):
            wrapped_lines[-1] = wrapped_lines[-1].rstrip(". ") + "..."
        for line in wrapped_lines:
            draw.text((padding_x, text_y), line, fill=(28, 28, 28), font=body_font)
            text_y += line_height

    return np.asarray(panel)


def render_camera_stack_panel(
    primary_frame: np.ndarray,
    instruction: str,
    auxiliary_frames: Sequence[Tuple[str, np.ndarray]],
) -> np.ndarray:
    panels = [
        render_camera_panel(
            primary_frame,
            title="Rollout Main View",
            instruction=instruction,
        )
    ]
    for label, frame in auxiliary_frames:
        panels.append(render_camera_panel(frame, title=label, instruction=None))

    panel_gap = 8
    width = max(panel.shape[1] for panel in panels)
    height = sum(panel.shape[0] for panel in panels) + panel_gap * max(0, len(panels) - 1)
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
    canvas_height = max(plot_frame.shape[0], video_panel.shape[0]) + outer_padding * 2
    canvas_width = plot_frame.shape[1] + video_panel.shape[1] + panel_gap + outer_padding * 2
    canvas = np.full((canvas_height, canvas_width, 3), 244, dtype=np.uint8)

    plot_y = outer_padding + (canvas_height - 2 * outer_padding - plot_frame.shape[0]) // 2
    video_y = outer_padding + (canvas_height - 2 * outer_padding - video_panel.shape[0]) // 2
    plot_x = outer_padding
    video_x = plot_x + plot_frame.shape[1] + panel_gap

    canvas[plot_y:plot_y + plot_frame.shape[0], plot_x:plot_x + plot_frame.shape[1]] = plot_frame
    canvas[video_y:video_y + video_panel.shape[0], video_x:video_x + video_panel.shape[1]] = video_panel

    separator_x = plot_x + plot_frame.shape[1] + panel_gap // 2
    canvas[outer_padding:canvas_height - outer_padding, separator_x:separator_x + 1] = 223
    return canvas


def create_video_stream_av(container, fps: float):
    stream = container.add_stream("h264", rate=max(1, int(round(fps))))
    codec_context = stream.codec_context
    codec_context.pix_fmt = "yuv420p"
    codec_context.options = {"crf": "18", "profile:v": "high"}
    return stream


def write_video_frame_av(container, stream, frame_rgb: np.ndarray) -> None:
    if stream.width == 0 or stream.height == 0:
        height, width, _ = frame_rgb.shape
        stream.width = width
        stream.height = height
    video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
    for packet in stream.encode(video_frame):
        container.mux(packet)


def truncate_video_streams(
    primary_frames: List[np.ndarray],
    wrist_frames: Optional[List[np.ndarray]],
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
    if wrist_frames is None:
        return primary_frames, wrist_frames
    if len(primary_frames) == len(wrist_frames):
        return primary_frames, wrist_frames
    target_length = min(len(primary_frames), len(wrist_frames))
    if target_length <= 0:
        raise ValueError("Video alignment produced zero frames.")
    return primary_frames[:target_length], wrist_frames[:target_length]


def staircase_expand(steps: np.ndarray, values: np.ndarray, frame_count: int) -> np.ndarray:
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


def resolve_ylim(*arrays: np.ndarray) -> Tuple[float, float]:
    valid = []
    for arr in arrays:
        values = np.asarray(arr, dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size > 0:
            valid.append(values)
    if not valid:
        return -1.0, 1.0
    merged = np.concatenate(valid)
    low = float(merged.min())
    high = float(merged.max())
    if abs(high - low) < 1e-6:
        margin = 0.1 if abs(low) < 1e-6 else abs(low) * 0.1
        return low - margin, high + margin
    margin = (high - low) * 0.12
    return low - margin, high + margin


def render_plot_frame(
    *,
    frame_idx: int,
    frame_count: int,
    steps: np.ndarray,
    critics: np.ndarray,
    signals: np.ndarray,
    signal_staircase: np.ndarray,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    _, plt, FormatStrFormatter = ensure_matplotlib_runtime()
    dpi = 100
    fig = plt.figure(figsize=(panel_width / dpi, panel_height / dpi), dpi=dpi)
    fig.patch.set_facecolor("#f6f3ec")
    ax = fig.add_axes([0.10, 0.14, 0.84, 0.74])
    ax.set_facecolor("#fffdf8")
    x = np.arange(frame_count, dtype=np.int32)

    ylim_low, ylim_high = resolve_ylim(signal_staircase, critics)

    ax.step(
        x,
        signal_staircase,
        where="post",
        color="#1f6f8b",
        linewidth=2.4,
        label="VLAC signal",
    )
    ax.scatter(steps, signals, color="#1f6f8b", s=16, zorder=3)
    ax.plot(
        steps,
        critics,
        color="#d97706",
        linewidth=1.6,
        marker="o",
        markersize=4.0,
        alpha=0.92,
        label="Chunk critic",
    )

    current_signal = float(signal_staircase[min(frame_idx, frame_count - 1)])
    critic_step_idx = np.searchsorted(steps, frame_idx, side="right") - 1
    current_critic = None
    if 0 <= critic_step_idx < len(critics):
        current_critic = float(critics[critic_step_idx])

    ax.axvline(frame_idx, color="#222222", linestyle="--", linewidth=1.1)
    ax.set_xlim(0, max(frame_count - 1, 1))
    ax.set_ylim(ylim_low, ylim_high)
    ax.set_xlabel("Rollout Step", fontsize=10, labelpad=3)
    ax.set_ylabel("Value", fontsize=10, labelpad=3)
    ax.set_title("VLAC Online Signal", fontsize=13, pad=4)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.18, color="#7f7f7f")
    for spine in ax.spines.values():
        spine.set_color("#3a3a3a")
        spine.set_linewidth(1.0)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.92)
    text_lines = [f"step={frame_idx}", f"signal={current_signal:.6f}"]
    if current_critic is not None:
        text_lines.append(f"critic={current_critic:.6f}")
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.88},
    )

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    plot_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)[..., :3]
    plt.close(fig)
    return plot_image


def main():
    args = parse_args()
    log_path = Path(args.log_path).expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Episode log not found: {log_path}")

    lines = read_text_lines(log_path)
    task_id, episode_idx = parse_episode_identifiers(log_path)
    del task_id
    success_flag = parse_success_flag(lines)
    instruction = args.instruction or parse_instruction(lines)
    if instruction is None:
        raise ValueError("Failed to infer task instruction from the episode log.")
    entries = parse_vlac_entries(lines)

    main_video_path, wrist_video_path = locate_rollout_videos(
        log_path=log_path,
        instruction=instruction,
        episode_idx=episode_idx,
        success_flag=success_flag,
        main_override=None if args.main_video_path is None else Path(args.main_video_path).expanduser().resolve(),
        wrist_override=None if args.wrist_video_path is None else Path(args.wrist_video_path).expanduser().resolve(),
    )

    primary_frames, fps = read_video(main_video_path)
    wrist_frames = None
    if wrist_video_path is not None:
        wrist_frames, _ = read_video(wrist_video_path)
    primary_frames, wrist_frames = truncate_video_streams(primary_frames, wrist_frames)

    frame_count = len(primary_frames)
    steps = np.asarray([entry["step"] for entry in entries], dtype=np.int32)
    critics = np.asarray([entry["critic"] for entry in entries], dtype=np.float32)
    signals = np.asarray([entry["signal"] for entry in entries], dtype=np.float32)
    signal_staircase = staircase_expand(steps, signals, frame_count)

    if output_path := args.output_path:
        overlay_path = Path(output_path).expanduser().resolve()
    else:
        overlay_path = main_video_path.with_name(f"{main_video_path.stem}_vlac_online_overlay.mp4")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    video_width = primary_frames[0].shape[1]
    video_height = primary_frames[0].shape[0]
    plot_width = int(video_width * float(args.plot_width_scale))

    right_frame_width = video_width
    if wrist_frames is not None:
        right_frame_width = max(video_width, wrist_frames[0].shape[1])

    if wrist_frames is not None and wrist_frames[0].shape[1] != right_frame_width:
        wrist_frames = [resize_to_width(frame, right_frame_width) for frame in wrist_frames]
    if primary_frames[0].shape[1] != right_frame_width:
        primary_frames = [resize_to_width(frame, right_frame_width) for frame in primary_frames]

    auxiliary_labels_frames: List[Tuple[str, List[np.ndarray]]] = []
    if wrist_frames is not None:
        auxiliary_labels_frames.append(("Wrist View", wrist_frames))

    sample_right_panel = render_camera_stack_panel(
        primary_frames[0],
        instruction,
        [(label, frames[0]) for label, frames in auxiliary_labels_frames],
    )
    plot_height = sample_right_panel.shape[0]

    if av is not None:
        container = av.open(str(overlay_path), mode="w")
        stream = create_video_stream_av(container, fps)
        try:
            for frame_idx, frame in enumerate(primary_frames):
                plot_frame = render_plot_frame(
                    frame_idx=frame_idx,
                    frame_count=frame_count,
                    steps=steps,
                    critics=critics,
                    signals=signals,
                    signal_staircase=signal_staircase,
                    panel_width=plot_width,
                    panel_height=plot_height,
                )
                right_panel = render_camera_stack_panel(
                    frame,
                    instruction,
                    [(label, frames[frame_idx]) for label, frames in auxiliary_labels_frames],
                )
                combined = compose_output_frame(plot_frame, right_panel)
                write_video_frame_av(container, stream, combined)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
    else:
        if imageio is None:
            raise RuntimeError("PyAV is unavailable and imageio is not installed, cannot write overlay video.")
        writer = imageio.get_writer(overlay_path, fps=fps)
        try:
            for frame_idx, frame in enumerate(primary_frames):
                plot_frame = render_plot_frame(
                    frame_idx=frame_idx,
                    frame_count=frame_count,
                    steps=steps,
                    critics=critics,
                    signals=signals,
                    signal_staircase=signal_staircase,
                    panel_width=plot_width,
                    panel_height=plot_height,
                )
                right_panel = render_camera_stack_panel(
                    frame,
                    instruction,
                    [(label, frames[frame_idx]) for label, frames in auxiliary_labels_frames],
                )
                combined = compose_output_frame(plot_frame, right_panel)
                writer.append_data(combined)
        finally:
            writer.close()

    reference_paths = sorted({entry["reference_video_path"] for entry in entries})
    print("[OK] VLAC online visualization complete")
    print(f"  Log: {log_path}")
    print(f"  Instruction: {instruction}")
    print(f"  Main video: {main_video_path}")
    print(f"  Wrist video: {wrist_video_path if wrist_video_path is not None else 'N/A'}")
    print(f"  Frames: {frame_count}")
    print(f"  Chunk samples: {len(entries)}")
    print(f"  Reference video(s): {reference_paths}")
    print(f"  Overlay video: {overlay_path}")


if __name__ == "__main__":
    main()
