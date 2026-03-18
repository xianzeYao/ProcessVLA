import argparse
from pathlib import Path
from typing import Optional

from starVLA.model.framework.signal_utils import (
    compute_signal_curve,
    default_signal_curve_path,
    default_signal_overlay_path,
    load_instruction_from_hidden_dir,
    load_signal_curve_npz,
    read_video,
    render_signal_overlay_video,
    save_signal_curve_npz,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline signal pipeline: compute curves from rollout videos and optionally render them."
    )
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--hidden-dir", type=str, default=None)
    parser.add_argument(
        "--signal-names",
        nargs="*",
        default=["signal"],
        help="Signal names to compute offline when --curve-paths is not provided.",
    )
    parser.add_argument(
        "--curve-paths",
        nargs="*",
        default=None,
        help="Optional precomputed curve files. If provided, skip offline computation and render these files directly.",
    )
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def resolve_instruction(instruction: Optional[str], hidden_dir: Optional[str]) -> str:
    if instruction is not None:
        return str(instruction)
    if hidden_dir is not None:
        return load_instruction_from_hidden_dir(hidden_dir)
    raise ValueError("Provide --instruction or --hidden-dir when computing offline curves.")


def main():
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    frames, fps = read_video(video_path)

    if args.curve_paths:
        curves = [load_signal_curve_npz(path, len(frames)) for path in args.curve_paths]
        instruction = curves[0]["instruction"] if curves else ""
    else:
        instruction = resolve_instruction(args.instruction, args.hidden_dir)
        curves = []
        for signal_name in args.signal_names:
            curve = compute_signal_curve(frames, instruction, signal_name=signal_name)
            curve_path = default_signal_curve_path(video_path, signal_name=signal_name)
            save_signal_curve_npz(
                output_path=curve_path,
                curve=curve,
                signal_name=signal_name,
                instruction=instruction,
                video_path=video_path,
                fps=fps,
                status="offline_placeholder",
            )
            curves.append(load_signal_curve_npz(curve_path, len(frames)))
            print(f"[saved] {curve_path}")

    if args.skip_render:
        return

    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else default_signal_overlay_path(video_path, "offline_signals_overlay")
    )
    render_signal_overlay_video(
        video_path=video_path,
        curves=curves,
        instruction=instruction,
        output_path=output_path,
        title="Offline Signal Curves",
    )
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
