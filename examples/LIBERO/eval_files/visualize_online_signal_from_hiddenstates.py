import argparse
from pathlib import Path

from starVLA.model.framework.signal_utils import (
    build_online_signal_curve,
    default_signal_curve_path,
    default_signal_overlay_path,
    load_signal_curve_npz,
    read_video,
    render_signal_overlay_video,
    save_signal_curve_npz,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize online signal saved during rollout inference from hidden-state metadata."
    )
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--hidden-dir", type=str, required=True)
    parser.add_argument("--output-curve-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    hidden_dir = Path(args.hidden_dir).expanduser().resolve()
    frames, fps = read_video(video_path)
    curve, instruction = build_online_signal_curve(hidden_dir, len(frames))

    curve_path = (
        Path(args.output_curve_path).expanduser().resolve()
        if args.output_curve_path is not None
        else default_signal_curve_path(video_path, signal_name="online_signal")
    )
    save_signal_curve_npz(
        output_path=curve_path,
        curve=curve,
        signal_name="online_signal",
        instruction=instruction,
        video_path=video_path,
        fps=fps,
        status="online_saved_signal",
    )
    print(f"[saved] {curve_path}")

    curves = [load_signal_curve_npz(curve_path, len(frames))]
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else default_signal_overlay_path(video_path, "online_signal_overlay")
    )
    render_signal_overlay_video(
        video_path=video_path,
        curves=curves,
        instruction=instruction,
        output_path=output_path,
        title="Online Signal",
    )
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
