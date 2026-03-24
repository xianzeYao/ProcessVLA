from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_processvla_import_path() -> None:
    processvla_root = Path(__file__).resolve().parents[3]
    path_str = str(processvla_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _ensure_repo_import_path(repo_root: str | Path) -> None:
    resolved_root = Path(repo_root).expanduser().resolve()
    if not resolved_root.exists():
        raise ImportError(f"VLAC repo root does not exist: {resolved_root}")
    path_str = str(resolved_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _resolve_ffmpeg() -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        return ffmpeg_bin
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError(
            "ffmpeg is required in the VLAC environment to transcode AV1 inputs. "
            "Install ffmpeg or `pip install imageio-ffmpeg` in the VLAC environment."
        ) from exc


def _transcode_video_for_vlac(
    video_path: str | Path,
    *,
    temp_root: str | Path,
    stem_suffix: str,
) -> Path:
    source_path = Path(video_path).expanduser().resolve()
    output_dir = Path(temp_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_path.stem}_{stem_suffix}_h264.mp4"
    command = [
        _resolve_ffmpeg(),
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


def _load_reference_frames(reference_video_path: str, cache: dict, temp_root: str | Path):
    resolved_path = str(Path(reference_video_path).expanduser().resolve())
    if resolved_path in cache:
        return cache[resolved_path]
    from starVLA.model.framework.signal_common import read_video

    transcoded_path = _transcode_video_for_vlac(
        resolved_path,
        temp_root=temp_root,
        stem_suffix="reference",
    )
    frames_np, _fps = read_video(transcoded_path)
    reference_frames = [Image.fromarray(frame).convert("RGB") for frame in frames_np]
    cache[resolved_path] = reference_frames
    return reference_frames


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-type", default="internvl2")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_processvla_import_path()
    _ensure_repo_import_path(args.repo_root)
    from evo_vlac import GAC_model

    critic = GAC_model(tag="critic")
    critic.init_model(
        model_path=args.model_path,
        model_type=args.model_type,
        device_map=args.device,
    )
    critic.temperature = 0.5
    critic.top_k = 1
    critic.set_config()
    critic.set_system_prompt()

    reference_cache = {}
    with tempfile.TemporaryDirectory(prefix="vlac_online_ref_") as temp_root:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                previous_frame = Image.open(request["previous_frame_path"]).convert("RGB")
                current_frame = Image.open(request["current_frame_path"]).convert("RGB")
                reference_frames = _load_reference_frames(
                    request["reference_video_path"],
                    cache=reference_cache,
                    temp_root=temp_root,
                )
                critic_list, _value_list = critic.get_trajectory_critic(
                    task=str(request["instruction"]),
                    image_list=[previous_frame, current_frame],
                    ref_image_list=reference_frames,
                    batch_num=int(request["batch_num"]),
                    ref_num=int(request["ref_num"]),
                    think=bool(request["think"]),
                    skip=1,
                    rich=bool(request["rich"]),
                    reverse_eval=False,
                    frame_skip=True,
                    addition_scale=1,
                    bias=0,
                    related_critic=False,
                    positive_clip=0,
                    negative_clip=0,
                )
                critic_value = 0.0 if not critic_list else float(critic_list[-1])
                print(json.dumps({"critic": critic_value}), flush=True)
            except Exception as exc:
                print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
