from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_repo_import_path(repo_name: str) -> None:
    repo_root = _workspace_root() / repo_name
    if not repo_root.exists():
        raise ImportError(
            f"Local repo '{repo_name}' not found under {_workspace_root()}.")
    path_str = str(repo_root)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: vlac_subprocess_runner.py <request.json> <response.npz>")

    request_path = Path(sys.argv[1]).expanduser().resolve()
    response_path = Path(sys.argv[2]).expanduser().resolve()
    request = json.loads(request_path.read_text())

    _ensure_repo_import_path("VLAC")
    from evo_vlac import GAC_model

    critic = GAC_model(tag="critic")
    critic.init_model(
        model_path=request["model_path"],
        model_type=request["model_type"],
        device_map=request["device"],
    )
    critic.temperature = 0.5
    critic.top_k = 1
    critic.set_config()
    critic.set_system_prompt()

    result_video_path, value_list, critic_list, done_list = critic.web_trajectory_critic(
        task_description=request["instruction"],
        main_video_path=request["video_path"],
        reference_video_path=request["reference_video_path"],
        batch_num=int(request["batch_num"]),
        ref_num=int(request["ref_num"]),
        think=bool(request["think"]),
        skip=int(request["skip"]),
        rich=bool(request["rich"]),
        reverse_eval=False,
        output_path=request["output_path"],
        fps=float(request["fps"]),
        frame_skip=bool(request["frame_skip"]),
        done_flag=bool(request["done_flag"]),
        in_context_done=bool(request["in_context_done"]),
        done_threshold=float(request["done_threshold"]),
        video_output=bool(request["video_output"]),
    )

    response_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        response_path,
        result_video_path=np.asarray(
            [("" if result_video_path is None else str(result_video_path))], dtype=object),
        value_list=np.asarray(value_list, dtype=np.float32),
        critic_list=np.asarray(critic_list, dtype=np.float32),
        done_list=np.asarray(
            [] if done_list is None else done_list, dtype=np.float32),
    )


if __name__ == "__main__":
    main()
