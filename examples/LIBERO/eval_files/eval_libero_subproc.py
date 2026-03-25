import dataclasses
import json
import logging
import os
import pathlib
import subprocess
import sys
from typing import Optional

import tqdm
import tyro
from libero.libero import benchmark

from examples.LIBERO.eval_files.eval_libero import Args as BaseArgs


@dataclasses.dataclass
class Args(BaseArgs):
    episode_timeout_seconds: int = 1800
    results_jsonl: Optional[str] = None


def _build_eval_command(args: Args, task_id: int, episode_idx: int, result_path: pathlib.Path):
    eval_script = pathlib.Path(__file__).with_name("eval_libero.py")
    cmd = [sys.executable, str(eval_script)]

    arg_dict = dataclasses.asdict(args)
    for key in ("task_id", "episode_idx", "episode_result_path", "episode_timeout_seconds", "results_jsonl"):
        arg_dict.pop(key, None)

    for key, value in arg_dict.items():
        if value is None:
            continue
        if key == "post_process_action":
            if value is False:
                cmd.append("--args.no-post-process-action")
            continue
        cli_key = f"--args.{key.replace('_', '-')}"
        if isinstance(value, bool):
            cmd.extend([cli_key, "True" if value else "False"])
        elif isinstance(value, (list, tuple)):
            cmd.append(cli_key)
            cmd.extend(str(item) for item in value)
        else:
            cmd.extend([cli_key, str(value)])

    cmd.extend([
        "--args.task-id", str(task_id),
        "--args.episode-idx", str(episode_idx),
        "--args.episode-result-path", str(result_path),
    ])
    return cmd


def eval_libero_subproc(args: Args) -> None:
    logging.info("Arguments: %s", json.dumps(dataclasses.asdict(args), indent=4))

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    task_ids = [args.task_id] if args.task_id is not None else list(range(num_tasks_in_suite))
    run_root = pathlib.Path(args.video_out_path)
    episode_log_dir = run_root / "_episode_logs"
    episode_log_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = pathlib.Path(args.results_jsonl) if args.results_jsonl else run_root / "episode_results.jsonl"
    results_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_episodes, task_successes = 0, 0
        logging.info("Task: %s", task.language)

        if args.episode_idx is not None:
            if not (0 <= args.episode_idx < len(initial_states)):
                raise ValueError(
                    f"episode_idx={args.episode_idx} out of range for task_id={task_id}; "
                    f"valid range is [0, {len(initial_states) - 1}]"
                )
            episode_indices = [args.episode_idx]
        else:
            episode_indices = list(range(args.num_trials_per_task))

        for episode_idx in tqdm.tqdm(episode_indices):
            episode_result_path = episode_log_dir / f"task{task_id:03d}_ep{episode_idx:03d}.jsonl"
            episode_log_path = episode_log_dir / f"task{task_id:03d}_ep{episode_idx:03d}.log"
            if episode_result_path.exists():
                episode_result_path.unlink()

            cmd = _build_eval_command(args, task_id, episode_idx, episode_result_path)
            logging.info("Starting isolated episode: task_id=%s episode_idx=%s", task_id, episode_idx)
            result = {
                "task_id": task_id,
                "episode_idx": episode_idx,
                "success": False,
                "end_reason": None,
                "returncode": None,
                "error": None,
            }

            with episode_log_path.open("w", encoding="utf-8") as log_file:
                try:
                    proc = subprocess.run(
                        cmd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        env=os.environ.copy(),
                        timeout=args.episode_timeout_seconds,
                        check=False,
                    )
                    result["returncode"] = proc.returncode
                    if proc.returncode == 0 and episode_result_path.exists():
                        lines = [line for line in episode_result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                        if lines:
                            payload = json.loads(lines[-1])
                            result["success"] = bool(payload.get("success", False))
                            result["end_reason"] = payload.get("end_reason")
                        else:
                            result["error"] = "empty result file"
                    elif proc.returncode != 0:
                        result["error"] = f"subprocess_returncode={proc.returncode}"
                    else:
                        result["error"] = "missing result file"
                except subprocess.TimeoutExpired:
                    result["error"] = f"timeout>{args.episode_timeout_seconds}s"

            task_episodes += 1
            total_episodes += 1
            if result["success"]:
                task_successes += 1
                total_successes += 1

            with results_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")

            logging.info(
                "task_id=%s episode_idx=%s success=%s total=%s/%s (%.1f%%) end_reason=%s error=%s log=%s",
                task_id,
                episode_idx,
                result["success"],
                total_successes,
                total_episodes,
                (100.0 * total_successes / total_episodes) if total_episodes else 0.0,
                result["end_reason"],
                result["error"],
                episode_log_path,
            )

        logging.info(
            "Task %s done: %s/%s = %.1f%%",
            task_id,
            task_successes,
            task_episodes,
            (100.0 * task_successes / task_episodes) if task_episodes else 0.0,
        )

    logging.info(
        "Total success rate: %.4f (%s/%s)",
        (float(total_successes) / float(total_episodes)) if total_episodes else 0.0,
        total_successes,
        total_episodes,
    )
    logging.info("Results JSONL: %s", results_jsonl)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    tyro.cli(eval_libero_subproc)
