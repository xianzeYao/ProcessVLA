import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from typing import Optional

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.simBenchmarks.LIBERO.eval_files.model2libero_interface import ModelClient

try:
    from .eval_utils import validate_task_range
except ImportError:
    from eval_utils import validate_task_range

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size: list[int] = dataclasses.field(default_factory=lambda: [224, 224])

    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    start_idx: int = 0
    end_idx: int = -1

    video_out_path: str = "experiments/libero/logs"
    log_path: str = "experiments/libero/logs"
    save_video: bool = False
    video_views: str = "all"  # Options: agentview, all
    episode_result_path: Optional[str] = None

    seed: int = 7
    pretrained_path: str = ""
    post_process_action: bool = True
    job_name: str = "test"


def _max_steps_for_suite(task_suite_name: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }.get(task_suite_name, -1)


def _write_json_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _append_jsonl(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def eval_libero(args: Args) -> None:
    logging.info("Arguments: %s", json.dumps(dataclasses.asdict(args), indent=2))
    if args.video_views not in {"agentview", "all"}:
        raise ValueError(f"Unknown video_views={args.video_views!r}; expected 'agentview' or 'all'.")
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = int(task_suite.n_tasks)
    max_steps = _max_steps_for_suite(args.task_suite_name)
    if max_steps < 0:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")
    start_idx, end_idx = validate_task_range(
        args.start_idx,
        num_tasks_in_suite if args.end_idx < 0 else args.end_idx,
        num_tasks_in_suite,
    )
    logging.info(
        "Task suite=%s range=[%d,%d) tasks=%d trials_per_task=%d save_video=%s video_views=%s",
        args.task_suite_name,
        start_idx,
        end_idx,
        num_tasks_in_suite,
        args.num_trials_per_task,
        args.save_video,
        args.video_views,
    )

    if args.save_video:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.log_path).mkdir(parents=True, exist_ok=True)

    # The CoT server owns action unnormalization. The LIBERO client only
    # connects to the already-loaded policy server, so it must not reload the
    # checkpoint or duplicate dataset-statistics handling here.
    client_model = ModelClient(
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
    )

    libero_home = os.environ.get("LIBERO_HOME", "path_to_LIBERO-plus_home")
    classification_path = pathlib.Path(libero_home) / "libero/libero/benchmark/task_classification.json"
    with classification_path.open(encoding="utf-8") as handle:
        task_mapping = json.load(handle)[args.task_suite_name]
    id2category = {
        int(item["id"]): (str(item["category"]), str(item["name"]))
        for item in task_mapping
    }

    episode_rows: list[dict] = []
    task_ids = list(range(start_idx, end_idx))
    total_episodes = 0
    total_successes = 0
    category_results: dict[str, dict[str, int]] = {}
    episode_jsonl_path = pathlib.Path(args.episode_result_path) if args.episode_result_path else None

    for task_id in tqdm.tqdm(task_ids, desc=args.task_suite_name):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if args.num_trials_per_task < 1 or args.num_trials_per_task > len(initial_states):
            raise ValueError(
                f"num_trials_per_task={args.num_trials_per_task} is invalid for task_id={task_id}; "
                f"available initial states={len(initial_states)}"
            )
        task_description = task.language
        category, category_name = id2category.get(task_id + 1, ("unknown", str(task_id)))
        category_results.setdefault(category, {"total_count": 0, "success_count": 0})

        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        try:
            for episode_idx in range(args.num_trials_per_task):
                logging.info("Starting task_id=%d episode_idx=%d", task_id, episode_idx)
                done = False
                end_reason = "max_steps"
                exception_text = None
                replay_images: list[np.ndarray] = []
                replay_wrist_images: list[np.ndarray] = []
                action_trace: list[list[float]] = []
                t = 0
                step = 0

                try:
                    client_model.reset(task_description=task_description)
                    env.reset()
                    obs = env.set_init_state(initial_states[episode_idx])

                    while t < max_steps + args.num_steps_wait:
                        if t < args.num_steps_wait:
                            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        if args.save_video:
                            replay_images.append(img)
                            if args.video_views == "all":
                                replay_wrist_images.append(wrist_img)

                        example_dict = {
                            "image": [img, wrist_img],
                            "lang": str(task_description),
                        }
                        response = client_model.step(example=example_dict, step=step)
                        raw_action = response["raw_action"]
                        world_vector_delta = np.asarray(raw_action["world_vector"], dtype=np.float32).reshape(-1)
                        rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float32).reshape(-1)
                        open_gripper = np.asarray(raw_action["open_gripper"], dtype=np.float32).reshape(-1)
                        gripper = _binarize_gripper_open(open_gripper)
                        if not (
                            world_vector_delta.size == 3
                            and rotation_delta.size == 3
                            and open_gripper.size == 1
                        ):
                            raise ValueError(
                                "Invalid action sizes: "
                                f"world_vector={world_vector_delta.shape}, "
                                f"rotation_delta={rotation_delta.shape}, "
                                f"open_gripper={open_gripper.shape}"
                            )
                        delta_action = np.concatenate(
                            [world_vector_delta, rotation_delta, gripper], axis=0
                        ).astype(np.float32)
                        action_trace.append(delta_action.tolist())
                        logging.info(
                            "action task_id=%d episode_idx=%d policy_step=%d action=%s",
                            task_id,
                            episode_idx,
                            step,
                            np.array2string(delta_action, precision=6, separator=", "),
                        )

                        obs, _, done, _ = env.step(delta_action.tolist())
                        if done:
                            end_reason = "done"
                            break
                        t += 1
                        step += 1
                    else:
                        end_reason = "max_steps"
                except Exception as exc:
                    exception_text = f"{type(exc).__name__}: {exc}"
                    end_reason = "exception"
                    logging.exception(
                        "Episode failed task_id=%d episode_idx=%d",
                        task_id,
                        episode_idx,
                    )

                success = bool(done)
                total_episodes += 1
                total_successes += int(success)
                category_results[category]["total_count"] += 1
                category_results[category]["success_count"] += int(success)
                episode_row = {
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "category": category,
                    "category_name": category_name,
                    "success": success,
                    "end_reason": end_reason,
                    "exception": exception_text,
                    "action_count": len(action_trace),
                }
                if args.save_video and replay_images:
                    task_segment = category_name.replace(" ", "_").replace("/", "_")
                    suffix = "success" if success else "failure"
                    primary_path = pathlib.Path(args.video_out_path) / (
                        f"rollout_{task_segment}_task{task_id}_episode{episode_idx}_{suffix}.mp4"
                    )
                    wrist_path = pathlib.Path(args.video_out_path) / (
                        f"rollout_{task_segment}_wrist_task{task_id}_episode{episode_idx}_{suffix}.mp4"
                    )
                    imageio.mimwrite(primary_path, replay_images, fps=10)
                    episode_row["video_path"] = str(primary_path)
                    if args.video_views == "all" and replay_wrist_images:
                        imageio.mimwrite(wrist_path, replay_wrist_images, fps=10)
                        episode_row["wrist_video_path"] = str(wrist_path)

                # Keep the slice summary compact; the per-episode JSONL retains
                # the full action trace for detailed debugging/replay.
                episode_rows.append(episode_row)
                if episode_jsonl_path is not None:
                    episode_log_row = {**episode_row, "actions": action_trace}
                    _append_jsonl(episode_jsonl_path, episode_log_row)
                logging.info(
                    "episode_result task_id=%d episode_idx=%d success=%s end_reason=%s total=%d successes=%d",
                    task_id,
                    episode_idx,
                    success,
                    end_reason,
                    total_episodes,
                    total_successes,
                )
        finally:
            try:
                env.close()
            except Exception:
                logging.exception("Failed to close env for task_id=%d", task_id)

    slice_payload = {
        "suite": args.task_suite_name,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "episode_count": total_episodes,
        "success_count": total_successes,
        "success_rate": total_successes / total_episodes if total_episodes else 0.0,
        "categories": category_results,
        "episodes": episode_rows,
    }
    slice_path = pathlib.Path(args.log_path) / f"{args.task_suite_name}_{start_idx}_{end_idx}.json"
    _write_json_atomic(slice_path, slice_payload)
    if start_idx == 0 and end_idx == num_tasks_in_suite:
        _write_json_atomic(pathlib.Path(args.log_path) / f"{args.task_suite_name}.json", category_results)
    logging.info(
        "Finished suite=%s range=[%d,%d) success_rate=%.4f episodes=%d result=%s",
        args.task_suite_name,
        start_idx,
        end_idx,
        slice_payload["success_rate"],
        total_episodes,
        slice_path,
    )


def _get_libero_env(task, resolution, seed):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    tyro.cli(eval_libero)
