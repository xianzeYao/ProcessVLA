"""Resumable raw LIBERO V3 left/right/wrist rollout collection.

Pure record, cadence, and alignment functions deliberately have no LIBERO or
MuJoCo imports. Simulator imports are lazy inside :func:`collect_rollout`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .libero_trace_audit_metrics import align_realized_trace, canonicalize_v3_uvd, compute_anchor_metrics, metrics_to_jsonable
from .libero_trace_audit_selection import AuditCase
from .probe_utils import build_latency_fields

RECORD_VERSION = 1
LANDMARK_COUNT = 3
LANDMARK_ORDER = ("left", "right", "wrist")
LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
DEFAULT_ACTION_HORIZON, DEFAULT_DUMMY_STEPS = 8, 10
_ARRAY_FIELDS = (
    "agent_rgb", "wrist_rgb", "agent_depth", "executed_actions", "anchor_steps",
    "predicted_uvd", "predicted_uvd_time", "predicted_uvd_landmark_ids",
    "predicted_depth_current", "predicted_depth_future", "realized_uvd", "realized_xyz",
    "realized_valid", "realized_in_frame", "anchor_target_uvd", "anchor_target_valid",
    "dense_depth_current_target", "dense_depth_future_target", "latency_ms",
)


@dataclass(frozen=True)
class RolloutRecord:
    """Versioned raw arrays and strict JSON case/outcome/metric metadata.

    RGB is 180-degree-flipped uint8 ``[step,H,W,3]``. UVD is normalized
    ``(u,v,metric camera depth in metres)`` and XYZ is world-space metres.
    ``realized_valid`` is projection-valid; ``realized_in_frame`` is a
    separate bounds mask. V3 predictions are time-major left/right/wrist.
    """
    agent_rgb: np.ndarray; wrist_rgb: np.ndarray; agent_depth: np.ndarray; executed_actions: np.ndarray
    anchor_steps: np.ndarray; predicted_uvd: np.ndarray; predicted_uvd_time: np.ndarray
    predicted_uvd_landmark_ids: np.ndarray; predicted_depth_current: np.ndarray; predicted_depth_future: np.ndarray
    realized_uvd: np.ndarray; realized_xyz: np.ndarray; realized_valid: np.ndarray; realized_in_frame: np.ndarray
    anchor_target_uvd: np.ndarray; anchor_target_valid: np.ndarray; dense_depth_current_target: np.ndarray
    dense_depth_future_target: np.ndarray; latency_ms: np.ndarray; metadata: dict[str, object]

    @classmethod
    def array_field_names(cls) -> tuple[str, ...]: return _ARRAY_FIELDS


@dataclass(frozen=True)
class CadenceTrace:
    """Pure evidence of policy steps; dummy stabilization is excluded."""
    anchor_steps: np.ndarray; actions: np.ndarray; responses: tuple[Mapping[str, object], ...]


def derive_inference_seed(audit_seed: int, anchor_step: int) -> int:
    """Injectively encode non-negative audit seed and anchor (<65536)."""
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in (audit_seed, anchor_step)):
        raise ValueError("audit_seed and anchor_step must be integers")
    if audit_seed < 0 or anchor_step < 0 or anchor_step >= 2**16:
        raise ValueError("audit_seed and anchor_step must be non-negative; anchor_step must be < 65536")
    return (int(audit_seed) << 16) | int(anchor_step)


def build_geometry_request(observation: Mapping[str, object], *, inference_seed: int, unnorm_key: str | None = None) -> dict[str, object]:
    """One Task-1 request retaining primary then wrist image order."""
    images = observation.get("image")
    if not isinstance(images, Sequence) or len(images) != 2: raise ValueError("observation must contain exactly primary and wrist images")
    result: dict[str, object] = {"examples": [{"image": [images[0], images[1]], "lang": str(observation.get("lang", ""))}], "do_sample": False, "use_ddim": True, "num_ddim_steps": 10, "return_geometry": True, "inference_seed": int(inference_seed)}
    if unnorm_key is not None: result["unnorm_key"] = unnorm_key
    return result


def execute_cadenced_actions(*, max_steps: int, action_horizon: int, audit_seed: int, dummy_steps: int, stabilize: Callable[[np.ndarray], object], observe: Callable[[int], Mapping[str, object]], request: Callable[[dict[str, object]], Mapping[str, object]], execute: Callable[[np.ndarray, int], bool], unnorm_key: str | None = None) -> CadenceTrace:
    """Run chunks at 0,horizon,... while dummy steps never advance anchors."""
    if max_steps < 1 or action_horizon < 1 or dummy_steps < 0: raise ValueError("invalid cadence values")
    for _ in range(dummy_steps): stabilize(LIBERO_DUMMY_ACTION.copy())
    anchors: list[int] = []; actions: list[np.ndarray] = []; responses: list[Mapping[str, object]] = []; chunk = None
    for step in range(max_steps):
        observation = observe(step)
        if step % action_horizon == 0:
            anchors.append(step); response = request(build_geometry_request(observation, inference_seed=derive_inference_seed(audit_seed, step), unnorm_key=unnorm_key)); data = response.get("data", response)
            if not isinstance(data, Mapping) or "actions" not in data: raise ValueError("policy response must contain actions")
            chunk = _squeeze_action_chunk(data["actions"], action_horizon); responses.append(data)
        action = chunk[step % action_horizon].copy()  # type: ignore[index]
        actions.append(action)
        if execute(action, step): break
    return CadenceTrace(np.asarray(anchors, np.int32), np.asarray(actions, np.float32).reshape(-1, 7), tuple(responses))


def complete_anchor_targets(*, predicted_uvd: np.ndarray, predicted_uvd_time: np.ndarray, predicted_uvd_landmark_ids: np.ndarray, anchor_steps: np.ndarray, realized_uvd: np.ndarray, realized_valid: np.ndarray, action_horizon: int, image_size: int | tuple[int, int], camera_k: np.ndarray | None = None, depth_dead_zone_m: float = 0.002) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Validate time-major V3 tokens, align true LRW, and make strict metrics."""
    prediction, times, ids, anchors = np.asarray(predicted_uvd, np.float32), np.asarray(predicted_uvd_time), np.asarray(predicted_uvd_landmark_ids), np.asarray(anchor_steps)
    if prediction.ndim != 4 or prediction.shape[2:] != (3, 3): raise ValueError("predicted_uvd must have shape [anchor,time,3,3]")
    count, points = prediction.shape[:2]
    if times.shape != (count, points * 3) or ids.shape != times.shape: raise ValueError("V3 response metadata shape mismatch")
    if anchors.shape != (count,) or anchors.dtype != np.int32: raise ValueError("anchor_steps must be int32 [anchor]")
    if action_horizon < 1: raise ValueError("action_horizon must be positive")
    targets = np.full_like(prediction, np.nan); valid = np.zeros(prediction.shape[:-1], np.bool_); output: dict[str, object] = {}
    for index, anchor in enumerate(anchors):
        canonical = canonicalize_v3_uvd(prediction[index].reshape(points * 3, 3), time_points=points, uvd_time=times[index], uvd_landmark_ids=ids[index])
        offsets_float = times[index].reshape(points, 3)[:, 0] * action_horizon; offsets = np.rint(offsets_float).astype(np.int64)
        if np.any(offsets < 0) or np.any(offsets > action_horizon) or not np.allclose(offsets_float, offsets, atol=1e-5): raise ValueError("V3 uvd_time must map exactly to action-horizon offsets")
        target, target_valid = align_realized_trace(realized_uvd, int(anchor), offsets, step_valid=realized_valid)
        targets[index], valid[index] = target, target_valid
        output[f"anchor_{index}"] = metrics_to_jsonable(compute_anchor_metrics(canonical, target, target_valid, image_size=image_size, depth_dead_zone_m=depth_dead_zone_m, camera_k=camera_k, uvd_time=times[index], uvd_landmark_ids=ids[index]))
    json.dumps(output, allow_nan=False)
    return targets, valid, output


def save_rollout_record(path: str | Path, record: RolloutRecord) -> tuple[Path, Path]:
    """Write compressed NPZ then strict JSON atomically; hash prevents partial resume."""
    validate_rollout_record(record); npz_path, json_path = _record_paths(path); npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(getattr(record, name)) for name in _ARRAY_FIELDS}
    with tempfile.NamedTemporaryFile(dir=npz_path.parent, prefix=npz_path.stem + ".", suffix=".npz", delete=False) as handle: temporary_npz = Path(handle.name)
    temporary_json: Path | None = None
    try:
        np.savez_compressed(temporary_npz, **arrays)
        manifest = {"version": RECORD_VERSION, "npz_sha256": _file_sha256(temporary_npz), "arrays": {name: _array_spec(value) for name, value in arrays.items()}, "metadata": metrics_to_jsonable(record.metadata)}
        with tempfile.NamedTemporaryFile(dir=json_path.parent, prefix=json_path.name + ".", suffix=".tmp", mode="w", encoding="utf-8", delete=False) as handle:
            temporary_json = Path(handle.name); handle.write(json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":")))
        os.replace(temporary_npz, npz_path); os.replace(temporary_json, json_path)
    finally:
        temporary_npz.unlink(missing_ok=True)
        if temporary_json is not None: temporary_json.unlink(missing_ok=True)
    return npz_path, json_path


def load_rollout_record(path: str | Path) -> RolloutRecord:
    """Accept only full, hash-matched, versioned records for resume."""
    npz_path, json_path = _record_paths(path)
    if not npz_path.is_file() or not json_path.is_file(): raise FileNotFoundError(f"complete rollout requires {npz_path.name} and {json_path.name}")
    try: manifest = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error: raise ValueError("rollout metadata JSON is corrupt") from error
    _validate_manifest(manifest)
    if _file_sha256(npz_path) != manifest["npz_sha256"]: raise ValueError("rollout NPZ sha256 does not match metadata")
    try:
        with np.load(npz_path, allow_pickle=False) as source:
            if set(source.files) != set(_ARRAY_FIELDS): raise ValueError("rollout NPZ array keys are incomplete or unexpected")
            arrays = {name: np.asarray(source[name]) for name in _ARRAY_FIELDS}
    except (OSError, ValueError) as error:
        if "array keys" in str(error): raise
        raise ValueError("rollout NPZ is corrupt") from error
    for name, array in arrays.items():
        if manifest["arrays"].get(name) != _array_spec(array): raise ValueError(f"rollout array {name} does not match metadata shape or dtype")
    record = RolloutRecord(**arrays, metadata=manifest["metadata"]); validate_rollout_record(record); return record


def validate_rollout_record(record: RolloutRecord) -> None:
    """Strictly validate required arrays, dtypes, version-independent shapes, and anchors."""
    values = {name: np.asarray(getattr(record, name)) for name in _ARRAY_FIELDS}; _require(values["agent_rgb"], (None, None, None, 3), np.uint8, "agent_rgb")
    steps, height, width, _ = values["agent_rgb"].shape
    if steps < 1 or height < 1 or width < 1: raise ValueError("agent_rgb must contain non-empty frames")
    _require(values["wrist_rgb"], (steps, height, width, 3), np.uint8, "wrist_rgb"); _require(values["agent_depth"], (steps, height, width), np.float32, "agent_depth"); _require(values["executed_actions"], (steps, 7), np.float32, "executed_actions")
    for name in ("realized_uvd", "realized_xyz"): _require(values[name], (steps, 3, 3), np.float32, name)
    for name in ("realized_valid", "realized_in_frame"): _require(values[name], (steps, 3), np.bool_, name)
    if np.any(values["realized_in_frame"] & ~values["realized_valid"]): raise ValueError("realized_in_frame cannot exceed realized_valid")
    metadata = record.metadata
    if not isinstance(metadata, dict) or {"case", "outcome", "action_horizon", "image_size", "metrics"} - set(metadata): raise ValueError("record metadata is missing required case/outcome/schema fields")
    horizon = metadata["action_horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1: raise ValueError("metadata action_horizon must be a positive integer")
    anchors = values["anchor_steps"]
    if anchors.dtype != np.int32 or anchors.ndim != 1 or len(anchors) < 1: raise ValueError("anchor_steps must be non-empty int32 [anchor]")
    if anchors[0] != 0 or np.any(np.diff(anchors) != horizon) or np.any(anchors >= steps): raise ValueError("anchor_steps must start at zero, be strictly horizon-spaced, and lie in episode")
    count = len(anchors); prediction = values["predicted_uvd"]
    if prediction.dtype != np.float32 or prediction.ndim != 4 or prediction.shape[0] != count or prediction.shape[2:] != (3, 3): raise ValueError("predicted_uvd must be float32 [anchor,time,3,3]")
    points = prediction.shape[1]
    if points < 2: raise ValueError("predicted_uvd must have at least two V3 time points")
    _require(values["predicted_uvd_time"], (count, points * 3), np.float32, "predicted_uvd_time"); _require(values["predicted_uvd_landmark_ids"], (count, points * 3), np.int64, "predicted_uvd_landmark_ids")
    depth_shape = values["predicted_depth_current"].shape
    for name in ("predicted_depth_current", "predicted_depth_future", "dense_depth_current_target", "dense_depth_future_target"):
        array = values[name]
        if array.dtype != np.float32 or array.ndim != 3 or array.shape[0] != count or array.shape[1:] != depth_shape[1:]: raise ValueError(f"{name} must be float32 [anchor,depth_height,depth_width]")
    _require(values["anchor_target_uvd"], prediction.shape, np.float32, "anchor_target_uvd"); _require(values["anchor_target_valid"], prediction.shape[:-1], np.bool_, "anchor_target_valid"); _require(values["latency_ms"], (count,), np.float64, "latency_ms")
    for index in range(count): canonicalize_v3_uvd(prediction[index].reshape(points * 3, 3), time_points=points, uvd_time=values["predicted_uvd_time"][index], uvd_landmark_ids=values["predicted_uvd_landmark_ids"][index])
    try: json.dumps(metrics_to_jsonable(metadata), allow_nan=False)
    except (TypeError, ValueError) as error: raise ValueError("record metadata must be strict JSON") from error


def collect_rollout(case: AuditCase, client: Any, args: Any) -> RolloutRecord:
    """Collect one selected task with agent/wrist RGB and agentview metric depth only."""
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    resolution, horizon, dummy = int(getattr(args, "resolution", 256)), int(getattr(args, "action_horizon", 8)), int(getattr(args, "dummy_steps", 10))
    max_steps = int(getattr(args, "max_steps", _max_steps_for_suite(case.suite)))
    if resolution < 2 or horizon < 1 or dummy < 0 or max_steps < 1: raise ValueError("invalid rollout settings")
    suite = benchmark.get_benchmark_dict()[case.suite](); task = suite.get_task(case.task_id); states = suite.get_task_init_states(case.task_id)
    if not 0 <= case.initial_state_index < len(states): raise ValueError("audit case initial_state_index is outside task initial states")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=resolution, camera_widths=resolution, camera_depths=True)
    try:
        env.seed(case.seed); env.reset(); obs = env.set_init_state(states[case.initial_state_index])
        for _ in range(dummy): obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
        frames: list[dict[str, np.ndarray]] = []; predictions: list[dict[str, np.ndarray]] = []; actions: list[np.ndarray] = []; latency: list[float] = []; chunk = None; done = False
        for step in range(max_steps):
            frame = _capture_frame(env, obs, resolution); frames.append(frame)
            if step % horizon == 0:
                request = build_geometry_request({"image": [frame["model_rgb"], frame["model_wrist_rgb"]], "lang": case.language}, inference_seed=derive_inference_seed(case.seed, step), unnorm_key=getattr(args, "unnorm_key", None))
                started = time.perf_counter(); response = client.predict_action(request); elapsed = (time.perf_counter() - started) * 1000.0; data = response.get("data", response)
                if not isinstance(data, Mapping) or "geometry" not in data: raise ValueError("geometry-enabled policy response is required")
                chunk = _squeeze_action_chunk(data.get("actions"), horizon); predictions.append(_decode_geometry_response(data["geometry"])); latency.append(float(build_latency_fields(elapsed, data.get("timing")).get("latency_ms", elapsed)))
            raw = chunk[step % horizon].copy()  # type: ignore[index]
            actions.append(raw); obs, _, done, _ = env.step(_libero_action(raw).tolist())
            if done: break
        return _finalize_record(case, frames, predictions, actions, latency, horizon, bool(done), "done" if done else "max_steps")
    finally: env.close()


def _capture_frame(env: Any, obs: Mapping[str, object], resolution: int) -> dict[str, np.ndarray]:
    rgb = np.ascontiguousarray(np.asarray(obs["agentview_image"], np.uint8)[::-1, ::-1]); wrist = np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"], np.uint8)[::-1, ::-1]); depth = np.ascontiguousarray(_metric_depth(env, np.asarray(obs["agentview_depth"]))[::-1, ::-1]).astype(np.float32)
    xyz, uvd, valid, in_frame = _realized_lrw(env, resolution, resolution)
    return {"rgb": rgb, "wrist_rgb": wrist, "depth": depth, "model_rgb": _resize_rgb(rgb, (224, 224)), "model_wrist_rgb": _resize_rgb(wrist, (224, 224)), "world_xyz": xyz, "uvd": uvd, "projection_valid": valid, "in_frame": in_frame}


def _realized_lrw(env: Any, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from robosuite.utils import camera_utils
    from starVLA.gripper_triangle import LANDMARK_BODY_NAMES, project_world_to_agentview_uvd
    body_ids = [env.sim.model.body_name2id(name) for name in LANDMARK_BODY_NAMES]; world = np.asarray(env.sim.data.body_xpos[body_ids], np.float32)
    k = camera_utils.get_camera_intrinsic_matrix(env.sim, "agentview", height, width); world_from_camera = camera_utils.get_camera_extrinsic_matrix(env.sim, "agentview")
    pixels, projection, _ = project_world_to_agentview_uvd(world[None], k, world_from_camera, width=width, height=height); pixels, projection = pixels[0], projection[0].astype(np.bool_)
    finite = np.isfinite(pixels[:, :2]).all(axis=1); pixels[finite, 0] = width - 1 - pixels[finite, 0]; pixels[finite, 1] = height - 1 - pixels[finite, 1]
    in_frame = projection & finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height); uvd = pixels.astype(np.float32); uvd[:, 0] /= width - 1; uvd[:, 1] /= height - 1
    return world, uvd, projection, in_frame.astype(np.bool_)


def _finalize_record(case: AuditCase, frames: list[dict[str, np.ndarray]], predicted: list[dict[str, np.ndarray]], actions: list[np.ndarray], latency: list[float], horizon: int, success: bool, end_reason: str) -> RolloutRecord:
    if not frames or not predicted: raise ValueError("rollout completed without policy steps")
    anchors = np.arange(0, len(frames), horizon, dtype=np.int32)
    if len(anchors) != len(predicted) or len(anchors) != len(latency): raise ValueError("cached prediction count does not match action cadence")
    prediction, times, ids = np.stack([x["uvd"] for x in predicted]).astype(np.float32), np.stack([x["uvd_time"] for x in predicted]).astype(np.float32), np.stack([x["uvd_landmark_ids"] for x in predicted]).astype(np.int64)
    realized, realized_valid = np.stack([x["uvd"] for x in frames]).astype(np.float32), np.stack([x["projection_valid"] for x in frames]).astype(np.bool_)
    targets, target_valid, metrics = complete_anchor_targets(predicted_uvd=prediction, predicted_uvd_time=times, predicted_uvd_landmark_ids=ids, anchor_steps=anchors, realized_uvd=realized, realized_valid=realized_valid, action_horizon=horizon, image_size=frames[0]["rgb"].shape[:2])
    depth_shape = predicted[0]["depth_current"].shape; current = np.stack([_resize_depth(frames[i]["depth"], depth_shape) for i in anchors]).astype(np.float32); future = np.full_like(current, np.nan)
    for index, anchor in enumerate(anchors):
        if anchor + horizon < len(frames): future[index] = _resize_depth(frames[anchor + horizon]["depth"], depth_shape)
    return RolloutRecord(np.stack([x["rgb"] for x in frames]).astype(np.uint8), np.stack([x["wrist_rgb"] for x in frames]).astype(np.uint8), np.stack([x["depth"] for x in frames]).astype(np.float32), np.asarray(actions, np.float32), anchors, prediction, times, ids, np.stack([x["depth_current"] for x in predicted]).astype(np.float32), np.stack([x["depth_future"] for x in predicted]).astype(np.float32), realized, np.stack([x["world_xyz"] for x in frames]).astype(np.float32), realized_valid, np.stack([x["in_frame"] for x in frames]).astype(np.bool_), targets, target_valid, current, future, np.asarray(latency, np.float64), {"case": asdict(case), "outcome": {"success": success, "end_reason": end_reason}, "action_horizon": horizon, "image_size": list(frames[0]["rgb"].shape[:2]), "metrics": metrics, "record_convention": {"rgb_flip": "180-degree", "realized_xyz": "world metres", "realized_valid": "projection-valid"}})


def _decode_geometry_response(geometry: object) -> dict[str, np.ndarray]:
    required = {"depth_current", "depth_future", "uvd", "uvd_time", "uvd_landmark_ids"}
    if not isinstance(geometry, Mapping) or set(geometry) != required: raise ValueError("geometry response must have exact Task-1 fields")
    uvd, times, ids = _squeeze_batch(geometry["uvd"], "uvd"), _squeeze_batch(geometry["uvd_time"], "uvd_time"), _squeeze_batch(geometry["uvd_landmark_ids"], "uvd_landmark_ids")
    if uvd.ndim != 2 or uvd.shape[1:] != (3,) or times.ndim != 1 or ids.ndim != 1 or len(uvd) != len(times) or len(uvd) != len(ids) or len(uvd) % 3: raise ValueError("geometry V3 metadata shape mismatch")
    points = len(uvd) // 3; canonicalize_v3_uvd(uvd, time_points=points, uvd_time=times, uvd_landmark_ids=ids)
    return {"uvd": uvd.reshape(points, 3, 3).astype(np.float32), "uvd_time": times.astype(np.float32), "uvd_landmark_ids": ids.astype(np.int64), "depth_current": _squeeze_depth(geometry["depth_current"], "depth_current"), "depth_future": _squeeze_depth(geometry["depth_future"], "depth_future")}


def _squeeze_action_chunk(value: object, horizon: int) -> np.ndarray:
    array = np.asarray(value, np.float32); array = array[0] if array.ndim == 3 and array.shape[0] == 1 else array
    if array.shape != (horizon, 7): raise ValueError(f"expected action chunk {(horizon, 7)}, got {array.shape}")
    return array
def _squeeze_batch(value: object, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim < 1 or value.shape[0] != 1: raise ValueError(f"geometry {name} must have leading batch size one")
    return value[0]
def _squeeze_depth(value: object, name: str) -> np.ndarray:
    value = _squeeze_batch(value, name).astype(np.float32); value = value[0] if value.ndim == 3 and value.shape[0] == 1 else value
    if value.ndim != 2: raise ValueError(f"geometry {name} must resolve to one [height,width] map")
    return value
def _libero_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, np.float32).reshape(-1)
    if action.shape != (7,): raise ValueError("executed policy action must be 7D")
    return np.concatenate((action[:6], [1.0 - 2.0 * float(action[6] > 0.5)])).astype(np.float32)
def _metric_depth(env: Any, raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, np.float32); finite = raw[np.isfinite(raw)]
    if finite.size and finite.min() >= -1e-6 and finite.max() <= 1.000001:
        from robosuite.utils import camera_utils
        return np.asarray(camera_utils.get_real_depth_map(env.sim, raw), np.float32)
    return raw
def _resize_rgb(image: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(image).resize((target[1], target[0]), Image.BILINEAR), np.uint8)
def _resize_depth(depth: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    from .sim_geometry_utils import resize_depth
    result, _ = resize_depth(depth, np.isfinite(depth) & (depth > 0), target); return result.astype(np.float32)
def _max_steps_for_suite(suite: str) -> int:
    values = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
    if suite not in values: raise ValueError(f"unknown LIBERO suite {suite!r}")
    return values[suite]
def _record_paths(path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    if path.suffix == ".npz": return path, path.with_suffix(".json")
    if path.suffix == ".json": return path.with_suffix(".npz"), path
    return path.with_suffix(".npz"), path.with_suffix(".json")
def _array_spec(array: np.ndarray) -> dict[str, object]: return {"shape": list(array.shape), "dtype": array.dtype.str}
def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()
def _validate_manifest(manifest: object) -> None:
    if not isinstance(manifest, dict) or set(manifest) != {"version", "npz_sha256", "arrays", "metadata"}: raise ValueError("rollout metadata has invalid fields")
    if manifest["version"] != RECORD_VERSION: raise ValueError(f"unsupported rollout record version {manifest['version']!r}")
    if not isinstance(manifest["npz_sha256"], str) or len(manifest["npz_sha256"]) != 64: raise ValueError("rollout metadata has invalid npz_sha256")
    if not isinstance(manifest["arrays"], dict) or set(manifest["arrays"]) != set(_ARRAY_FIELDS): raise ValueError("rollout metadata has incomplete array schema")
    if not isinstance(manifest["metadata"], dict): raise ValueError("rollout metadata payload must be a mapping")
def _require(array: np.ndarray, shape: tuple[int | None, ...], dtype: type[np.generic], name: str) -> None:
    if array.ndim != len(shape) or any(expected is not None and actual != expected for actual, expected in zip(array.shape, shape)): raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
    if array.dtype != np.dtype(dtype): raise ValueError(f"{name} dtype must be {np.dtype(dtype)}, got {array.dtype}")
