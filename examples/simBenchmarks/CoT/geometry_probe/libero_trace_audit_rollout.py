"""Resumable state-timeline LIBERO V3 rollout records.

All pure storage/cadence code is CPU-only; LIBERO imports stay lazy.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .libero_trace_audit_metrics import align_realized_trace, canonicalize_v3_uvd, compute_anchor_metrics, metrics_to_jsonable

RECORD_VERSION = 2
LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], np.float32)
_CASE_KEYS = ("suite", "task_id", "language", "rank_group", "initial_state_index", "seed", "original_success")
_ARRAY_FIELDS = ("agent_rgb", "wrist_rgb", "agent_depth", "policy_actions_raw", "executed_actions", "anchor_steps", "predicted_uvd", "predicted_uvd_time", "predicted_uvd_landmark_ids", "predicted_depth_current", "predicted_depth_future", "realized_uvd", "realized_xyz", "realized_valid", "realized_in_frame", "anchor_target_uvd", "anchor_target_valid", "dense_depth_current_target", "dense_depth_future_target", "latency_ms", "camera_k_agentview_flipped")


@dataclass(frozen=True)
class RolloutRecord:
    """State timeline: S actions and S+1 image/depth/LRW states."""
    agent_rgb: np.ndarray; wrist_rgb: np.ndarray; agent_depth: np.ndarray
    policy_actions_raw: np.ndarray; executed_actions: np.ndarray; anchor_steps: np.ndarray
    predicted_uvd: np.ndarray; predicted_uvd_time: np.ndarray; predicted_uvd_landmark_ids: np.ndarray
    predicted_depth_current: np.ndarray; predicted_depth_future: np.ndarray
    realized_uvd: np.ndarray; realized_xyz: np.ndarray; realized_valid: np.ndarray; realized_in_frame: np.ndarray
    anchor_target_uvd: np.ndarray; anchor_target_valid: np.ndarray
    dense_depth_current_target: np.ndarray; dense_depth_future_target: np.ndarray; latency_ms: np.ndarray
    camera_k_agentview_flipped: np.ndarray; metadata: dict[str, object]
    @classmethod
    def array_field_names(cls) -> tuple[str, ...]: return _ARRAY_FIELDS


@dataclass(frozen=True)
class StateTimeline:
    policy_actions_raw: np.ndarray; executed_actions: np.ndarray; state_count: int


def _libero_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, np.float32).reshape(-1)
    if action.shape != (7,): raise ValueError("policy action must be 7D")
    return np.concatenate((action[:6], [1.0 - 2.0 * float(action[6] > .5)])).astype(np.float32)


def finalize_state_timeline(policy_actions_raw: np.ndarray, states: Sequence[object]) -> StateTimeline:
    raw = np.asarray(policy_actions_raw, np.float32)
    if raw.ndim != 2 or raw.shape[1:] != (7,): raise ValueError("policy_actions_raw must be [action,7]")
    if len(states) != len(raw) + 1: raise ValueError("state timeline must contain initial plus one post-action state")
    return StateTimeline(raw, np.stack([_libero_action(x) for x in raw]), len(states))


def derive_inference_seed(audit_seed: int, anchor_step: int) -> int:
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in (audit_seed, anchor_step)) or audit_seed < 0 or not 0 <= anchor_step < 2**16: raise ValueError("invalid audit seed or anchor")
    return (int(audit_seed) << 16) | int(anchor_step)


def build_geometry_request(observation: Mapping[str, object], *, inference_seed: int, unnorm_key: str | None = None) -> dict[str, object]:
    images = observation.get("image")
    if not isinstance(images, Sequence) or len(images) != 2: raise ValueError("need primary then wrist image")
    request: dict[str, object] = {"examples": [{"image": [images[0], images[1]], "lang": str(observation.get("lang", ""))}], "do_sample": False, "use_ddim": True, "num_ddim_steps": 10, "return_geometry": True, "inference_seed": int(inference_seed)}
    if unnorm_key is not None: request["unnorm_key"] = unnorm_key
    return request


def flip_camera_intrinsics(camera_k: np.ndarray, *, image_size: tuple[int, int]) -> np.ndarray:
    """K after the same 180-degree RGB/depth rotation (H_180 @ K)."""
    h, w = image_size; k = np.asarray(camera_k, np.float32)
    if k.shape != (3, 3) or h < 2 or w < 2: raise ValueError("camera K/image size invalid")
    return np.asarray([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], np.float32) @ k


def complete_anchor_targets(*, predicted_uvd: np.ndarray, predicted_uvd_time: np.ndarray, predicted_uvd_landmark_ids: np.ndarray, anchor_steps: np.ndarray, realized_uvd: np.ndarray, realized_valid: np.ndarray, action_horizon: int, image_size: int | tuple[int, int], camera_k: np.ndarray | None = None, depth_dead_zone_m: float = .002) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    pred, times, ids, anchors = np.asarray(predicted_uvd, np.float32), np.asarray(predicted_uvd_time), np.asarray(predicted_uvd_landmark_ids), np.asarray(anchor_steps)
    if pred.ndim != 4 or pred.shape[2:] != (3, 3) or anchors.dtype != np.int32 or anchors.shape != (len(pred),): raise ValueError("invalid V3 anchor arrays")
    count, points = pred.shape[:2]
    if times.shape != (count, points * 3) or ids.shape != times.shape: raise ValueError("V3 metadata shape mismatch")
    targets = np.full_like(pred, np.nan); valid = np.zeros(pred.shape[:-1], bool); out: dict[str, object] = {}
    for n, anchor in enumerate(anchors):
        canonical = canonicalize_v3_uvd(pred[n].reshape(points * 3, 3), time_points=points, uvd_time=times[n], uvd_landmark_ids=ids[n])
        offset_f = times[n].reshape(points, 3)[:, 0] * action_horizon; offsets = np.rint(offset_f).astype(np.int64)
        if np.any(offsets < 0) or np.any(offsets > action_horizon) or not np.allclose(offset_f, offsets): raise ValueError("V3 time does not map to action horizon")
        target, mask = align_realized_trace(realized_uvd, int(anchor), offsets, step_valid=realized_valid)
        targets[n], valid[n] = target, mask
        out[f"anchor_{n}"] = metrics_to_jsonable(compute_anchor_metrics(canonical, target, mask, image_size=image_size, camera_k=camera_k, depth_dead_zone_m=depth_dead_zone_m, uvd_time=times[n], uvd_landmark_ids=ids[n]))
    json.dumps(out, allow_nan=False); return targets, valid, out


def execute_cadenced_actions(*, max_steps: int, action_horizon: int, audit_seed: int, dummy_steps: int, stabilize: Callable[[np.ndarray], object], observe: Callable[[int], Mapping[str, object]], request: Callable[[dict[str, object]], Mapping[str, object]], execute: Callable[[np.ndarray, int], bool], unnorm_key: str | None = None) -> tuple[np.ndarray, np.ndarray, tuple[Mapping[str, object], ...]]:
    for _ in range(dummy_steps): stabilize(LIBERO_DUMMY_ACTION.copy())
    anchors: list[int] = []; raw: list[np.ndarray] = []; response_data: list[Mapping[str, object]] = []; chunk = None
    for step in range(max_steps):
        if step % action_horizon == 0:
            anchors.append(step); data = request(build_geometry_request(observe(step), inference_seed=derive_inference_seed(audit_seed, step), unnorm_key=unnorm_key)).get("data", request)  # type: ignore[union-attr]
            if not isinstance(data, Mapping) or "actions" not in data: raise ValueError("policy response lacks actions")
            chunk = np.asarray(data["actions"], np.float32); chunk = chunk[0] if chunk.ndim == 3 and chunk.shape[0] == 1 else chunk
            if chunk.shape != (action_horizon, 7): raise ValueError("action chunk shape")
            response_data.append(data)
        action = chunk[step % action_horizon].copy(); raw.append(action)  # type: ignore[index]
        if execute(_libero_action(action), step): break
    return np.asarray(anchors, np.int32), np.asarray(raw, np.float32), tuple(response_data)


def save_rollout_record(path: str | Path, record: RolloutRecord) -> tuple[Path, Path]:
    """Publish only a complete generation via one atomic manifest pointer."""
    validate_rollout_record(record); base = Path(path); base.parent.mkdir(parents=True, exist_ok=True); generation = f"{base.name}.v{RECORD_VERSION}.{os.urandom(8).hex()}"; npz = base.parent / f"{generation}.npz"; data = base.parent / f"{generation}.json"; manifest = base.with_suffix(".manifest.json")
    arrays = {name: np.asarray(getattr(record, name)) for name in _ARRAY_FIELDS}
    with tempfile.NamedTemporaryFile(dir=base.parent, suffix=".npz", delete=False) as handle: temp_npz = Path(handle.name)
    temp_data: Path | None = None; temp_manifest: Path | None = None
    try:
        np.savez_compressed(temp_npz, **arrays); digest = _sha(temp_npz); payload = {"version": RECORD_VERSION, "npz": npz.name, "data": data.name, "sha256": digest, "arrays": {k: _spec(v) for k, v in arrays.items()}, "metadata": metrics_to_jsonable(record.metadata)}
        _strict_json(payload)
        with tempfile.NamedTemporaryFile(dir=base.parent, suffix=".json", mode="w", encoding="utf-8", delete=False) as handle: temp_data = Path(handle.name); handle.write(json.dumps(payload, allow_nan=False, sort_keys=True))
        os.replace(temp_npz, npz); os.replace(temp_data, data)
        with tempfile.NamedTemporaryFile(dir=base.parent, suffix=".manifest.json", mode="w", encoding="utf-8", delete=False) as handle: temp_manifest = Path(handle.name); handle.write(json.dumps({"version": RECORD_VERSION, "generation": generation}, allow_nan=False))
        os.replace(temp_manifest, manifest)
    finally:
        for temp in (temp_npz, temp_data, temp_manifest):
            if temp is not None: temp.unlink(missing_ok=True)
    return npz, manifest


def load_rollout_record(path: str | Path, *, expected_case: Mapping[str, object] | None = None, expected_config_identity: Mapping[str, object] | None = None) -> RolloutRecord:
    base = Path(path); manifest = base.with_suffix(".manifest.json")
    if not manifest.is_file(): raise FileNotFoundError("no published rollout manifest")
    try: pointer = json.loads(manifest.read_text())
    except json.JSONDecodeError as error: raise ValueError("corrupt rollout manifest") from error
    if not isinstance(pointer, dict) or pointer.get("version") != RECORD_VERSION or not isinstance(pointer.get("generation"), str): raise ValueError("unsupported or partial rollout manifest")
    root = manifest.parent; npz = root / f"{pointer['generation']}.npz"; data = root / f"{pointer['generation']}.json"
    if not npz.is_file() or not data.is_file(): raise ValueError("manifest points to partial generation")
    try: payload = json.loads(data.read_text())
    except json.JSONDecodeError as error: raise ValueError("corrupt generation metadata") from error
    _validate_payload(payload, npz)
    with np.load(npz, allow_pickle=False) as source:
        if set(source.files) != set(_ARRAY_FIELDS): raise ValueError("generation has incorrect array keys")
        arrays = {key: np.asarray(source[key]) for key in _ARRAY_FIELDS}
    if any(payload["arrays"].get(key) != _spec(value) for key, value in arrays.items()): raise ValueError("generation array schema mismatch")
    record = RolloutRecord(**arrays, metadata=payload["metadata"]); validate_rollout_record(record); _check_identity(record, expected_case, expected_config_identity); return record


def _check_identity(record: RolloutRecord, case: Mapping[str, object] | None, config: Mapping[str, object] | None) -> None:
    if case is not None and dict(case) != record.metadata["case"]: raise ValueError("expected_case does not match rollout")
    if config is not None and any(record.metadata.get(key) != value for key, value in config.items()): raise ValueError("expected_config_identity does not match rollout")


def validate_rollout_record(record: RolloutRecord) -> None:
    v = {key: np.asarray(getattr(record, key)) for key in _ARRAY_FIELDS}; _need(v["agent_rgb"], (None, None, None, 3), np.uint8, "agent_rgb"); states, h, w, _ = v["agent_rgb"].shape
    if states < 2: raise ValueError("state timeline needs initial and post-action state")
    for key in ("wrist_rgb",): _need(v[key], (states, h, w, 3), np.uint8, key)
    _need(v["agent_depth"], (states, h, w), np.float32, "agent_depth"); actions = states - 1
    for key in ("policy_actions_raw", "executed_actions"): _need(v[key], (actions, 7), np.float32, key)
    if not np.array_equal(v["executed_actions"], np.stack([_libero_action(x) for x in v["policy_actions_raw"]])): raise ValueError("executed_actions must be postprocessed policy_actions_raw")
    for key in ("realized_uvd", "realized_xyz"): _need(v[key], (states, 3, 3), np.float32, key)
    for key in ("realized_valid", "realized_in_frame"): _need(v[key], (states, 3), np.bool_, key)
    if np.any(v["realized_in_frame"] & ~v["realized_valid"]): raise ValueError("in_frame exceeds projection valid")
    m = record.metadata
    if not isinstance(m, dict) or set(("case", "outcome", "action_horizon", "image_size", "metrics", "schema")) - set(m) or m["schema"] != "state_timeline_v2": raise ValueError("invalid metadata schema")
    if not isinstance(m["case"], dict) or set(m["case"]) != set(_CASE_KEYS) or not isinstance(m["outcome"], dict) or set(m["outcome"]) != {"success", "end_reason"} or not isinstance(m["outcome"]["success"], bool) or not isinstance(m["outcome"]["end_reason"], str): raise ValueError("invalid case/outcome metadata")
    horizon = m["action_horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1 or m["image_size"] != [h, w]: raise ValueError("invalid config metadata")
    anchors = v["anchor_steps"]
    if anchors.dtype != np.int32 or anchors.ndim != 1 or len(anchors) < 1 or anchors[0] != 0 or np.any(np.diff(anchors) != horizon) or np.any(anchors >= actions): raise ValueError("invalid anchor cadence")
    count = len(anchors); pred = v["predicted_uvd"]
    if pred.dtype != np.float32 or pred.ndim != 4 or pred.shape[0] != count or pred.shape[2:] != (3, 3): raise ValueError("invalid predicted UVD")
    points = pred.shape[1]
    _need(v["predicted_uvd_time"], (count, points * 3), np.float32, "predicted_uvd_time"); _need(v["predicted_uvd_landmark_ids"], (count, points * 3), np.int64, "predicted_uvd_landmark_ids")
    depth_shape = v["predicted_depth_current"].shape
    for key in ("predicted_depth_current", "predicted_depth_future", "dense_depth_current_target", "dense_depth_future_target"):
        if v[key].dtype != np.float32 or v[key].ndim != 3 or v[key].shape[0] != count or v[key].shape[1:] != depth_shape[1:]: raise ValueError("invalid dense depth array")
    _need(v["anchor_target_uvd"], pred.shape, np.float32, "anchor_target_uvd"); _need(v["anchor_target_valid"], pred.shape[:-1], np.bool_, "anchor_target_valid"); _need(v["latency_ms"], (count,), np.float64, "latency_ms"); _need(v["camera_k_agentview_flipped"], (3, 3), np.float32, "camera_k_agentview_flipped")
    for n in range(count): canonicalize_v3_uvd(pred[n].reshape(points * 3, 3), time_points=points, uvd_time=v["predicted_uvd_time"][n], uvd_landmark_ids=v["predicted_uvd_landmark_ids"][n])
    _strict_json(metrics_to_jsonable(m))


def _need(value: np.ndarray, shape: tuple[int | None, ...], dtype: type[np.generic], name: str) -> None:
    if value.dtype != np.dtype(dtype) or value.ndim != len(shape) or any(a != b for a, b in zip(value.shape, shape) if b is not None): raise ValueError(f"invalid {name} dtype/shape")
def _spec(value: np.ndarray) -> dict[str, object]: return {"shape": list(value.shape), "dtype": value.dtype.str}
def _sha(path: Path) -> str:
    digest = sha256();
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()
def _strict_json(value: object) -> None: json.dumps(metrics_to_jsonable(value), allow_nan=False)
def _validate_payload(payload: object, npz: Path) -> None:
    if not isinstance(payload, dict) or set(payload) != {"version", "npz", "data", "sha256", "arrays", "metadata"} or payload["version"] != RECORD_VERSION or payload["npz"] != npz.name or not isinstance(payload["sha256"], str) or _sha(npz) != payload["sha256"]: raise ValueError("invalid or mixed rollout generation")
    if not isinstance(payload["arrays"], dict) or set(payload["arrays"]) != set(_ARRAY_FIELDS) or not isinstance(payload["metadata"], dict): raise ValueError("partial rollout generation")
