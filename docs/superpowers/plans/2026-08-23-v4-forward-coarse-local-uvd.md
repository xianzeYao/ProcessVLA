# QwenGR00TCoT V4 Forward Coarse→Local UVD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated QwenGR00TCoTV4 for LIBERO and RoboCasa in which a forward stride-2 coarse UVD plan causally precedes a dense local UVD plan and the action head, while restoring V2/V3 to their pre-reverse-full behavior.

**Architecture:** V4 keeps the V2 no-query, depth-conditioned Qwen path and appends `[depth_current, depth_future, coarse, local]` query groups. For action horizon `H`, local predicts `H` points at offsets `1..H`, coarse predicts `H` points at offsets `2,4,..,2H`, and both use terminal-repeat rather than horizon padding. LIBERO uses `H=8`, one hand; RoboCasa uses `H=16`, two hands in time-major order.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, OmegaConf, Hugging Face Transformers/Qwen3.5, Accelerate, DeepSpeed ZeRO-2, pytest, Bash, tmux.

**Spec:** `docs/superpowers/specs/2026-08-23-v4-forward-coarse-local-uvd-design.md`

## Global Constraints

- Active V2/V3 source, tests, trainer metrics, and YAMLs must contain no `uvd_full`/`full_uvd` execution path; the removed implementation remains recoverable through commits `868aad3`, `d2c9ead`, `1cf29f3`, and `3711c93`.
- For every benchmark, `local_uvd_num_points == coarse_uvd_num_points == action_horizon` and `coarse_uvd_stride == 2`.
- Local indices are `min(t + j + 1, T)`; coarse indices are `min(t + 2*(j + 1), T)`.
- Horizon overflow repeats terminal frame `T` and stays supervised; only genuine projection/depth validity may mask a repeated point.
- Geometry order is `[native, depth_current, depth_future, coarse, local]`; coarse cannot read local, while every local token can read the complete coarse group.
- Coarse and local use separate learned seeds, time-embedding MLPs, regression heads, losses, outputs, token diagnostics, and gradient diagnostics.
- The first V4 uses direct UVD regression only: no flow head, ground-truth coarse teacher forcing, object anchor, reverse trajectory, or cross-scale consistency loss.
- Loss weights are action `1.0`, current depth `0.14`, future depth `0.15`, local UVD `0.62`, coarse UVD `0.20`; each UVD group uses absolute plus `0.1 * relative` loss with its own masked mean.
- LIBERO and RoboCasa both receive unit tests, config dry-runs, real smoke runs, and timing gates. Only the already authorized LIBERO 60k formal run is started automatically; RoboCasa 100k is left ready to launch.
- Preserve unrelated user changes and use `apply_patch` for repository file edits.

## File and Responsibility Map

- `starVLA/dataloader/gr00t_lerobot/cot_geometry.py`: clean V2 projection, depth, and sparse-local dataset behavior only.
- `starVLA/dataloader/gr00t_lerobot/cot_geometry_v4.py`: V4 horizon contract, forward terminal-repeat sampler, target conversion, and V4 single-episode dataset.
- `starVLA/dataloader/cot_v4_lerobot_datasets.py`: LIBERO V4 dataset factory.
- `starVLA/dataloader/robocasa_v4_lerobot_datasets.py`: RoboCasa V4 bilateral pinch geometry loader and dataset factory.
- `starVLA/model/modules/geometric_cot_v2.py`: restored V2 depth/local layout only.
- `starVLA/model/modules/geometric_cot_v3.py`: restored V3 layout without a synthetic full-UVD slice.
- `starVLA/model/modules/geometric_cot_v4.py`: V4 layout, embeddings, mask, fixed packing, and depth pooling helpers.
- `starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py`: restored V2 framework only.
- `starVLA/model/framework/VLM4A/QwenGR00TCoTV4.py`: standalone V4 framework, losses, predictions, and interventions.
- `starVLA/training/train_starvla_cot_v1.py`: version-neutral metric extension hooks; no reverse-full metric names.
- `starVLA/training/cot_test_diagnostics.py`: version-neutral named-UVD metric primitives and optional module discovery.
- `starVLA/training/cot_v4_diagnostics.py`: cross-scale and terminal-repeat V4 diagnostics.
- `starVLA/training/train_starvla_cot_v4.py`: V4 trainer and CLI.
- `examples/modelExtensions/CoT/configs/*CoT_v4*.yaml`: one q0-depth V4 experiment per benchmark.
- `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v4_common.sh`: shared Accelerate launcher.
- `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_*_CoT_v4.sh`: benchmark wrappers.
- `examples/modelExtensions/CoT/scripts/compare_cot_v4_timing.py`: deterministic JSONL timing/memory gate.
- `tests/test_*v4*.py`: V4-only unit, integration, launcher, diagnostics, and timing tests.

---

### Task 1: Restore the V2/V3 Contract and Remove the Active Reverse-Full Experiment

**Files:**
- Modify: `starVLA/dataloader/gr00t_lerobot/cot_geometry.py`
- Modify: `starVLA/model/modules/geometric_cot_v2.py`
- Modify: `starVLA/model/modules/geometric_cot_v3.py`
- Modify: `starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py`
- Modify: `starVLA/training/train_starvla_cot_v1.py`
- Delete: `examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v2_q0_depthcond_reverse_full_uvd_s4.yaml`
- Modify: `tests/test_cot_geometry.py`
- Modify: `tests/test_geometric_cot_v2.py`
- Modify: `tests/test_qwen_gr00t_cot_v2.py`
- Modify: `tests/test_cot_trainer_objective.py`
- Modify: `tests/test_cot_v2_entrypoints.py`

**Interfaces:**
- Consumes: clean behavior recorded immediately before commit `868aad3` plus later non-full V2 diagnostics.
- Produces: V2 `GeometryTokenLayout(depth_query_count, uvd_points_per_hand, hand_count)` and V2 outputs containing local `uvd` only.

- [ ] **Step 1: Add failing isolation regressions before removing the old path**

Add focused assertions while retaining all non-full tests:

```python
import inspect

def test_v2_contract_has_no_full_trajectory_extension(self):
    assert "full_uvd_points_per_hand" not in inspect.signature(GeometryTokenLayout).parameters
    assert not hasattr(cot_geometry_module, "sample_reverse_uvd_indices")
    assert "uvd_full" not in GeometrySequenceSlices.__dataclass_fields__


def test_reverse_full_experiment_is_not_an_active_v2_yaml():
    path = ROOT / "examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v2_q0_depthcond_reverse_full_uvd_s4.yaml"
    assert not path.exists()
```

Remove the reverse-full positive tests only after these isolation regressions exist.

- [ ] **Step 2: Run the new regressions and verify the current tree fails**

Run:

```bash
pytest -q \
  tests/test_cot_geometry.py::CotGeometryTest::test_v2_contract_has_no_full_trajectory_extension \
  tests/test_cot_v2_entrypoints.py::test_reverse_full_experiment_is_not_an_active_v2_yaml
```

Expected: FAIL because `sample_reverse_uvd_indices`, the layout field, and the YAML still exist.

- [ ] **Step 3: Remove only the four reverse-full commit changes**

Use the parent versions of the touched regions as the source of truth:

```text
868aad3^ -> cot_geometry.py and its reverse-target tests
d2c9ead^ -> geometric_cot_v2.py, geometric_cot_v3.py, and full-layout tests
1cf29f3^ -> QwenGR00TCoTV2.py and full-framework tests
3711c93^ -> trainer full metrics, trainer test, entrypoint test, and reverse YAML
```

Keep later V2 attention pooling, q0 depth conditioning, gradient diagnostics, token utilization, action interventions, timing, and checkpoint guards. The resulting V2 sequence is exactly:

```text
[native, depth_current, depth_future, local_uvd]
```

The V2 trainer base must have no `uvd_full_*` metric aggregation.

- [ ] **Step 4: Run V2/V3 regression tests**

Run:

```bash
pytest -q \
  tests/test_cot_geometry.py \
  tests/test_geometric_cot_v2.py \
  tests/test_qwen_gr00t_cot_v2.py \
  tests/test_geometric_cot_v3.py \
  tests/test_qwen_gr00t_cot_v3.py \
  tests/test_cot_trainer_objective.py \
  tests/test_cot_v2_entrypoints.py \
  tests/test_cot_v3_entrypoints.py
```

Expected: PASS.

- [ ] **Step 5: Prove no active reverse-full symbols remain**

Run:

```bash
rg -n "uvd_full|full_uvd|sample_reverse_uvd" \
  starVLA/dataloader starVLA/model starVLA/training \
  examples/modelExtensions/CoT/configs tests
```

Expected: no matches. Documentation and Git history are intentionally outside this scan.

- [ ] **Step 6: Commit the restored V2/V3 boundary**

```bash
git add -u \
  starVLA/dataloader/gr00t_lerobot/cot_geometry.py \
  starVLA/model/modules/geometric_cot_v2.py \
  starVLA/model/modules/geometric_cot_v3.py \
  starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py \
  starVLA/training/train_starvla_cot_v1.py \
  examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v2_q0_depthcond_reverse_full_uvd_s4.yaml \
  tests/test_cot_geometry.py \
  tests/test_geometric_cot_v2.py \
  tests/test_qwen_gr00t_cot_v2.py \
  tests/test_cot_trainer_objective.py \
  tests/test_cot_v2_entrypoints.py
git commit -m "refactor: restore isolated V2 geometry contract"
```

---

### Task 2: Add V4 Forward Horizons and the LIBERO Dataset Adapter

**Files:**
- Create: `starVLA/dataloader/gr00t_lerobot/cot_geometry_v4.py`
- Create: `starVLA/dataloader/cot_v4_lerobot_datasets.py`
- Create: `tests/test_cot_v4_geometry.py`
- Create: `tests/test_cot_v4_lerobot_dataset.py`

**Interfaces:**
- Consumes: V2 `_EpisodeGeometryCache`, `_read_npz_array`, `_resize_depth`, `project_eef_to_agentview_uvd`, and `transform_uvd_to_model_space`.
- Produces: `V4HorizonSpec`, `sample_forward_uvd_indices`, `CoTV4LeRobotSingleDataset`, and LIBERO `get_vla_dataset`.

- [ ] **Step 1: Write failing sampler and contract tests**

```python
def test_forward_horizons_have_equal_point_counts_and_double_coarse_span():
    spec = V4HorizonSpec(
        action_horizon=8,
        local_uvd_num_points=8,
        coarse_uvd_num_points=8,
        coarse_uvd_stride=2,
        terminal_repeat=True,
    )
    np.testing.assert_array_equal(spec.local_indices(3, 30), np.arange(4, 12))
    np.testing.assert_array_equal(spec.coarse_indices(3, 30), np.arange(5, 20, 2))


def test_forward_horizons_repeat_terminal_without_padding_slots():
    spec = V4HorizonSpec(8, 8, 8, 2, True)
    np.testing.assert_array_equal(spec.local_indices(6, 7), np.full(8, 7))
    np.testing.assert_array_equal(spec.coarse_indices(6, 7), np.full(8, 7))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"local_uvd_num_points": 7}, "local.*action_horizon"),
        ({"coarse_uvd_num_points": 9}, "coarse.*action_horizon"),
        ({"coarse_uvd_stride": 1}, "stride.*2"),
        ({"terminal_repeat": False}, "terminal_repeat.*true"),
    ],
)
def test_v4_horizon_contract_fails_fast(kwargs, message):
    values = dict(action_horizon=8, local_uvd_num_points=8,
                  coarse_uvd_num_points=8, coarse_uvd_stride=2,
                  terminal_repeat=True)
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        V4HorizonSpec(**values)
```

- [ ] **Step 2: Run the sampler tests and verify imports fail**

Run:

```bash
pytest -q tests/test_cot_v4_geometry.py -k "forward_horizons or horizon_contract"
```

Expected: collection ERROR because `cot_geometry_v4` does not exist.

- [ ] **Step 3: Implement the fixed forward horizon API**

Use these exact public signatures:

```python
@dataclass(frozen=True)
class V4HorizonSpec:
    action_horizon: int
    local_uvd_num_points: int
    coarse_uvd_num_points: int
    coarse_uvd_stride: int
    terminal_repeat: bool

    @classmethod
    def from_data_cfg(cls, data_cfg: Any) -> "V4HorizonSpec":
        geometry = data_cfg.get("cot_geometry", {})
        horizon = int(geometry.get("action_horizon", 8))
        return cls(
            action_horizon=horizon,
            local_uvd_num_points=int(geometry.get("local_uvd_num_points", horizon)),
            coarse_uvd_num_points=int(geometry.get("coarse_uvd_num_points", horizon)),
            coarse_uvd_stride=int(geometry.get("coarse_uvd_stride", 2)),
            terminal_repeat=bool(geometry.get("terminal_repeat", False)),
        )

    def local_indices(self, start: int, terminal: int) -> np.ndarray:
        return sample_forward_uvd_indices(
            start, terminal, num_points=self.local_uvd_num_points, stride=1
        )

    def coarse_indices(self, start: int, terminal: int) -> np.ndarray:
        return sample_forward_uvd_indices(
            start,
            terminal,
            num_points=self.coarse_uvd_num_points,
            stride=self.coarse_uvd_stride,
        )


def sample_forward_uvd_indices(
    start: int,
    terminal: int,
    *,
    num_points: int,
    stride: int,
) -> np.ndarray:
    if terminal < start:
        raise ValueError(f"terminal must be >= start, got {start}/{terminal}")
    if num_points < 1 or stride < 1:
        raise ValueError("num_points and stride must be positive")
    offsets = stride * np.arange(1, num_points + 1, dtype=np.int64)
    return np.minimum(start + offsets, terminal)
```

`V4HorizonSpec.__post_init__` enforces all four global horizon constraints.

- [ ] **Step 4: Write failing LIBERO target tests**

Construct `CoTV4LeRobotSingleDataset` with `__new__`, inject a 20-frame episode, and assert:

```python
targets = dataset._geometry_targets()
np.testing.assert_array_equal(targets["uvd_frame_indices"], np.arange(3, 11))
np.testing.assert_array_equal(targets["uvd_coarse_frame_indices"], np.arange(4, 19, 2))
np.testing.assert_allclose(targets["uvd_time"], np.arange(1, 9) / 8)
np.testing.assert_allclose(targets["uvd_coarse_time"], np.arange(1, 9) / 8)
assert targets["uvd"].shape == (8, 3)
assert targets["uvd_coarse"].shape == (8, 3)
assert targets["uvd_valid_mask"].all()
assert targets["uvd_coarse_valid_mask"].all()
```

Add a `t=T-1` case that asserts eight terminal indices, eight supervised valid entries when terminal geometry is valid, and eight invalid entries when only terminal projection validity is false.

- [ ] **Step 5: Implement V4 target conversion and the LIBERO factory**

`CoTV4LeRobotSingleDataset._geometry_targets()` must:

```text
1. Load `(depth, eef_uvd, eef_valid, state)` through the inherited cache/loader.
2. Clamp base index to `T` and build local/coarse indices through V4HorizonSpec.
3. Decode current depth at `t` and future depth at `min(t+H,T)`.
4. Transform both UVD groups independently into model space.
5. Preserve `[time, hand, 3]` when an episode has a hand axis.
6. Emit local `uvd*` keys and coarse `uvd_coarse*` keys with fixed nominal time.
7. Set endpoint indices to `[0, H-1]`; never create `-1` padding indices.
```

The LIBERO factory mirrors its existing modality/action definitions but instantiates only `CoTV4LeRobotSingleDataset`.

- [ ] **Step 6: Run V4 LIBERO data tests plus V2 regression**

```bash
pytest -q \
  tests/test_cot_v4_geometry.py \
  tests/test_cot_v4_lerobot_dataset.py \
  tests/test_cot_geometry.py
```

Expected: PASS.

- [ ] **Step 7: Commit the V4 LIBERO data contract**

```bash
git add -f \
  starVLA/dataloader/gr00t_lerobot/cot_geometry_v4.py \
  starVLA/dataloader/cot_v4_lerobot_datasets.py \
  tests/test_cot_v4_geometry.py \
  tests/test_cot_v4_lerobot_dataset.py
git commit -m "feat: add forward terminal-repeated V4 targets"
```

---

### Task 3: Add the RoboCasa V4 Bilateral Dataset Adapter

**Files:**
- Create: `starVLA/dataloader/robocasa_v4_lerobot_datasets.py`
- Create: `tests/test_robocasa_v4_lerobot_dataset.py`

**Interfaces:**
- Consumes: `CoTV4LeRobotSingleDataset`, `RoboCasaGR1DataConfig`, `dataset_specs`, `select_robocasa_uvd_world_columns`, and `dial_content_region_mask`.
- Produces: `RoboCasaV4CoTLeRobotSingleDataset` and RoboCasa V4 `get_vla_dataset`.

- [ ] **Step 1: Write failing dual-hand horizon and loader tests**

Inject a 40-frame, two-hand UVD episode into an uninitialized dataset and assert:

```python
targets = dataset._geometry_targets()
assert targets["uvd"].shape == (16, 2, 3)
assert targets["uvd_coarse"].shape == (16, 2, 3)
assert targets["uvd_valid_mask"].shape == (16, 2)
np.testing.assert_array_equal(targets["uvd_frame_indices"], np.arange(6, 22))
np.testing.assert_array_equal(targets["uvd_coarse_frame_indices"], np.arange(7, 38, 2))
```

Add a loader test with distinct left/right world coordinates and validity masks, proving the output hand order remains `[left, right]` and the dial mask is applied per hand.

- [ ] **Step 2: Run the new test and verify its module import fails**

```bash
pytest -q tests/test_robocasa_v4_lerobot_dataset.py
```

Expected: collection ERROR because `robocasa_v4_lerobot_datasets` does not exist.

- [ ] **Step 3: Implement the RoboCasa V4 loader and factory**

Define:

```python
class RoboCasaV4CoTLeRobotSingleDataset(CoTV4LeRobotSingleDataset):
    def _load_episode_geometry(
        self, trajectory_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cot_cache.get(trajectory_id)
        if cached is not None:
            return cached
        if self.curr_traj_data is None:
            raise RuntimeError("trajectory data is not loaded")
        row0 = self.curr_traj_data.iloc[0]
        depth = _read_npz_array(
            self.dataset_path / str(row0["observation.depth.image_m_path"]), "depth_m"
        )
        camera = self.dataset_path / str(row0["observation.camera.params_path"])
        k = _read_npz_array(camera, "agentview_K").astype(np.float32)
        pose = _read_npz_array(camera, "agentview_T_world_camera").astype(np.float32)
        left_key, right_key = select_robocasa_uvd_world_columns(self.curr_traj_data.columns)
        left = np.stack(self.curr_traj_data[left_key].to_numpy()).astype(np.float32)
        right = np.stack(self.curr_traj_data[right_key].to_numpy()).astype(np.float32)
        world = np.stack([left, right], axis=1)
        uvd, valid = project_eef_to_agentview_uvd(
            world, k, pose, width=int(depth.shape[-1]), height=int(depth.shape[-2])
        )
        valid = dial_content_region_mask(uvd, valid, output_size=int(depth.shape[-1]))
        state = np.stack(self.curr_traj_data["observation.state"].to_numpy()).astype(np.float32)
        value = (depth, uvd, valid, state)
        self._cot_cache.put(trajectory_id, value)
        return value


def get_vla_dataset(
    data_cfg: Any,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    **kwargs: Any,
):
    root = Path(str(data_cfg.data_root_dir))
    paths, weights = dataset_specs(data_cfg, root)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"RoboCasa dataset does not exist: {missing}")
    config = RoboCasaGR1DataConfig()
    datasets = [
        RoboCasaV4CoTLeRobotSingleDataset(
            path,
            modality_configs=config.modality_config(),
            transforms=config.transform(),
            embodiment_tag=config.embodiment_tag,
            video_backend=data_cfg.get("video_backend", "torchvision_av"),
            delete_pause_frame=bool(data_cfg.get("delete_pause_frame", False)),
            data_cfg=data_cfg,
        )
        for path in paths
    ]
    return LeRobotMixtureDataset(
        list(zip(datasets, weights)),
        mode=mode,
        balance_dataset_weights=bool(balance_dataset_weights),
        balance_trajectory_weights=bool(balance_trajectory_weights),
        seed=int(data_cfg.get("seed", 42)),
        data_cfg=data_cfg,
        **kwargs,
    )
```

The loader must select the current pinch columns rather than hard-code an alternate EEF field, stack left/right on axis 1, project with the existing agentview calibration, apply `dial_content_region_mask`, and cache `(depth, uvd, valid, state)`.

- [ ] **Step 4: Run bilateral, factory, and existing RoboCasa tests**

```bash
pytest -q \
  tests/test_robocasa_v4_lerobot_dataset.py \
  tests/test_paired_geometry_probe.py -k robocasa
```

Expected: PASS.

- [ ] **Step 5: Commit RoboCasa V4 data support**

```bash
git add -f \
  starVLA/dataloader/robocasa_v4_lerobot_datasets.py \
  tests/test_robocasa_v4_lerobot_dataset.py
git commit -m "feat: add RoboCasa V4 geometry adapter"
```

---

### Task 4: Build the V4 Coarse→Local Geometry Token Module

**Files:**
- Create: `starVLA/model/modules/geometric_cot_v4.py`
- Create: `tests/test_geometric_cot_v4.py`

**Interfaces:**
- Consumes: example dictionaries emitted by Task 2/3 and V2 `SharedDepthAttentionPool` behavior.
- Produces: `GeometryTokenLayout`, `GeometrySequenceSlices`, `GeometryTokenEmbedding`, `PackedUVDTargets`, `append_geometry_slots`, `build_geometry_full_attention_mask`, `pack_local_uvd_targets_time_major`, and `pack_coarse_uvd_targets_time_major`.

- [ ] **Step 1: Write failing layout, mask, and embedding tests**

```python
def test_v4_layout_orders_coarse_before_local():
    layout = GeometryTokenLayout(
        depth_query_count=2,
        local_uvd_points_per_hand=3,
        coarse_uvd_points_per_hand=3,
        hand_count=2,
    )
    slices = layout.sequence_slices(native_token_count=4)
    assert layout.geometry_token_count == 16
    assert slices.depth_current == slice(4, 6)
    assert slices.depth_future == slice(6, 8)
    assert slices.uvd_coarse == slice(8, 14)
    assert slices.uvd_local == slice(14, 20)


def test_local_reads_all_coarse_but_coarse_cannot_read_local():
    layout = GeometryTokenLayout(1, 2, 2, 1)
    mask = build_geometry_full_attention_mask(
        torch.ones(1, 2 + layout.geometry_token_count, dtype=torch.bool), layout
    )[0, 0]
    slices = layout.sequence_slices(2)
    assert mask[slices.uvd_local.start, slices.uvd_coarse].all()
    assert not mask[slices.uvd_coarse.stop - 1, slices.uvd_local.start]


def test_scales_have_independent_seed_time_mlp_and_head_inputs():
    module = GeometryTokenEmbedding(hidden_dim=8, layout=GeometryTokenLayout(1, 2, 2, 1))
    assert module.coarse_trajectory_seed is not module.local_trajectory_seed
    assert module.coarse_time_embedding is not module.local_time_embedding
```

- [ ] **Step 2: Write failing strict packing tests**

Test local/coarse time-major packing for one and two hands. A target with fewer than exactly `H` temporal rows must raise instead of silently padding:

```python
with pytest.raises(ValueError, match="exactly 3 time points"):
    pack_local_uvd_targets_time_major(short_examples, layout, device=torch.device("cpu"))
```

- [ ] **Step 3: Run the module tests and verify import failure**

```bash
pytest -q tests/test_geometric_cot_v4.py
```

Expected: collection ERROR because `geometric_cot_v4` does not exist.

- [ ] **Step 4: Implement layout, embeddings, and attention**

Use these layout fields and token order:

```python
@dataclass(frozen=True)
class GeometrySequenceSlices:
    native: slice
    depth_current: slice
    depth_future: slice
    uvd_coarse: slice
    uvd_local: slice


@dataclass(frozen=True)
class PackedUVDTargets:
    target: torch.Tensor
    valid: torch.Tensor
    times: torch.Tensor
    hand_ids: torch.Tensor


@dataclass(frozen=True)
class GeometryTokenLayout:
    depth_query_count: int
    local_uvd_points_per_hand: int
    coarse_uvd_points_per_hand: int
    hand_count: int = 1

    @property
    def local_uvd_token_count(self) -> int:
        return self.local_uvd_points_per_hand * self.hand_count
    @property
    def coarse_uvd_token_count(self) -> int:
        return self.coarse_uvd_points_per_hand * self.hand_count
    @property
    def geometry_token_count(self) -> int:
        return 2 * self.depth_query_count + self.coarse_uvd_token_count + self.local_uvd_token_count

    def sequence_slices(self, native_token_count: int) -> GeometrySequenceSlices:
        current = slice(native_token_count, native_token_count + self.depth_query_count)
        future = slice(current.stop, current.stop + self.depth_query_count)
        coarse = slice(future.stop, future.stop + self.coarse_uvd_token_count)
        local = slice(coarse.stop, coarse.stop + self.local_uvd_token_count)
        return GeometrySequenceSlices(
            native=slice(0, native_token_count),
            depth_current=current,
            depth_future=future,
            uvd_coarse=coarse,
            uvd_local=local,
        )
```

`GeometryTokenEmbedding.forward()` returns:

```text
cat(current_depth_queries,
    future_depth_queries,
    coarse_seed + coarse_time_embedding + hand_embedding,
    local_seed + local_time_embedding + hand_embedding)
```

The mask starts from standard causal order, makes each depth group fully connected, and allows same-time cross-hand reads independently within coarse and local groups.

- [ ] **Step 5: Implement exact named packing**

Define this private strict packer and the two public wrappers:

```python
def _pack_named_uvd_targets_time_major(
    examples: list[dict[str, Any]],
    *,
    points_per_hand: int,
    hand_count: int,
    value_key: str,
    valid_key: str,
    time_key: str,
    device: torch.device,
) -> PackedUVDTargets:
    batch_size = len(examples)
    token_count = points_per_hand * hand_count
    target = torch.empty(batch_size, token_count, 3, device=device, dtype=torch.float32)
    valid = torch.empty(batch_size, token_count, device=device, dtype=torch.bool)
    times = torch.empty(batch_size, token_count, device=device, dtype=torch.float32)
    hand_ids = torch.arange(hand_count, device=device).repeat(points_per_hand)
    for batch_index, example in enumerate(examples):
        values = np.asarray(example[value_key], dtype=np.float32)
        masks = np.asarray(example[valid_key], dtype=np.bool_)
        nominal_times = np.asarray(example[time_key], dtype=np.float32)
        expected_values = (points_per_hand, 3) if hand_count == 1 else (points_per_hand, hand_count, 3)
        expected_masks = (points_per_hand,) if hand_count == 1 else (points_per_hand, hand_count)
        if values.shape != expected_values or masks.shape != expected_masks:
            raise ValueError(
                f"{value_key} must contain exactly {points_per_hand} time points and {hand_count} hands"
            )
        if nominal_times.shape != (points_per_hand,):
            raise ValueError(f"{time_key} must have shape ({points_per_hand},)")
        target[batch_index] = torch.as_tensor(values.reshape(token_count, 3), device=device)
        valid[batch_index] = torch.as_tensor(masks.reshape(token_count), device=device)
        times[batch_index] = torch.as_tensor(
            np.repeat(nominal_times, hand_count), device=device
        )
    return PackedUVDTargets(
        target=target,
        valid=valid,
        times=times,
        hand_ids=hand_ids.unsqueeze(0).expand(batch_size, -1),
    )
```

Then expose:

```python
def pack_local_uvd_targets_time_major(
    examples: list[dict[str, Any]], layout: GeometryTokenLayout, *, device: torch.device
) -> PackedUVDTargets:
    return _pack_named_uvd_targets_time_major(
        examples,
        points_per_hand=layout.local_uvd_points_per_hand,
        hand_count=layout.hand_count,
        value_key="uvd",
        valid_key="uvd_valid_mask",
        time_key="uvd_time",
        device=device,
    )

def pack_coarse_uvd_targets_time_major(
    examples: list[dict[str, Any]], layout: GeometryTokenLayout, *, device: torch.device
) -> PackedUVDTargets:
    return _pack_named_uvd_targets_time_major(
        examples,
        points_per_hand=layout.coarse_uvd_points_per_hand,
        hand_count=layout.hand_count,
        value_key="uvd_coarse",
        valid_key="uvd_coarse_valid_mask",
        time_key="uvd_coarse_time",
        device=device,
    )
```

The wrappers consume `uvd`/`uvd_valid_mask`/`uvd_time` and
`uvd_coarse`/`uvd_coarse_valid_mask`/`uvd_coarse_time`, respectively. They flatten only after validating `[H, hands, 3]` or `[H, 3]` and repeat each nominal time across hands.

- [ ] **Step 6: Run V4 geometry plus Qwen geometry-forward tests**

```bash
pytest -q tests/test_geometric_cot_v4.py tests/test_qwen35_geometry_forward.py
```

Expected: PASS.

- [ ] **Step 7: Commit the V4 token module**

```bash
git add -f starVLA/model/modules/geometric_cot_v4.py tests/test_geometric_cot_v4.py
git commit -m "feat: add coarse-local V4 geometry tokens"
```

---

### Task 5: Implement the Standalone QwenGR00TCoTV4 Framework

**Files:**
- Create: `starVLA/model/framework/VLM4A/QwenGR00TCoTV4.py`
- Create: `tests/test_qwen_gr00t_cot_v4.py`

**Interfaces:**
- Consumes: Task 4 geometry API, `Qwen_GR00T`, `SharedFiLMConvStack`, CoT loss functions, and `forward_qwen35_with_geometry`.
- Produces: registered framework `QwenGR00TCoTV4`, local output `uvd`, coarse output `uvd_coarse`, and the five weighted objectives.

- [ ] **Step 1: Write failing framework structure and validation tests**

Use an uninitialized model helper, following existing V2 test style, and assert:

```python
def test_v4_is_registered_and_does_not_subclass_v2():
    assert FRAMEWORK_REGISTRY["QwenGR00TCoTV4"] is Qwen_GR00T_CoT_V4
    assert issubclass(Qwen_GR00T_CoT_V4, Qwen_GR00T)
    assert Qwen_GR00T_CoT_V2 not in Qwen_GR00T_CoT_V4.__mro__


def test_v4_action_condition_is_native_depth_coarse_local():
    split = GeometryHiddenSplit(
        native=torch.tensor([[[0.0]]]),
        depth_current=torch.tensor([[[1.0]]]),
        depth_future=torch.tensor([[[2.0]]]),
        uvd_coarse=torch.tensor([[[3.0]], [[4.0]]]).transpose(0, 1),
        uvd_local=torch.tensor([[[5.0]], [[6.0]]]).transpose(0, 1),
    )
    condition, mask = model._build_action_condition(split, native_attention_mask=torch.ones(1, 1, dtype=torch.bool))
    assert condition.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert mask.all()
```

Add config cases that reject model/data horizon mismatch, stride other than 2, false terminal-repeat, and V2 reverse-full checkpoint keys.

- [ ] **Step 2: Write failing loss/head tests**

Assert `local_uvd_head is not coarse_uvd_head`, both heads receive gradients, and:

```python
expected = (
    action
    + 0.14 * depth_current
    + 0.15 * depth_future
    + 0.62 * (local_abs + 0.1 * local_relative)
    + 0.20 * (coarse_abs + 0.1 * coarse_relative)
)
torch.testing.assert_close(output["total_loss"], expected)
assert set(output) >= {
    "uvd_loss", "uvd_absolute_loss", "uvd_relative_loss",
    "uvd_coarse_loss", "uvd_coarse_absolute_loss", "uvd_coarse_relative_loss",
}
```

- [ ] **Step 3: Run framework tests and verify import failure**

```bash
pytest -q tests/test_qwen_gr00t_cot_v4.py
```

Expected: collection ERROR because `QwenGR00TCoTV4` does not exist.

- [ ] **Step 4: Implement V4 initialization and causal backbone**

Create a standalone `Qwen_GR00T_CoT_V4(Qwen_GR00T)` registered under `QwenGR00TCoTV4`. Its constructor must build:

```python
self.geometry_layout = GeometryTokenLayout(
    depth_query_count=geometry["depth_query_count"],
    local_uvd_points_per_hand=geometry["local_uvd_num_points"],
    coarse_uvd_points_per_hand=geometry["coarse_uvd_num_points"],
    hand_count=geometry["uvd_hand_count"],
)
self.geometry_tokens = GeometryTokenEmbedding(hidden_dim, self.geometry_layout)
self.depth_attention_pool = SharedDepthAttentionPool(hidden_dim)
self.depth_decoder = SharedFiLMConvStack(
    hidden_dim=hidden_dim,
    features=int(geometry.get("depth_decoder_features", 256)),
    stage_count=int(geometry.get("depth_decoder_stages", 3)),
)
self.local_uvd_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
self.coarse_uvd_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
```

Validate the complete model/data contract during construction. Keep Qwen native-input building, main-image extraction, depth pooling/decoding, action loss, and action inference behavior identical to clean V2; do not call or subclass the V2 framework.

- [ ] **Step 5: Implement split, prediction, losses, and outputs**

Define:

```python
@dataclass(frozen=True)
class GeometryHiddenSplit:
    native: torch.Tensor
    depth_current: torch.Tensor
    depth_future: torch.Tensor
    uvd_coarse: torch.Tensor
    uvd_local: torch.Tensor
```

Use separate `_predict_local_uvd` and `_predict_coarse_uvd` methods with sigmoid `u/v` and softplus `d`. Training packs both targets before the backbone, performs one causal Qwen pass, computes both mean-normalized UVD objectives, and returns all keys required in Step 2. `predict_geometry()` returns depth, `uvd`, and `uvd_coarse` without requiring ground-truth geometry fields.

Set `uvd_hand_count`, `uvd_token_order = "time_major"`, and the `geometry_query` compatibility
property. `predict_geometry_diagnostics()` must additionally return `uvd_tokens`,
`uvd_coarse_tokens`, depth tokens, and depth pooling weights so Task 6 can compute independent
utilization metrics.

- [ ] **Step 6: Implement factorized action interventions**

Support whole-geometry and component interventions for:

```text
depth_current, depth_future, depth,
uvd_coarse, uvd_local, uvd (coarse+local),
within_task_shuffle, cross_task_swap,
zero_geometry, native_only, coarse_only, local_only, coarse+local
```

Every fixed-length counterfactual keeps the original condition shape and mask. Whole-bundle shuffles use one donor permutation for depth/coarse/local; component shuffles move only the named group. Reject identity permutations, impossible task groupings, nonfinite conditions, and direct-depth-only variants when depth is absent from the normal action condition.

- [ ] **Step 7: Run V4 framework and V2 isolation tests**

```bash
pytest -q \
  tests/test_qwen_gr00t_cot_v4.py \
  tests/test_geometric_cot_v4.py \
  tests/test_qwen_gr00t_cot_v2.py \
  tests/test_qwen_gr00t_cot_v3.py
```

Expected: PASS.

- [ ] **Step 8: Commit the standalone V4 framework**

```bash
git add -f \
  starVLA/model/framework/VLM4A/QwenGR00TCoTV4.py \
  tests/test_qwen_gr00t_cot_v4.py
git commit -m "feat: add QwenGR00T coarse-local V4"
```

---

### Task 6: Add V4 Trainer Metrics and Diagnostics Without Reintroducing V2 Branches

**Files:**
- Modify: `starVLA/training/train_starvla_cot_v1.py`
- Modify: `starVLA/training/cot_test_diagnostics.py`
- Create: `starVLA/training/cot_v4_diagnostics.py`
- Create: `starVLA/training/train_starvla_cot_v4.py`
- Modify: `tests/test_cot_trainer_objective.py`
- Modify: `tests/test_cot_diagnostics.py`
- Create: `tests/test_cot_v4_trainer.py`
- Create: `tests/test_cot_v4_diagnostics.py`

**Interfaces:**
- Consumes: V4 forward outputs, `predict_geometry_diagnostics`, existing depth/local metrics, and `CotV1Trainer` lifecycle.
- Produces: `CotV4Trainer`, coarse loss metrics, named coarse errors, overlap gap, repeat ratio, and separate head gradient/token metrics.

- [ ] **Step 1: Write failing trainer hook and weighted-balance tests**

Add a no-op extension contract to the shared trainer tests and V4-specific expected metrics:

```python
def test_v4_trainer_adds_coarse_loss_to_weighted_aux_balance():
    metrics = trainer._train_step([])
    assert metrics["uvd_coarse_loss"] == pytest.approx(0.425)
    assert metrics["weighted_uvd_coarse_loss"] == pytest.approx(0.2 * 0.425)
    assert metrics["weighted_aux_loss"] == pytest.approx(
        0.14 * 0.5 + 0.15 * 0.25 + 0.62 * 0.15 + 0.2 * 0.425
    )
```

The shared `CotV1Trainer` test must still show no coarse/full keys for V1/V2 outputs.

- [ ] **Step 2: Refactor the shared trainer to expose no-op hooks**

Add exact extension methods:

```python
class CotV1Trainer(VLATrainer):
    def _extra_objective_metrics(
        self, output_dict: dict[str, torch.Tensor]
    ) -> tuple[dict[str, float], float]:
        return {}, 0.0

    def _extra_geometry_metrics(
        self,
        *,
        model,
        predictions: dict,
        examples: list[dict],
        diagnostic_config,
    ) -> dict[str, float]:
        return {}
```

Call `_extra_objective_metrics` once after base local metrics and add its returned scalar to `weighted_aux_loss`. Call `_extra_geometry_metrics` once after base geometry/token metrics. These hooks contain no V4 names and preserve V1/V2 behavior.

- [ ] **Step 3: Implement `CotV4Trainer` coarse metric aggregation**

`CotV4Trainer._extra_objective_metrics()` returns:

```text
uvd_coarse_loss
weighted_uvd_coarse_loss
uvd_coarse_absolute_loss
weighted_uvd_coarse_absolute_loss
uvd_coarse_relative_loss
weighted_uvd_coarse_relative_loss
```

The returned weighted-extra scalar is exactly `lambda_uvd_coarse * uvd_coarse_loss`.
The V4 CLI follows the V2 entry point lifecycle but logs `QwenGR00TCoT V4 training` and builds `CotV4Trainer`.

- [ ] **Step 4: Write failing named-UVD, overlap, repeat, and module tests**

Use a two-hand synthetic batch whose valid target points are all `[0.2, 0.2, 1.0]`, whose coarse
predictions are `[0.3, 0.3, 1.0]`, and whose local predictions are `[0.25, 0.3, 1.0]`. Give both
groups all-true validity masks and repeated frame indices such as `[9, 9, 9, 9]`. With
`image_size=101`, assert:

```python
metrics = compute_v4_geometry_metrics(predictions, examples, depth_scale=1.0, image_size=101, hand_count=2)
assert metrics["uvd_coarse_xy_mae_pixel"] == pytest.approx(10.0)
assert metrics["uvd_coarse_depth_mae_m"] == pytest.approx(0.0)
assert metrics["cross_scale/overlap_xy_gap_pixel"] == pytest.approx(5.0)
assert metrics["data/terminal_repeat_sample_ratio"] == 1.0
```

Extend module-discovery tests so V4 reports nonzero `grad/local_uvd_head_norm` and
`grad/coarse_uvd_head_norm`, while V2 continues to report `grad/uvd_head_norm`.

- [ ] **Step 5: Implement V4 diagnostics**

Expose from `cot_v4_diagnostics.py`:

```python
def compute_v4_geometry_metrics(
    predictions: dict[str, Any],
    examples: list[dict],
    *,
    depth_scale: float,
    image_size: int,
    hand_count: int,
) -> dict[str, float]:
    metrics = compute_named_uvd_metrics(
        predictions["uvd_coarse"],
        examples,
        value_key="uvd_coarse",
        valid_key="uvd_coarse_valid_mask",
        prefix="uvd_coarse",
        depth_scale=depth_scale,
        image_size=image_size,
        hand_count=hand_count,
        order="time_major",
    )
    metrics.update(
        compute_cross_scale_overlap_metrics(
            predictions["uvd_coarse"],
            predictions["uvd"],
            examples,
            image_size=image_size,
            hand_count=hand_count,
        )
    )
    metrics["data/terminal_repeat_sample_ratio"] = terminal_repeat_sample_ratio(examples)
    return metrics

def terminal_repeat_sample_ratio(examples: list[dict]) -> float:
    repeated = []
    for example in examples:
        local = np.asarray(example["uvd_frame_indices"], dtype=np.int64)
        coarse = np.asarray(example["uvd_coarse_frame_indices"], dtype=np.int64)
        repeated.append(bool(np.any(local[1:] == local[:-1]) or np.any(coarse[1:] == coarse[:-1])))
    return float(np.mean(repeated)) if repeated else 0.0
```

Extract the existing UVD coordinate, adjacent-motion, path, endpoint, and per-time block from
`compute_geometry_metrics` into the version-neutral function
`compute_named_uvd_metrics(prediction: torch.Tensor, examples: list[dict], *, value_key: str,
valid_key: str, prefix: str, depth_scale: float, image_size: int, hand_count: int,
order: str) -> dict[str, float]`. Make the existing local call use `value_key="uvd"`,
`valid_key="uvd_valid_mask"`, and `prefix="uvd"`; this regression-proves the extraction preserves
all current local metric names and values.

Implement the V4-only overlap helper as:

```python
def compute_cross_scale_overlap_metrics(
    coarse_prediction: torch.Tensor,
    local_prediction: torch.Tensor,
    examples: list[dict],
    *,
    image_size: int,
    hand_count: int,
) -> dict[str, float]:
    batch_size = coarse_prediction.shape[0]
    coarse = coarse_prediction.reshape(batch_size, -1, hand_count, 3)
    local = local_prediction.reshape(batch_size, -1, hand_count, 3)
    overlap_count = min(coarse.shape[1], local.shape[1] // 2)
    local_indices = torch.arange(overlap_count, device=local.device) * 2 + 1
    coarse = coarse[:, :overlap_count]
    local = local.index_select(1, local_indices)
    coarse_valid = torch.as_tensor(
        np.stack([example["uvd_coarse_valid_mask"] for example in examples]),
        device=coarse.device,
        dtype=torch.bool,
    ).reshape(batch_size, -1, hand_count)[:, :overlap_count]
    local_valid = torch.as_tensor(
        np.stack([example["uvd_valid_mask"] for example in examples]),
        device=local.device,
        dtype=torch.bool,
    ).reshape(batch_size, -1, hand_count).index_select(1, local_indices)
    valid = coarse_valid & local_valid
    if not valid.any():
        return {"cross_scale/overlap_valid_count": 0.0}
    xy_gap = torch.linalg.vector_norm(coarse[..., :2] - local[..., :2], dim=-1)
    return {
        "cross_scale/overlap_valid_count": float(valid.sum().item()),
        "cross_scale/overlap_xy_gap_pixel": float(
            (xy_gap[valid].mean() * (image_size - 1)).item()
        ),
    }
```

The overlap mapping is `coarse time j -> local time 2*j+1` for zero-based indices while that local index is `< H`. Compute gap only where both prediction validity masks are true. Reuse a version-neutral named-UVD metric helper extracted from `cot_test_diagnostics.py`; prefix local as `uvd` and coarse as `uvd_coarse`.

When predictions include `uvd_coarse_tokens`, add token utilization with prefix `uvd_coarse`; local remains `uvd`. Extend `collect_batch_valid_ratios` by iterating optional coarse validity/out-of-frame/clamp keys. Extend diagnostic module discovery by object presence, not framework-name checks.

- [ ] **Step 6: Run trainer and diagnostics tests**

```bash
pytest -q \
  tests/test_cot_trainer_objective.py \
  tests/test_cot_diagnostics.py \
  tests/test_cot_v4_trainer.py \
  tests/test_cot_v4_diagnostics.py
```

Expected: PASS, with no `uvd_full` key in trainer source.

- [ ] **Step 7: Commit trainer and diagnostics support**

```bash
git add -u \
  starVLA/training/train_starvla_cot_v1.py \
  starVLA/training/cot_test_diagnostics.py \
  tests/test_cot_trainer_objective.py \
  tests/test_cot_diagnostics.py
git add -f \
  starVLA/training/cot_v4_diagnostics.py \
  starVLA/training/train_starvla_cot_v4.py \
  tests/test_cot_v4_trainer.py \
  tests/test_cot_v4_diagnostics.py
git commit -m "feat: add V4 training diagnostics"
```

---

### Task 7: Add LIBERO/RoboCasa Configs, Launchers, and Timing Gate

**Files:**
- Create: `examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v4_q0_depthcond_coarse8_local8.yaml`
- Create: `examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v4_q0_depthcond_coarse16_local16.yaml`
- Create: `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v4_common.sh`
- Create: `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_libero_CoT_v4.sh`
- Create: `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_robocasa_fourier_CoT_v4.sh`
- Create: `examples/modelExtensions/CoT/scripts/compare_cot_v4_timing.py`
- Create: `tests/test_cot_v4_entrypoints.py`
- Create: `tests/test_cot_v4_timing_gate.py`

**Interfaces:**
- Consumes: Task 2/3 dataset module names, `QwenGR00TCoTV4`, `train_starvla_cot_v4.py`, and JSONL timing/memory keys from `VLATrainer`.
- Produces: two reproducible q0-depth experiments, dry-run launch commands, and a pass/fail timing report.

- [ ] **Step 1: Write failing YAML contract tests**

```python
@pytest.mark.parametrize(
    "filename,horizon,hands,dataset_py,max_steps",
    [
        ("qwen35_gr00t_libero_CoT_v4_q0_depthcond_coarse8_local8.yaml", 8, 1, "cot_v4_lerobot_datasets", 60000),
        ("qwen35_gr00t_robocasa_fourier_CoT_v4_q0_depthcond_coarse16_local16.yaml", 16, 2, "robocasa_v4_lerobot_datasets", 100000),
    ],
)
def test_v4_yaml_contract(filename, horizon, hands, dataset_py, max_steps):
    cfg = OmegaConf.load(CONFIG_ROOT / filename)
    assert cfg.framework.name == "QwenGR00TCoTV4"
    assert cfg.framework.action_model.action_horizon == horizon
    assert cfg.framework.action_model.num_target_vision_tokens == 0
    assert cfg.framework.geometry.include_depth_in_action_condition is True
    assert cfg.framework.geometry.local_uvd_num_points == horizon
    assert cfg.framework.geometry.coarse_uvd_num_points == horizon
    assert cfg.framework.geometry.coarse_uvd_stride == 2
    assert cfg.framework.geometry.uvd_hand_count == hands
    assert cfg.datasets.vla_data.dataset_py == dataset_py
    assert cfg.datasets.vla_data.cot_geometry.terminal_repeat is True
    assert cfg.trainer.max_train_steps == max_steps
```

Also compare each resolved YAML to its V2 q0-depth source after removing only the expected run/framework/dataset/geometry changes.

- [ ] **Step 2: Write failing launcher dry-run tests**

Run both exact cases with `DRY_RUN=1 NUM_PROCESSES=8`:

```python
cases = {
    "run_qwen35_gr00t_libero_CoT_v4.sh": (
        "qwen35_gr00t_libero_CoT_v4_q0_depthcond_coarse8_local8.yaml",
        "qwen35_gr00t_libero_CoT_v4_q0_depthcond_coarse8_local8_8gpu_bs16",
    ),
    "run_qwen35_gr00t_robocasa_fourier_CoT_v4.sh": (
        "qwen35_gr00t_robocasa_fourier_CoT_v4_q0_depthcond_coarse16_local16.yaml",
        "qwen35_gr00t_robocasa_fourier_CoT_v4_q0_depthcond_coarse16_local16_8gpu_bs16",
    ),
}
for script_name, (yaml_name, run_id) in cases.items():
    result = subprocess.run(
        ["bash", str(SCRIPT_ROOT / script_name)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", "NUM_PROCESSES": "8"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "starVLA/training/train_starvla_cot_v4.py" in result.stdout
    assert yaml_name in result.stdout
    assert run_id in result.stdout
    assert "--num_processes 8" in result.stdout
```

- [ ] **Step 3: Create the two YAMLs and V4 launch scripts**

Clone all non-geometry values from the corresponding V2 q0-depth YAML. Use these geometry values:

```yaml
geometry:
  depth_query_count: 8
  include_depth_in_action_condition: true
  local_uvd_num_points: ${framework.action_model.action_horizon}
  coarse_uvd_num_points: ${framework.action_model.action_horizon}
  coarse_uvd_stride: 2
  uvd_hand_count: 1  # LIBERO; 2 for RoboCasa
  lambda_action: 1.0
  lambda_depth_current: 0.14
  lambda_depth_future: 0.15
  lambda_uvd: 0.62
  lambda_uvd_relative: 0.1
  lambda_uvd_coarse: 0.2
  lambda_uvd_coarse_relative: 0.1
```

Repeat the resolved integer horizon values in `datasets.vla_data.cot_geometry`, with `terminal_repeat: true`. Keep LIBERO at 60k and RoboCasa at 100k.

The common launcher defaults to eight processes, uses a distinct default port, calls the V4 trainer, copies the YAML into the output directory, and appends `"$@"` as dotlist overrides. The benchmark wrappers only set config, run ID, output root, and exec the common launcher.

- [ ] **Step 4: Write failing timing-gate tests**

Create temporary baseline/candidate JSONL files and assert the CLI:

```text
- ignores records before --warmup-step;
- uses median timing/model;
- reports maximum system/gpu_memory_max_allocated_gb;
- exits 0 at candidate/baseline <= 1.5;
- exits 1 above 1.5 or on nonfinite total_loss;
- writes a JSON report containing baseline_median_model_s, candidate_median_model_s,
  model_time_ratio, candidate_max_allocated_gb, and passed.
```

- [ ] **Step 5: Implement the timing CLI**

Use this interface:

```python
def summarize_metrics(path: Path, *, warmup_step: int) -> dict[str, float | bool]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    stable = [row for row in records if int(row.get("step", -1)) >= warmup_step]
    model_times = [float(row["timing/model"]) for row in stable if "timing/model" in row]
    losses = [float(row["total_loss"]) for row in stable if "total_loss" in row]
    memory = [
        float(row["system/gpu_memory_max_allocated_gb"])
        for row in stable
        if "system/gpu_memory_max_allocated_gb" in row
    ]
    if not model_times or not losses or not memory:
        raise ValueError(f"stable metrics are incomplete in {path}")
    if statistics.median(model_times) <= 0.0:
        raise ValueError(f"median model time must be positive in {path}")
    return {
        "median_model_s": statistics.median(model_times),
        "max_allocated_gb": max(memory),
        "losses_finite": all(math.isfinite(value) for value in losses),
    }

def compare_runs(
    baseline: Path,
    candidate: Path,
    *,
    warmup_step: int,
    max_model_ratio: float,
) -> dict[str, float | bool]:
    base = summarize_metrics(baseline, warmup_step=warmup_step)
    cand = summarize_metrics(candidate, warmup_step=warmup_step)
    ratio = cand["median_model_s"] / base["median_model_s"]
    return {
        "baseline_median_model_s": base["median_model_s"],
        "candidate_median_model_s": cand["median_model_s"],
        "model_time_ratio": ratio,
        "candidate_max_allocated_gb": cand["max_allocated_gb"],
        "passed": bool(
            ratio <= max_model_ratio
            and base["losses_finite"]
            and cand["losses_finite"]
        ),
    }
```

CLI arguments are `--baseline`, `--candidate`, `--warmup-step` (default 20),
`--max-model-ratio` (default 1.5), and `--output`. Reject empty stable windows and missing required metrics with a clear nonzero exit.

- [ ] **Step 6: Run YAML, launcher, and timing tests**

```bash
pytest -q tests/test_cot_v4_entrypoints.py tests/test_cot_v4_timing_gate.py
```

Expected: PASS.

- [ ] **Step 7: Commit reproducible experiment entrypoints**

```bash
git add -f \
  examples/modelExtensions/CoT/configs/qwen35_gr00t_libero_CoT_v4_q0_depthcond_coarse8_local8.yaml \
  examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v4_q0_depthcond_coarse16_local16.yaml \
  examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v4_common.sh \
  examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_libero_CoT_v4.sh \
  examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_robocasa_fourier_CoT_v4.sh \
  examples/modelExtensions/CoT/scripts/compare_cot_v4_timing.py \
  tests/test_cot_v4_entrypoints.py \
  tests/test_cot_v4_timing_gate.py
git commit -m "exp: add LIBERO and RoboCasa V4 runs"
```

---

### Task 8: Verify Real Data, Run Timing Gates, and Start the Authorized LIBERO Training

**Files:**
- Verify only: all files changed in Tasks 1-7
- Runtime output: `/root/data/yxz/outputs/qwen35_gr00t_*_CoT_v4_*_smoke`
- Runtime output: `/root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v4_q0_depthcond_coarse8_local8_8gpu_bs16`

**Interfaces:**
- Consumes: both V4 wrappers, existing V2 q0-depth metrics, free GPU inventory, and tmux.
- Produces: complete test evidence, real-batch evidence, two smoke/timing reports, and one live LIBERO formal session.

- [ ] **Step 1: Run syntax and targeted regression checks**

```bash
python -m compileall -q \
  starVLA/dataloader/gr00t_lerobot/cot_geometry_v4.py \
  starVLA/dataloader/cot_v4_lerobot_datasets.py \
  starVLA/dataloader/robocasa_v4_lerobot_datasets.py \
  starVLA/model/modules/geometric_cot_v4.py \
  starVLA/model/framework/VLM4A/QwenGR00TCoTV4.py \
  starVLA/training/cot_v4_diagnostics.py \
  starVLA/training/train_starvla_cot_v4.py

pytest -q \
  tests/test_cot_geometry.py \
  tests/test_geometric_cot_v2.py \
  tests/test_qwen_gr00t_cot_v2.py \
  tests/test_geometric_cot_v3.py \
  tests/test_qwen_gr00t_cot_v3.py \
  tests/test_cot_v4_geometry.py \
  tests/test_cot_v4_lerobot_dataset.py \
  tests/test_robocasa_v4_lerobot_dataset.py \
  tests/test_geometric_cot_v4.py \
  tests/test_qwen_gr00t_cot_v4.py \
  tests/test_cot_v4_trainer.py \
  tests/test_cot_v4_diagnostics.py \
  tests/test_cot_v4_entrypoints.py \
  tests/test_cot_v4_timing_gate.py
```

Expected: PASS.

- [ ] **Step 2: Run the full repository test suite**

```bash
pytest -q
```

Expected: PASS. If unrelated environment-only tests skip, record their exact skip reasons; do not report them as passes.

- [ ] **Step 3: Confirm a clean repository and no active reverse-full path**

```bash
git diff --check
git status --short
rg -n "uvd_full|full_uvd|sample_reverse_uvd" \
  starVLA/dataloader starVLA/model starVLA/training \
  examples/modelExtensions/CoT/configs tests
```

Expected: clean status and no reverse-full matches.

- [ ] **Step 4: Check GPU and tmux state without stopping unrelated work**

```bash
nvidia-smi
tmux list-sessions
pgrep -af "train_starvla|accelerate.commands.launch"
```

Select only an entirely free eight-GPU set. Do not terminate processes or reuse an occupied port.

- [ ] **Step 5: Run a 100-step LIBERO smoke in tmux**

```bash
tmux new-session -d -s cot_v4_libero_smoke \
  "cd /home/yxz/CoT/CoT_vla && \
   NUM_PROCESSES=8 MAIN_PROCESS_PORT=29524 \
   RUN_ID=qwen35_gr00t_libero_CoT_v4_coarse8_local8_smoke \
   bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_libero_CoT_v4.sh \
   --trainer.max_train_steps 100 \
   --trainer.num_warmup_steps 10 \
   --trainer.eval_interval 100000 \
   --trainer.save_interval 100000"
```

Inspect the log until step 100 completes. Verify finite `total_loss`, nonzero local/coarse losses and head gradients, expected target shapes, and stable allocated/reserved memory.

- [ ] **Step 6: Gate LIBERO timing against the existing q0-depth V2 run**

```bash
python examples/modelExtensions/CoT/scripts/compare_cot_v4_timing.py \
  --baseline /root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v2_q0_depthcond_8gpu_bs16/train_metrics.jsonl \
  --candidate /root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v4_coarse8_local8_smoke/train_metrics.jsonl \
  --warmup-step 20 \
  --max-model-ratio 1.5 \
  --output /root/data/yxz/outputs/qwen35_gr00t_libero_CoT_v4_coarse8_local8_smoke/timing_gate.json
```

Expected: exit 0 and `passed: true`. If the baseline JSONL is absent, run the existing V2 q0-depth wrapper for the same 100-step settings under a new `_smoke` run ID, then use that file; never compare to the old 128-token reverse-full run.

- [ ] **Step 7: Run and gate a 100-step RoboCasa smoke sequentially**

After the LIBERO smoke releases its workers, run:

```bash
tmux new-session -d -s cot_v4_robocasa_smoke \
  "cd /home/yxz/CoT/CoT_vla && \
   NUM_PROCESSES=8 MAIN_PROCESS_PORT=29525 \
   RUN_ID=qwen35_gr00t_robocasa_fourier_CoT_v4_coarse16_local16_smoke \
   bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_robocasa_fourier_CoT_v4.sh \
   --trainer.max_train_steps 100 \
   --trainer.num_warmup_steps 10 \
   --trainer.eval_interval 100000 \
   --trainer.save_interval 100000"
```

Compare it to `/root/data/yxz/outputs/qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond_8gpu_bs16/train_metrics.jsonl` with the same timing CLI and 1.5 threshold. If that baseline is absent, create a matched 100-step V2 smoke first.

- [ ] **Step 8: Start the authorized LIBERO 60k run only after both gates pass**

```bash
tmux new-session -d -s cot_v4_libero \
  "cd /home/yxz/CoT/CoT_vla && \
   NUM_PROCESSES=8 MAIN_PROCESS_PORT=29526 \
   bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_libero_CoT_v4.sh"
```

Poll until multiple post-warmup metrics records exist. Report the tmux session, output directory, log path, step, total/action/depth/local/coarse losses, `timing/model`, allocated/reserved/max memory, and timing ratio. Do not start the RoboCasa 100k formal run without a new explicit request.

- [ ] **Step 9: Record final evidence and commit any verification-only fixes**

If verification exposed a defect, fix it through a failing regression test, rerun the affected suite, and commit the fix with a scoped message. End with:

```bash
git status --short
git log --oneline -10
```

Expected: clean status; the implementation commits and any evidence-backed fix commits are visible.
