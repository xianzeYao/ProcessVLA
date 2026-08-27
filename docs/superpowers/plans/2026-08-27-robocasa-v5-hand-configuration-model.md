# RoboCasa CoT V5 Hand-Configuration Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a RoboCasa-only q0+depth V5 in which the existing 12 Qwen hand-time trace tokens decode 36 bilateral thumb/index/wrist UVD targets while action conditioning continues to consume only the 12 Qwen states.

**Architecture:** A V5 dataset adapter reads the validated sidecars and exposes structured `[time, hand, landmark, UVD]` labels using V2's six frame samples. A V5 geometry module reuses V2's token layout, embedding, and attention mask, but adds decoder-only landmark embeddings and a strict packer. A separately registered framework subclasses V2, swaps only the UVD decoder/targets/loss metadata, and retains the V2 depth decoder and action condition; V5-specific trainer/config/launch files keep all prior versions isolated.

**Tech Stack:** PyTorch, Qwen3.5-4B, StarVLA/GR00T DiT, OmegaConf, Accelerate/DeepSpeed, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-robocasa-cot-v5-lrw-hand-configuration-design.md`

## Global Constraints

- Begin only after the complete sidecar validation and ten-task visual audit from `2026-08-27-robocasa-v5-lrw-sidecars-and-visualization.md` pass.
- V2 q0+depth is the exact baseline: action horizon 16, depth queries 8+8, UVD time points 6, hand count 2, and 12 time-major Qwen UVD tokens.
- Landmark order is `[thumb, index, wrist]`; flattened output order within each time is `[L_left,R_left,W_left,L_right,R_right,W_right]`.
- The UVD decoder produces 36 physical points from 12 Qwen tokens using decoder-only landmark embeddings; it does not append 36 tokens to Qwen.
- The action condition uses only native, current-depth, future-depth, and the 12 Qwen hand-configuration states. It never consumes decoded LRW features or ground truth.
- Preserve V2 depth decoder sharing, attention behavior, action model, optimizer, losses, schedule, and data mixture except for the explicitly defined V5 UVD target/decoder.
- Set `lambda_uvd_shape=0.0` for the first experiment.
- Do not modify V2/V3/V4 registry names, YAML semantics, checkpoint behavior, or launchers.
- Preserve all unrelated dirty-worktree changes.

---

## File map

- Create `starVLA/dataloader/robocasa_v5_lerobot_datasets.py`: sidecar-backed V5 adapter and existing 24-task factory.
- Modify `starVLA/dataloader/__init__.py`: add one exact dispatch branch for `robocasa_v5_lerobot_datasets` while keeping old branches unchanged.
- Create `starVLA/model/modules/geometric_cot_v5.py`: structured packer, metadata builders, and decoder-only LRW expansion head; reuse V2 token layout/attention functions.
- Create `starVLA/model/framework/VLM4A/QwenGR00TCoTV5.py`: independent registry/framework, losses, decoder, inference metadata, and checkpoint validation.
- Modify `starVLA/training/train_starvla_cot_v1.py`: make the existing diagnostic track-count hook prefer explicit `uvd_track_count`, defaulting identically for V1-V4.
- Modify `starVLA/training/cot_test_diagnostics.py`: accept structured `[T,H,K,3]` examples only when an explicit landmark count is supplied; preserve existing rank-2/rank-3 behavior.
- Create `starVLA/training/train_starvla_cot_v5.py`: isolated trainer entrypoint.
- Create `examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v5_q0_depthcond.yaml`: exact V2 q0+depth copy with V5 fields.
- Create `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v5_common.sh`: isolated Accelerate launcher.
- Create `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_robocasa_fourier_CoT_v5.sh`: RoboCasa wrapper.
- Create tests `tests/test_robocasa_v5_lerobot_dataset.py`, `tests/test_geometric_cot_v5.py`, `tests/test_qwen_gr00t_cot_v5.py`, `tests/test_cot_v5_dataloader_routing.py`, `tests/test_cot_v5_diagnostics.py`, and `tests/test_cot_v5_entrypoints.py`.

### Task 1: Sidecar-backed RoboCasa V5 dataset adapter

**Files:**
- Create: `starVLA/dataloader/robocasa_v5_lerobot_datasets.py`
- Test: `tests/test_robocasa_v5_lerobot_dataset.py`

**Interfaces:**
- Consumes: `hand_lrw_path`/`load_hand_lrw_sidecar`, `CoTLeRobotSingleDataset`, `RoboCasaGR1DataConfig`, and `dataset_specs`.
- Produces: `RoboCasaV5CoTLeRobotSingleDataset`, `get_vla_dataset(...)`, and `collate_fn(...)`.

- [ ] **Step 1: Write failing sidecar-loading and target-shape tests**

```python
def test_loader_keeps_time_hand_landmark_axes(tmp_path):
    dataset = uninitialized_v5_dataset(tmp_path, episode_id=7, base_index=1)
    depth, uvd, valid, state = dataset._load_episode_geometry(7)
    assert uvd.shape == (5, 2, 3, 3)
    assert valid.shape == (5, 2, 3)
    np.testing.assert_allclose(uvd[0, 0, 0], left_thumb_uvd)
    np.testing.assert_allclose(uvd[0, 1, 2], right_wrist_uvd)


def test_targets_use_v2_six_time_indices_and_explicit_ids():
    targets = dataset._geometry_targets()
    assert targets["uvd"].shape == (6, 2, 3, 3)
    assert targets["uvd_valid_mask"].shape == (6, 2, 3)
    np.testing.assert_array_equal(targets["uvd_hand_ids"][0], [[0,0,0],[1,1,1]])
    np.testing.assert_array_equal(targets["uvd_landmark_ids"][0], [[0,1,2],[0,1,2]])
```

Also assert sidecar frame mismatch/missing files fail, DIAL `agentview_in_frame` is used rather than projection validity, dataset order/weights match `fourier_gr1_unified_1000`, and V2's loader class remains unchanged.

- [ ] **Step 2: Run the focused test and confirm the adapter is missing**

Run: `PYTHONPATH=. pytest -q tests/test_robocasa_v5_lerobot_dataset.py`

Expected: collection failure for `robocasa_v5_lerobot_datasets`.

- [ ] **Step 3: Implement the minimal V5 adapter**

```python
class RoboCasaV5CoTLeRobotSingleDataset(CoTLeRobotSingleDataset):
    def _load_episode_geometry(self, trajectory_id):
        depth = _read_npz_array(depth_path, "depth_m")
        sidecar = load_hand_lrw_sidecar(
            hand_lrw_path(self.dataset_path, trajectory_id),
            frame_count=len(self.curr_traj_data),
            width=int(depth.shape[-1]),
            height=int(depth.shape[-2]),
        )
        return depth, sidecar.agentview_uvd_pixels, sidecar.agentview_in_frame, state

    def _geometry_targets(self):
        targets = super()._geometry_targets()
        time_count = int(targets["uvd"].shape[0])
        targets["uvd_hand_ids"] = np.broadcast_to(
            np.asarray([[[0,0,0],[1,1,1]]], dtype=np.int64), (time_count,2,3)
        ).copy()
        targets["uvd_landmark_ids"] = np.broadcast_to(
            np.asarray([[[0,1,2],[0,1,2]]], dtype=np.int64), (time_count,2,3)
        ).copy()
        return targets
```

Reuse `RoboCasaGR1DataConfig` and `dataset_specs`; do not duplicate task lists or transforms.

- [ ] **Step 4: Run dataset and V2/V4 regressions**

Run: `PYTHONPATH=. pytest -q tests/test_robocasa_v5_lerobot_dataset.py tests/test_robocasa_v4_lerobot_dataset.py`

Expected: pass.

- [ ] **Step 5: Commit the V5 dataset adapter**

```bash
git add starVLA/dataloader/robocasa_v5_lerobot_datasets.py tests/test_robocasa_v5_lerobot_dataset.py
git commit -m "feat: load RoboCasa LRW targets for CoT V5"
```

### Task 2: Twelve-token/36-point packer and LRW decoder

**Files:**
- Create: `starVLA/model/modules/geometric_cot_v5.py`
- Test: `tests/test_geometric_cot_v5.py`

**Interfaces:**
- Consumes: V2 `GeometryTokenLayout`, `GeometryTokenEmbedding`, `append_geometry_slots`, `build_geometry_full_attention_mask`, and `SharedDepthAttentionPool` unchanged.
- Produces: `HandConfigurationTokenLayout`, `PackedHandLRWTargets`, `HandLRWDecoder`, `build_time_major_hand_landmark_ids(layout)`, and `pack_hand_lrw_targets_time_major(examples, layout, device)`.

- [ ] **Step 1: Write failing layout, packing, decoder, and attention tests**

```python
def test_layout_keeps_12_qwen_tokens_but_declares_36_outputs():
    layout = HandConfigurationTokenLayout(
        depth_query_count=8, uvd_points_per_hand=6, hand_count=2, landmark_count=3
    )
    assert layout.uvd_token_count == 12
    assert layout.output_point_count == 36
    assert layout.geometry_token_count == 28


def test_decoder_expands_one_hand_token_to_lrw_without_qwen_tokens():
    decoder = HandLRWDecoder(hidden_dim=4, landmark_count=3)
    hidden = torch.zeros(1, 12, 4)
    raw = decoder(hidden)
    assert raw.shape == (1, 36, 3)
    assert decoder.landmark_embedding.weight.shape == (3, 4)
```

Packing must prove flattened order `[L_left,R_left,W_left,L_right,R_right,W_right]` for every time, metadata lengths 36, time repeated six times, and strict rejection of wrong hand/landmark axes or IDs. Re-run V2 attention tests to prove geometry length/mask remain identical.

- [ ] **Step 2: Run the focused test and confirm missing V5 symbols**

Run: `PYTHONPATH=. pytest -q tests/test_geometric_cot_v5.py`

Expected: import failure for `geometric_cot_v5`.

- [ ] **Step 3: Implement layout, packer, and decoder**

```python
@dataclass(frozen=True)
class HandConfigurationTokenLayout(GeometryTokenLayout):
    landmark_count: int = 3

    @property
    def output_point_count(self) -> int:
        return self.uvd_token_count * self.landmark_count


class HandLRWDecoder(nn.Module):
    def __init__(self, hidden_dim: int, landmark_count: int = 3):
        super().__init__()
        self.landmark_embedding = nn.Embedding(landmark_count, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))

    def forward(self, hand_tokens: torch.Tensor) -> torch.Tensor:
        expanded = hand_tokens[:, :, None, :] + self.landmark_embedding.weight[None, None]
        return self.mlp(expanded).reshape(hand_tokens.shape[0], -1, 3)
```

`pack_hand_lrw_targets_time_major` accepts only `[T,2,3,3]` plus `[T,2,3]` masks and canonical IDs; it pads missing tail times as invalid to fixed `[B,36,3]` without inventing labels.

- [ ] **Step 4: Run V5 and V2/V3 geometry regressions**

Run: `PYTHONPATH=. pytest -q tests/test_geometric_cot_v5.py tests/test_geometric_cot_v2.py tests/test_geometric_cot_v3.py`

Expected: pass.

- [ ] **Step 5: Commit the geometry module**

```bash
git add starVLA/model/modules/geometric_cot_v5.py tests/test_geometric_cot_v5.py
git commit -m "feat: decode bilateral LRW points from hand-time tokens"
```

### Task 3: Independent QwenGR00TCoTV5 framework

**Files:**
- Create: `starVLA/model/framework/VLM4A/QwenGR00TCoTV5.py`
- Test: `tests/test_qwen_gr00t_cot_v5.py`

**Interfaces:**
- Consumes: `Qwen_GR00T_CoT_V2`, Task 2 layout/decoder/packer, `uvd_regression_loss`, and `uvd_adjacent_relative_loss`.
- Produces: registered `Qwen_GR00T_CoT_V5`, flattened `[B,36,3]` predictions, and 36-entry inference metadata while retaining 12 hidden UVD states.

- [ ] **Step 1: Write failing framework contract tests**

```python
def test_v5_is_registered_and_keeps_action_condition_at_12_uvd_tokens():
    assert FRAMEWORK_REGISTRY["QwenGR00TCoTV5"] is Qwen_GR00T_CoT_V5
    split = GeometryHiddenSplit(
        native=torch.zeros(1, 5, 4),
        depth_current=torch.zeros(1, 8, 4),
        depth_future=torch.zeros(1, 8, 4),
        uvd=torch.zeros(1, 12, 4),
    )
    condition, _ = model._build_action_condition(split, native_attention_mask=torch.ones(1,5))
    assert condition.shape[1] == 5 + 8 + 8 + 12
    assert model._predict_uvd(split.uvd).shape == (1, 36, 3)
```

Also assert the full Qwen geometry length is native+28, decoder activations are sigmoid/softplus, temporal loss uses six independent `(hand,landmark)` streams, shape loss defaults to zero, geometry diagnostics expose 12 `uvd_tokens` but 36 predictions, inference metadata aligns times/hands/landmarks, and V2 checkpoints lacking `uvd_head.landmark_embedding` are rejected.

- [ ] **Step 2: Run the test and confirm V5 is missing**

Run: `PYTHONPATH=. pytest -q tests/test_qwen_gr00t_cot_v5.py`

Expected: import/registry failure.

- [ ] **Step 3: Implement the isolated framework**

In `__init__`, call V2 initialization, replace the V2 layout with `HandConfigurationTokenLayout` using the same counts, rebuild `GeometryTokenEmbedding` with that layout, replace `uvd_head` with `HandLRWDecoder`, set `landmark_count=3`, `uvd_track_count=6`, and keep `uvd_token_order="time_major"`.

Override only:

```python
def _prepare_uvd_targets(self, examples, device): ...
def _predict_uvd(self, tokens):
    raw = self.uvd_head(_cast_to_module_dtype(tokens, self.uvd_head))
    return torch.cat([torch.sigmoid(raw[..., :2]), F.softplus(raw[..., 2:3])], dim=-1)
def _compute_uvd_losses(self, pred, packed):
    absolute = uvd_regression_loss(pred, packed.target, packed.valid)
    temporal = uvd_adjacent_relative_loss(pred, packed.target, packed.valid, hand_count=6)
    return {"absolute": absolute, "temporal": temporal, "shape": zero, "total": absolute + 0.1 * temporal}
```

Implement `forward` and `predict_action(return_geometry=True)` with V3's single-backbone-pass pattern, but emit `uvd_time`, `uvd_hand_ids`, and `uvd_landmark_ids`, all length 36. Inherit V2 `_run_geometry_backbone`, attention, depth decode, action conditioning, diagnostics, and intervention logic.

- [ ] **Step 4: Run framework and all prior-version regressions**

Run: `PYTHONPATH=. pytest -q tests/test_qwen_gr00t_cot_v5.py tests/test_qwen_gr00t_cot_v2.py tests/test_qwen_gr00t_cot_v3.py tests/test_qwen_gr00t_cot_v4.py`

Expected: pass.

- [ ] **Step 5: Commit the framework**

```bash
git add starVLA/model/framework/VLM4A/QwenGR00TCoTV5.py tests/test_qwen_gr00t_cot_v5.py
git commit -m "feat: add RoboCasa hand-configuration CoT V5 framework"
```

### Task 4: V5-aware diagnostics without V2 behavior changes

**Files:**
- Modify: `starVLA/training/train_starvla_cot_v1.py`
- Modify: `starVLA/training/cot_test_diagnostics.py`
- Test: `tests/test_cot_v5_diagnostics.py`

**Interfaces:**
- Consumes: structured example UVD `[T,H,K,3]`, prediction UVD `[B,T*H*K,3]`, model `uvd_track_count=6`, and existing diagnostic functions.
- Produces: correct per-time/per-stream metrics, saved frame indices, and unchanged default behavior for V1-V4.

- [ ] **Step 1: Write failing structured-diagnostic tests**

```python
def test_metrics_flatten_hand_and_landmark_as_six_temporal_tracks():
    metrics = compute_geometry_metrics(
        predictions, examples, depth_scale=1.0, image_size=224,
        uvd_hand_count=6, uvd_order="time_major", include_uvd_time_metrics=True,
    )
    assert metrics["uvd/time_0/valid_count"] == 6.0
    assert metrics["uvd_adjacent_relative_smooth_l1"] == 0.0


def test_trainer_prefers_explicit_uvd_track_count():
    model = SimpleNamespace(uvd_track_count=6, landmark_count=3, uvd_hand_count=2)
    assert CotV1Trainer._uvd_track_count(model) == 6
```

Also validate prediction bundles expand each temporal frame index six times, reject rank-4 examples unless their combined track count equals the explicit count, and retain byte-for-byte V2 packing for rank-3 `[T,H,3]` inputs.

- [ ] **Step 2: Run diagnostic tests and confirm rank-4 failure**

Run: `PYTHONPATH=. pytest -q tests/test_cot_v5_diagnostics.py`

Expected: existing `_pad_uvd_examples` rejects `[T,2,3,3]`.

- [ ] **Step 3: Generalize diagnostics behind the explicit shape contract**

In `_pad_uvd_examples`, convert `[T,H,K,3]` to `[T,H*K,3]` and masks `[T,H,K]` to `[T,H*K]` only after validating `H*K == uvd_hand_count`. Keep old rank-2/rank-3 branches identical. Update the trainer hook to:

```python
return int(getattr(model, "uvd_track_count",
           getattr(model, "landmark_count", getattr(model, "uvd_hand_count", 1))))
```

- [ ] **Step 4: Run diagnostics regressions**

Run: `PYTHONPATH=. pytest -q tests/test_cot_v5_diagnostics.py tests/test_cot_diagnostics.py tests/test_cot_v3_losses.py tests/test_cot_v4_diagnostics.py`

Expected: pass.

- [ ] **Step 5: Commit gated diagnostic support**

```bash
git add starVLA/training/train_starvla_cot_v1.py starVLA/training/cot_test_diagnostics.py tests/test_cot_v5_diagnostics.py
git commit -m "feat: support structured LRW diagnostic targets"
```

### Task 5: Dataloader dispatch, trainer, YAML, and launchers

**Files:**
- Modify: `starVLA/dataloader/__init__.py`
- Create: `starVLA/training/train_starvla_cot_v5.py`
- Create: `examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v5_q0_depthcond.yaml`
- Create: `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v5_common.sh`
- Create: `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_robocasa_fourier_CoT_v5.sh`
- Test: `tests/test_cot_v5_dataloader_routing.py`
- Test: `tests/test_cot_v5_entrypoints.py`

**Interfaces:**
- Consumes: Tasks 1-4 and the V2 q0+depth YAML values.
- Produces: a runnable eight-GPU V5 experiment with independent names and override forwarding.

- [ ] **Step 1: Write failing routing/config/launcher tests**

```python
def test_v5_yaml_is_exact_robocasa_q0_depth_contract():
    cfg = OmegaConf.load(CONFIG)
    assert cfg.framework.name == "QwenGR00TCoTV5"
    assert cfg.framework.action_model.action_horizon == 16
    assert cfg.framework.action_model.num_target_vision_tokens == 0
    assert cfg.framework.geometry.depth_query_count == 8
    assert cfg.framework.geometry.uvd_num_points == 6
    assert cfg.framework.geometry.uvd_hand_count == 2
    assert cfg.framework.geometry.landmark_count == 3
    assert cfg.framework.geometry.lambda_uvd_shape == 0.0
    assert cfg.datasets.vla_data.dataset_py == "robocasa_v5_lerobot_datasets"
    assert cfg.trainer.max_train_steps == 100000
```

The launcher test must use `DRY_RUN=1`, confirm `train_starvla_cot_v5.py`, the V5 YAML, `NUM_PROCESSES`, port, run ID, and arbitrary dotlist overrides. The routing test injects a fake `robocasa_v5_lerobot_datasets` module and proves all old dataset selectors still resolve their original modules.

- [ ] **Step 2: Run entrypoint tests and confirm failure**

Run: `PYTHONPATH=. pytest -q tests/test_cot_v5_dataloader_routing.py tests/test_cot_v5_entrypoints.py`

Expected: V5 dispatch/config/files are missing.

- [ ] **Step 3: Add isolated entrypoints and exact YAML**

`CotV5Trainer` subclasses `CotV2Trainer`; its main function mirrors `train_starvla_cot_v3.py` but logs V5 and builds the registered V5 framework. Copy the V2 q0+depth YAML and change only:

```yaml
run_id: qwen35_gr00t_robocasa_fourier_CoT_v5_q0_depthcond_lrw_8gpu_bs16
framework:
  name: QwenGR00TCoTV5
  geometry:
    landmark_count: 3
    lambda_uvd_temporal: 0.1
    lambda_uvd_shape: 0.0
datasets:
  vla_data:
    dataset_py: robocasa_v5_lerobot_datasets
```

Remove `lambda_uvd_relative` in favor of `lambda_uvd_temporal`; retain every other V2 numeric setting. The common launcher invokes only `train_starvla_cot_v5.py`.

- [ ] **Step 4: Run routing, trainer-help, and launcher tests**

Run: `PYTHONPATH=. pytest -q tests/test_cot_v5_dataloader_routing.py tests/test_cot_v5_entrypoints.py tests/test_cot_v2_entrypoints.py tests/test_cot_v3_entrypoints.py tests/test_cot_v4_entrypoints.py`

Expected: pass.

- [ ] **Step 5: Commit V5 entrypoints**

```bash
git add starVLA/dataloader/__init__.py starVLA/training/train_starvla_cot_v5.py \
  examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v5_q0_depthcond.yaml \
  examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v5_common.sh \
  examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_robocasa_fourier_CoT_v5.sh \
  tests/test_cot_v5_dataloader_routing.py tests/test_cot_v5_entrypoints.py
git commit -m "feat: add isolated RoboCasa CoT V5 training entrypoints"
```

### Task 6: Integrated verification and one-batch smoke test

**Files:**
- Modify only if a failing verification exposes a V5 defect; do not broaden scope.

**Interfaces:**
- Consumes: all prior tasks and completed sidecars.
- Produces: evidence that V5 loads real LRW targets, produces 36 geometry outputs from 12 tokens, computes finite losses, and leaves old versions passing.

- [ ] **Step 1: Run the complete focused CPU suite**

```bash
PYTHONPATH=. pytest -q \
  tests/test_robocasa_hand_lrw_sidecar.py \
  tests/test_robocasa_hand_lrw_generator.py \
  tests/test_robocasa_hand_lrw_generator_cli.py \
  tests/test_robocasa_hand_lrw_visualization.py \
  tests/test_robocasa_v5_lerobot_dataset.py \
  tests/test_geometric_cot_v5.py \
  tests/test_qwen_gr00t_cot_v5.py \
  tests/test_cot_v5_diagnostics.py \
  tests/test_cot_v5_dataloader_routing.py \
  tests/test_cot_v5_entrypoints.py
```

Expected: pass.

- [ ] **Step 2: Run all V2/V3/V4 CoT regressions**

```bash
PYTHONPATH=. pytest -q \
  tests/test_geometric_cot_v2.py tests/test_qwen_gr00t_cot_v2.py tests/test_cot_v2_entrypoints.py \
  tests/test_geometric_cot_v3.py tests/test_qwen_gr00t_cot_v3.py tests/test_cot_v3_entrypoints.py \
  tests/test_qwen_gr00t_cot_v4.py tests/test_cot_v4_entrypoints.py \
  tests/test_robocasa_v4_lerobot_dataset.py
```

Expected: pass.

- [ ] **Step 3: Run a real dataset sample smoke test**

Using the V5 YAML and completed sidecars, instantiate `get_vla_dataset`, fetch one sample from a canonical task, and assert:

```text
uvd                     [6,2,3,3]
uvd_valid_mask          [6,2,3]
uvd_hand_ids            [6,2,3]
uvd_landmark_ids        [6,2,3]
action                  [16,29]
```

Expected: all valid UVD depths are finite and positive; the old midpoint column is not read.

- [ ] **Step 4: Run a one-batch model forward smoke test**

Launch with `trainer.max_train_steps=1`, `datasets.vla_data.num_workers=0`, one GPU, and a fresh smoke run ID. Confirm finite `action_loss`, depth losses, UVD absolute/temporal loss, and total loss; diagnostics must report 12 `uvd_tokens` and 36 UVD predictions.

```bash
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 MAIN_PROCESS_PORT=29545 \
RUN_ID=qwen35_gr00t_robocasa_CoT_v5_lrw_smoke \
  bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_robocasa_fourier_CoT_v5.sh \
  --trainer.max_train_steps=1 --trainer.save_interval=1 \
  --datasets.vla_data.num_workers=0 --datasets.vla_data.per_device_batch_size=1
```

- [ ] **Step 5: Review diff and commit any verification-only fixes**

Run `git diff --check`, inspect `git status --short`, and commit only V5 files or explicitly gated backward-compatible diagnostic changes. Do not stage unrelated dirty files.
