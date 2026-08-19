# Trace Intervention Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, reproducible Stage A probe that measures how QwenGR00TCoTV2 geometry hidden-token interventions change each Euler-step velocity and the final action chunk.

**Architecture:** Refactor the existing legacy flow integrator around one velocity evaluator, then add a V2-only intervention builder and one-backbone diagnostic entrypoint. A standalone offline runner reuses paired-probe materialized samples and writes recomputable raw and summarized artifacts.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pytest, existing StarVLA framework and geometry-probe utilities.

**Spec:** `docs/superpowers/specs/2026-08-19-trace-intervention-probe-design.md`

## Global Constraints

- All diagnostic behavior is explicit opt-in; default training and inference remain unchanged.
- Do not start training or simulator rollout.
- Run the Qwen backbone once per diagnostic batch and reuse the same native tokens.
- Compare local velocities on the correct-path `x_k`.
- Treat Stage A as hidden-channel causal dependence, not numerical-trace execution faithfulness.
- Preserve all user changes and ignored preview media.
- Implement and verify on the current `CoT` branch as explicitly requested by the user.

---

### Task 1: Common-noise action integration and flow diagnostics

**Files:**
- Modify: `starVLA/model/modules/action_model/GR00T_ActionHeader.py`
- Modify: `tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py`

**Interfaces:**
- Produces: `FlowStepDiagnostics` frozen dataclass.
- Produces: `FlowmatchingActionHead.predict_velocity(actions, *, t_cont, vl_embs, state=None, encoder_attention_mask=None) -> torch.Tensor`.
- Extends: `FlowmatchingActionHead.predict_action(..., initial_actions=None, condition_schedule=None, return_diagnostics=False)`.

- [ ] **Step 1: Write failing tests for common initial actions and validation**

```python
def test_legacy_action_head_reuses_supplied_initial_actions_without_mutating_them():
    head = _tiny_diagnostic_action_head()
    initial = torch.full((2, 3, 2), 0.25)
    snapshot = initial.clone()
    first = head.predict_action(torch.ones(2, 2, 2), initial_actions=initial)
    second = head.predict_action(torch.ones(2, 2, 2), initial_actions=initial)
    assert torch.equal(first, second)
    assert torch.equal(initial, snapshot)

@pytest.mark.parametrize("bad_shape", [(1, 3, 2), (2, 2, 2), (2, 3, 1)])
def test_legacy_action_head_rejects_wrong_initial_action_shape(bad_shape):
    head = _tiny_diagnostic_action_head()
    with pytest.raises(ValueError, match="initial_actions shape"):
        head.predict_action(torch.ones(2, 2, 2), initial_actions=torch.zeros(bad_shape))
```

- [ ] **Step 2: Run the focused tests and verify missing-keyword failures**

Run: `pytest -q tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py -k 'initial_actions or flow_diagnostics or condition_schedule'`

Expected: FAIL because `predict_action()` does not accept `initial_actions`.

- [ ] **Step 3: Implement the shared velocity path and diagnostics**

Move the existing action embedding, positional embedding, state/future token concatenation, DiT call, decoder call, and action-horizon slice into `_predict_velocity_from_features()`. Have both `predict_velocity()` and `predict_action()` call it. Validate a supplied condition schedule has exactly `num_inference_timesteps` entries and compatible batch/mask shapes.

- [ ] **Step 4: Run focused and legacy action-head regression tests**

Run: `pytest -q tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py`

Expected: PASS.

### Task 2: Geometry intervention construction and permutation contracts

**Files:**
- Modify: `starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py`
- Modify: `tests/test_qwen_gr00t_cot_v2.py`

**Interfaces:**
- Produces: `GeometryActionCondition` frozen dataclass.
- Produces: `build_geometry_permutation(task_ids, *, mode, generator=None) -> torch.Tensor`.
- Produces: `Qwen_GR00T_CoT_V2._build_intervention_condition(split, *, native_attention_mask, name, permutation=None) -> GeometryActionCondition`.

- [ ] **Step 1: Write failing tests for matched zero/native/UVD/depth variants**

```python
def test_zero_geometry_preserves_correct_shape_and_mask():
    model = make_uninitialized_model(depth_queries=1, points=2, hands=1, include_depth=True)
    split = model._split_geometry_hidden(torch.arange(6.0).view(1, 6, 1), native_token_count=2)
    correct = model._build_intervention_condition(split, native_attention_mask=torch.ones(1, 2), name="correct")
    zero = model._build_intervention_condition(split, native_attention_mask=torch.ones(1, 2), name="zero_geometry")
    assert zero.condition.shape == correct.condition.shape
    assert torch.equal(zero.condition_mask, correct.condition_mask)
    assert torch.count_nonzero(zero.condition[:, 2:]) == 0
```

- [ ] **Step 2: Write failing permutation tests**

Assert within-task donors have equal task IDs, cross-task donors have unequal task IDs, no permutation is identity, batch size one is rejected, and one permutation moves all three geometry tensors together.

- [ ] **Step 3: Run focused V2 tests and verify missing-interface failures**

Run: `pytest -q tests/test_qwen_gr00t_cot_v2.py -k 'intervention or permutation or zero_geometry or native_only'`

- [ ] **Step 4: Implement variants and strict validation**

Use the existing `_build_action_condition()` for `correct`. For matched variants, construct a new `GeometryHiddenSplit` and call the same builder. Only `native_only` and Q0 `depth_only` may change trained sequence length; mark Q0 `depth_only` as `diagnostic_counterfactual=True`.

- [ ] **Step 5: Run the full V2 test file**

Run: `pytest -q tests/test_qwen_gr00t_cot_v2.py`

Expected: PASS.

### Task 3: One-backbone intervention execution

**Files:**
- Modify: `starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py`
- Modify: `tests/test_qwen_gr00t_cot_v2.py`

**Interfaces:**
- Produces: `Qwen_GR00T_CoT_V2.predict_action_interventions(examples, *, variants, task_ids, initial_actions=None, seed=42, rollout_steps=None) -> dict[str, Any]`, where `None` expands to `("all", *range(num_inference_timesteps))`.

- [ ] **Step 1: Write a failing one-backbone orchestration test**

Use a fake backbone that increments a call counter and a tiny real diagnostic action head. Assert one backbone call, equal initial noise for all variants, one exact correct repeat, local velocities evaluated at each correct `x_before`, and a single-step rollout whose schedule differs only at the selected step.

- [ ] **Step 2: Run the focused test and verify the method is absent**

Run: `pytest -q tests/test_qwen_gr00t_cot_v2.py -k 'predict_action_interventions'`

- [ ] **Step 3: Implement the diagnostic entrypoint**

Build all conditions from one `GeometryHiddenSplit`. Run `correct`, repeat it, compute alternative local velocities using `action_model.predict_velocity(correct_step.x_before, ...)`, then run `all` and each requested single-step schedule from the same cloned initial actions. Return tensors without converting to NumPy so the runner controls serialization.

- [ ] **Step 4: Run action-head and V2 tests together**

Run: `pytest -q tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py tests/test_qwen_gr00t_cot_v2.py`

Expected: PASS.

### Task 4: Offline metrics, batching, artifacts, and CLI

**Files:**
- Create: `examples/simBenchmarks/CoT/geometry_probe/trace_intervention_probe.py`
- Create: `examples/simBenchmarks/CoT/geometry_probe/run_trace_intervention_probe.py`
- Create: `tests/test_trace_intervention_probe.py`

**Interfaces:**
- Produces: `build_intervention_batches(sample_paths, *, batch_size) -> list[list[Path]]`.
- Produces: `effect_metrics(reference, alternative, action_groups) -> dict[str, Any]`.
- Produces: `cluster_bootstrap_mean_ci(values, cluster_ids, *, seed, resamples=2000) -> tuple[float, float]`.
- Produces: `run_trace_intervention_checkpoint(...) -> dict[str, Any]`.
- CLI accepts `--checkpoint`, `--samples-dir`, `--output-dir`, `--batch-size`, `--seed`, `--device`, `--variants`, `--bootstrap-resamples`, and `--repeat-tolerance`.

- [ ] **Step 1: Write failing pure CPU tests**

Test two-task/two-samples-per-task batching, reject impossible batches, hand-check L2 and per-dimension RMS on a `[B,T,D]` tensor, verify cluster bootstrap resamples whole episodes, and round-trip JSONL/NPZ artifact writing.

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `pytest -q tests/test_trace_intervention_probe.py`

- [ ] **Step 3: Implement batching, effects, bootstrap, and serializers**

Use materialized sample metadata (`suite`, `episode_id`, `frame_index`) as task and cluster identifiers. Preserve every raw tensor needed to recompute summary metrics. Do not denormalize actions in Stage A.

- [ ] **Step 4: Implement sequential checkpoint runner and CLI**

Load one framework with `baseframework.from_pretrained()`, move it to `device`, call `.eval()`, create balanced intervention batches, call `predict_action_interventions()`, and write `config.json`, `per_sample.jsonl`, `trajectories.npz`, and `summary.json`. Reject non-V2 checkpoints that lack the diagnostic method.

- [ ] **Step 5: Run focused probe tests**

Run: `pytest -q tests/test_trace_intervention_probe.py`

Expected: PASS.

### Task 5: Verification and handoff

**Files:**
- No new production files unless verification exposes a tested defect.

**Interfaces:**
- Verifies Tasks 1-4 as one opt-in diagnostic path.

- [ ] **Step 1: Run focused regression tests**

Run: `pytest -q tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py tests/test_qwen_gr00t_cot_v2.py tests/test_paired_geometry_probe.py tests/test_trace_intervention_probe.py`

- [ ] **Step 2: Run lint on changed Python files**

Run: `ruff check starVLA/model/modules/action_model/GR00T_ActionHeader.py starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py examples/simBenchmarks/CoT/geometry_probe/trace_intervention_probe.py examples/simBenchmarks/CoT/geometry_probe/run_trace_intervention_probe.py tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py tests/test_qwen_gr00t_cot_v2.py tests/test_trace_intervention_probe.py`

- [ ] **Step 3: Verify default-interface compatibility and repository scope**

Run: `git diff --check && git status --short && git diff --stat`

Confirm ignored preview images/videos remain untouched and no training, rollout, or checkpoint files changed.

- [ ] **Step 4: Record the exact commands, pass counts, and remaining GPU probe command**

The handoff must distinguish CPU code verification from an unrun real-checkpoint GPU experiment.
