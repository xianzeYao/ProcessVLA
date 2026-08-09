# Q0 Depth Condition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in Q0 experiment that exposes existing supervised current-depth and future-depth Qwen hidden tokens directly to the unchanged action head.

**Architecture:** Keep geometry generation and the Q0 DiT intact. A strict boolean in `framework.geometry` controls whether the shared action-condition builder returns `[native, UVD]` or `[native, current depth, future depth, UVD]`; training and inference already consume that shared result.

**Tech Stack:** Python 3.10, PyTorch, OmegaConf YAML, pytest, Accelerate with DeepSpeed ZeRO-2.

## Global Constraints

- Keep Qwen3.5-4B on the existing 32-layer path.
- Keep current-depth, future-depth, and UVD geometry tokens and losses unchanged.
- Keep zero learned future/action query tokens.
- Keep the 16-layer, width-768 `DiT-B`, regular interleaved self/cross attention, repeat 8, and four inference steps.
- Do not add AlternateVLDiT, VLLN, a condition adapter, or new trainable parameters.
- Default `include_depth_in_action_condition` to `false` for checkpoint and YAML compatibility.
- Do not modify or stage unrelated dirty geometry-probe, Libero, or existing design-document files.

---

## File Structure

- Modify `starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py`: parse the strict flag and build the selected condition layout and mask.
- Modify `tests/test_qwen_gr00t_cot_v2.py`: cover default, enabled, masks, ordering, and invalid configuration values.
- Create `examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond.yaml`: preserve the Q0 run and enable only depth conditioning.
- Modify `tests/test_cot_v2_entrypoints.py`: prove the new YAML differs from Q0 only by `run_id` and the new flag.

### Task 1: Implement the opt-in depth condition

**Files:**
- Modify: `starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py:38-45,80-130,194-216`
- Test: `tests/test_qwen_gr00t_cot_v2.py:15-100`

**Interfaces:**
- Consumes: `GeometryHiddenSplit` with `native`, `depth_current`, `depth_future`, and `uvd` tensors shaped `[B,L,D]`.
- Produces: `Qwen_GR00T_CoT_V2.include_depth_in_action_condition: bool` and `_build_action_condition(...) -> tuple[torch.Tensor, torch.Tensor | None]`.

- [ ] **Step 1: Extend the test helper and write failing layout tests**

Change the helper to set the option explicitly:

```python
def make_uninitialized_model(*, depth_queries=2, points=3, hands=2, include_depth=False):
    model = Qwen_GR00T_CoT_V2.__new__(Qwen_GR00T_CoT_V2)
    model.geometry_layout = GeometryTokenLayout(
        depth_query_count=depth_queries,
        uvd_points_per_hand=points,
        hand_count=hands,
    )
    model.include_depth_in_action_condition = include_depth
    return model
```

Keep the existing exclusion test as the false/default contract and add:

```python
def test_action_condition_includes_depth_groups_in_causal_order_when_enabled():
    model = make_uninitialized_model(
        depth_queries=1,
        points=2,
        hands=1,
        include_depth=True,
    )
    all_hidden = torch.arange(6, dtype=torch.float32).view(1, 6, 1)
    split = model._split_geometry_hidden(all_hidden, native_token_count=2)

    condition, mask = model._build_action_condition(
        split,
        native_attention_mask=torch.tensor([[True, False]]),
    )

    assert condition.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert mask.tolist() == [[True, False, True, True, True, True]]
```

- [ ] **Step 2: Run the focused tests and confirm the enabled case fails**

Run:

```bash
pytest -q tests/test_qwen_gr00t_cot_v2.py -k 'action_condition'
```

Expected: the existing false-path test passes and the new enabled-path test fails because depth tokens are still excluded.

- [ ] **Step 3: Add strict boolean parsing and the minimal condition implementation**

Add a focused module helper:

```python
def _strict_bool_option(mapping: Any, name: str, *, default: bool = False) -> bool:
    value = mapping.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    return value
```

In `__init__`, after reading `geometry`, set:

```python
self.include_depth_in_action_condition = _strict_bool_option(
    geometry,
    "include_depth_in_action_condition",
)
```

Update `_build_action_condition` without changing its signature:

```python
groups = [split.native]
if self.include_depth_in_action_condition:
    groups.extend([split.depth_current, split.depth_future])
groups.append(split.uvd)
condition = torch.cat(groups, dim=1)

expected_geometry_tokens = self.geometry_layout.uvd_token_count
if self.include_depth_in_action_condition:
    expected_geometry_tokens += 2 * self.geometry_layout.depth_query_count
expected_length = split.native.shape[1] + expected_geometry_tokens
if condition.shape[1] != expected_length:
    raise RuntimeError(
        f"action condition has {condition.shape[1]} tokens, expected {expected_length}"
    )
```

When a native mask exists, append `expected_geometry_tokens` true values rather than only the UVD count.

- [ ] **Step 4: Add and run the strict-option test**

Import `_strict_bool_option` and add:

```python
def test_depth_condition_option_rejects_non_boolean_values():
    with pytest.raises(ValueError, match="must be a boolean"):
        _strict_bool_option({"include_depth_in_action_condition": "true"},
                            "include_depth_in_action_condition")
```

Run:

```bash
pytest -q tests/test_qwen_gr00t_cot_v2.py -k 'action_condition or depth_condition_option'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the source and focused tests**

```bash
git add starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py tests/test_qwen_gr00t_cot_v2.py
git commit -m "feat: condition q0 actions on depth latents"
```

### Task 2: Add an isolated RoboCasa experiment YAML

**Files:**
- Create: `examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond.yaml`
- Modify: `tests/test_cot_v2_entrypoints.py`

**Interfaces:**
- Consumes: `framework.geometry.include_depth_in_action_condition` implemented in Task 1.
- Produces: a launchable Q0 depth-condition experiment configuration with a unique run ID.

- [ ] **Step 1: Write a failing YAML-equivalence test**

Add:

```python
def test_robocasa_q0_depth_condition_yaml_changes_only_condition_content():
    config_root = ROOT / "examples/modelExtensions/CoT/configs"
    base = OmegaConf.to_container(
        OmegaConf.load(config_root / "qwen35_gr00t_robocasa_fourier_CoT_v2_q0.yaml"),
        resolve=True,
    )
    experiment = OmegaConf.to_container(
        OmegaConf.load(config_root / "qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond.yaml"),
        resolve=True,
    )

    assert experiment.pop("run_id") == (
        "qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond_8gpu_bs16"
    )
    base.pop("run_id")
    assert experiment["framework"]["geometry"].pop(
        "include_depth_in_action_condition"
    ) is True
    assert "include_depth_in_action_condition" not in base["framework"]["geometry"]
    assert experiment == base
```

- [ ] **Step 2: Run the test and confirm the missing YAML failure**

Run:

```bash
pytest -q tests/test_cot_v2_entrypoints.py::test_robocasa_q0_depth_condition_yaml_changes_only_condition_content
```

Expected: FAIL because the new YAML does not exist.

- [ ] **Step 3: Create the YAML from the exact Q0 baseline**

Copy the Q0 YAML, change only:

```yaml
run_id: qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond_8gpu_bs16
```

and add under `framework.geometry`:

```yaml
include_depth_in_action_condition: true
```

- [ ] **Step 4: Run config tests**

Run:

```bash
pytest -q tests/test_cot_v2_entrypoints.py::test_robocasa_q0_depth_condition_yaml_changes_only_condition_content
```

Expected: PASS.

- [ ] **Step 5: Commit the YAML and config test**

```bash
git add examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond.yaml tests/test_cot_v2_entrypoints.py
git commit -m "exp: add robocasa q0 depth condition config"
```

### Task 3: Verify compatibility and produce the launch command

**Files:**
- Verify only; no source file is expected to change.

**Interfaces:**
- Consumes: Task 1 source/tests and Task 2 experiment YAML.
- Produces: passing focused regression evidence and an exact eight-GPU command.

- [ ] **Step 1: Run the V2 framework and entrypoint suites**

```bash
pytest -q tests/test_qwen_gr00t_cot_v2.py tests/test_cot_v2_entrypoints.py
```

Expected: all tests pass.

- [ ] **Step 2: Compile the changed Python modules**

```bash
python -m py_compile \
  starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py \
  tests/test_qwen_gr00t_cot_v2.py \
  tests/test_cot_v2_entrypoints.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Dry-run the exact eight-GPU launch**

```bash
CONFIG_YAML=examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond.yaml \
RUN_ID=qwen35_gr00t_robocasa_fourier_CoT_v2_q0_depthcond_8gpu_bs16 \
NUM_PROCESSES=8 \
MAIN_PROCESS_PORT=29520 \
DRY_RUN=1 \
bash examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_CoT_v2_common.sh
```

Expected output includes `--num_processes 8`, the new YAML path,
`train_starvla_cot_v2.py`, and the new run ID.

- [ ] **Step 4: Inspect final scope**

```bash
git status --short
git log -3 --oneline
```

Expected: implementation commits contain only the two source/test tasks; the pre-existing unrelated dirty files remain unstaged and unchanged.

