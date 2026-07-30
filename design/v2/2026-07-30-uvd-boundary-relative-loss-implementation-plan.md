# UVD Boundary and Relative-Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:test-driven-development` and verify each task before proceeding.

**Goal:** Make UVD supervision representable at image boundaries, add an
adjacent-motion auxiliary term to V2, and expose diagnostics that distinguish
coordinate accuracy from trajectory-motion accuracy.

**Architecture:** LIBERO, CALVIN, and RoboCasa continue to share the existing
UVD projection/target path. Pixel validity is tightened to the network's
representable pixel-center range, while normalized coordinates receive only a
final numerical clamp. The V2 objective combines the existing all-point
Smooth-L1 term with a time-major, same-hand adjacent-delta Smooth-L1 term;
V1 remains unchanged.

**Tech Stack:** NumPy, PyTorch, OmegaConf YAML, pytest.

## Global Constraints

- Preserve all unrelated dirty-worktree changes.
- Do not change V1's objective.
- Keep every fixed V2 UVD query token active; coordinate validity masks only
  numeric supervision.
- Use `lambda_uvd_relative: 0.1` in all four V2 YAMLs.
- Do not add depth-map/UVD consistency.

### Task 1: Strict UVD boundary contract

**Files:**
- Modify: `tests/test_cot_geometry.py`
- Modify: `starVLA/dataloader/gr00t_lerobot/cot_geometry.py`

- [x] Add a failing test proving `u == W-1` is valid while
  `W-1 < u < W` is invalid.
- [x] Add a failing test proving normalization clamps floating-point overflow
  into `[0,1]`.
- [x] Run the focused tests and confirm the intended failures.
- [x] Tighten projection validity to `[0,W-1] x [0,H-1]`, retain positive
  finite depth checks, and add numerical normalization clamp.
- [x] Attach sampled out-of-frame and boundary-clamp masks to training examples.
- [x] Run the focused tests and confirm they pass.

### Task 2: V2 adjacent relative UVD objective

**Files:**
- Modify: `tests/test_cot_losses.py`
- Modify: `tests/test_qwen_gr00t_cot_v2.py`
- Modify: `starVLA/model/modules/cot_losses.py`
- Modify: `starVLA/model/framework/VLM4A/QwenGR00TCoTV2.py`
- Modify: `examples/modelExtensions/CoT/configs/qwen35_gr00t_*_CoT_v2.yaml`

- [x] Add failing tests for same-hand, adjacent-time deltas, invalid-segment
  masking, and V2 loss composition.
- [x] Run the focused tests and confirm the intended failures.
- [x] Implement masked adjacent-delta Smooth L1 for time-major tensors.
- [x] Return `uvd_absolute_loss`, `uvd_relative_loss`, and their V2 combination.
- [x] Add `lambda_uvd_relative: 0.1` to all V2 configurations.
- [x] Run the focused tests and confirm they pass.

### Task 3: Diagnostics and training logs

**Files:**
- Modify: `tests/test_cot_diagnostics.py`
- Modify: `tests/test_cot_trainer_objective.py`
- Modify: `starVLA/training/cot_test_diagnostics.py`
- Modify: `starVLA/training/train_starvla_cot_v1.py`

- [x] Add failing tests for U/V/D split metrics, adjacent UVD motion error,
  boundary ratios, and optional V2 sub-loss logging.
- [x] Run the focused tests and confirm the intended failures.
- [x] Add the metrics while preserving V1 output compatibility.
- [x] Run the focused tests and confirm they pass.

### Task 4: Verification and documentation

**Files:**
- Modify: `design/v2/depth-uvd-geometric-cot-v2-design-report.md`

- [x] Run all focused CoT geometry/loss/trainer tests.
- [x] Run the broader CoT regression set.
- [x] Run Python syntax compilation and YAML loading checks.
- [x] Update the V2 report with the exact objective, boundary contract, and
  diagnostic fields.
- [x] Inspect the final diff and report what is code-verified versus what still
  requires a training experiment.
