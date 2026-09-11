# RoboCasa DA3 Feature Alignment Implementation Plan

> Implement the committed design in `docs/superpowers/specs/2026-09-05-robocasa-da3-feature-alignment-design.md` as an isolated V2 framework variant.

## Goal

Train the existing eight causal future-geometry tokens against frozen DA3 layer-23 future-image features while keeping numerical current/future depth reconstruction disabled and keeping the Action Expert input equal to native Qwen plus UVD tokens.

## Task 1: Define the contracts with failing tests

**Files:**

- Create: `tests/test_cot_v2_da3_feature_alignment.py`

Cover the RoboCasa `[0, 16]` video offsets and separated `future_image`, terminal clamping, 16x16-to-2x4 pooling, cosine loss validation, eight-token framework invariants, total loss composition, frozen teacher checkpoint exclusion, inference isolation, YAML values, and launcher dry-run.

Run:

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest -q tests/test_cot_v2_da3_feature_alignment.py
```

Expected: fail because the DA3 module, framework, data option, YAML, and launcher do not yet exist.

## Task 2: Implement the frozen DA3 teacher utilities

**Files:**

- Create: `starVLA/model/modules/da3_feature_alignment.py`

Implement future-RGB preprocessing, lazy local DA3 construction without importing its optional high-level export dependencies, strict local checkpoint loading, layer-23 patch extraction after the DualDPT input LayerNorm, 2x4 adaptive pooling, finite/shape checks, and token-wise cosine distance. Store the teacher behind a plain Python object so it is absent from student parameters and state dicts.

Run the utility-focused tests until green.

## Task 3: Add the isolated student framework

**Files:**

- Create: `starVLA/model/framework/VLM4A/QwenGR00TCoTV2DA3.py`
- Modify: `starVLA/training/train_starvla_cot_v2.py`

Register `QwenGR00TCoTV2DA3`, reuse V2 token ordering and action conditioning, replace the numerical depth decoder with a trainable 2560-to-2048 alignment projector, compute `action + 0.62*UVD + 0.15*DA3`, return zero numerical depth losses, and log raw/weighted DA3 loss through the existing V2 trainer. Ensure `predict_action` never creates the teacher.

Run framework/loss/isolation tests until green.

## Task 4: Add future RGB to RoboCasa only when requested

**Files:**

- Modify: `starVLA/dataloader/robocasa_lerobot_datasets.py`

When `cot_geometry.future_image_alignment` is enabled, request video offsets `[0, horizon]`. Keep `sample["image"]` at offset 0 and add only offset `horizon` as `sample["future_image"]`; rely on the existing video loader's terminal clamping.

Run dataset tests and existing RoboCasa geometry regressions.

## Task 5: Add the experiment entrypoints

**Files:**

- Create: `examples/modelExtensions/CoT/configs/qwen35_gr00t_robocasa_fourier_CoT_v2_q32_nodepthcond_da3_feature_alignment.yaml`
- Create: `examples/modelExtensions/CoT/scripts/run_qwen35_gr00t_robocasa_fourier_CoT_v2_q32_nodepthcond_da3_feature_alignment.sh`

Clone the future-only training setup, switch to the isolated framework, disable both numerical depth losses, enable DA3 alignment at 0.15, and record the model/source/layer/dimension/pooling/image-size fields in configuration.

Run YAML comparison and launcher dry-run tests.

## Task 6: Verify the real integration

Run:

```bash
/root/data/yxz/miniforge3/envs/CoT_linearATT/bin/python -m pytest -q \
  tests/test_cot_v2_da3_feature_alignment.py \
  tests/test_cot_v2_entrypoints.py \
  tests/test_cot_v2_uv_only.py \
  tests/test_qwen_gr00t_cot_v2.py \
  tests/test_robocasa_v4_lerobot_dataset.py
```

Then perform a local DA3 feature extraction smoke test at 224x224 and, resources permitting without disturbing active evaluations, one real training-batch forward. Confirm `[B,256,2048] -> [B,8,2048]`, finite losses, no numerical depth-decoder call, no DA3 tensors in the student state dict, and successful launcher dry-run.

Finally review `git diff`, commit only the implementation, and report the exact eight-GPU training command.
