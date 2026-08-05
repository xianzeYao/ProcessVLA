# GR00T N1.6 and N1.7 Action Heads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable, official-style GR00T N1.6 and N1.7 flow-matching action heads while preserving the existing StarVLA N1.5-derived head.

**Architecture:** Add a shared `AlternateVLDiT` to the existing DiT module, then build N1.6 and N1.7 heads in separate files. Both heads use one state token plus an action chunk, one noise/time sample per input item, category-specific encoders/decoders, action-only learned absolute positions, and masked flow-matching loss. N1.7 extends N1.6 with optional VLM self-attention refinement and state-history support.

**Tech Stack:** Python, PyTorch, pytest, OmegaConf-compatible configuration objects.

---

### Task 1: Add AlternateVLDiT routing

**Files:**
- Modify: `starVLA/model/modules/action_model/cross_attention_dit.py`
- Test: `tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py`

- [ ] Write a failing test that replaces transformer blocks with recorders and verifies the route `non-image cross -> full self -> image cross -> full self`.
- [ ] Run the focused test and confirm it fails because `AlternateVLDiT` does not exist.
- [ ] Implement `AlternateVLDiT` additively without changing the existing `DiT` behavior.
- [ ] Validate image/text masks and preserve the same time-conditioned output projection contract as `DiT`.
- [ ] Run the focused routing test and existing action-model tests.

### Task 2: Implement the N1.6 head

**Files:**
- Create: `starVLA/model/modules/action_model/GR00T_N16_ActionHeader.py`
- Modify: `tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py`

- [ ] Write failing tiny-config tests for the public forward signature, one state plus `H` action tokens, absence of future tokens, default embodiment zero, and exactly one `sample_time(B)` call.
- [ ] Write failing tests for masked loss, required `image_mask`, horizon/action/state shape checks, and inference output shape.
- [ ] Implement configuration helpers, VLLN conditioning, category-specific state/action encoders and decoder, action-only learned position embeddings, flow-matching training, and Euler inference.
- [ ] Keep repeat expansion outside the head; do not repeat inputs internally.
- [ ] Run all N1.6 tests and static compilation.

### Task 3: Implement the N1.7 head

**Files:**
- Create: `starVLA/model/modules/action_model/GR00T_N17_ActionHeader.py`
- Modify: `tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py`

- [ ] Write failing tests showing N1.7 accepts state history, flattens it into one state token, and invokes optional VLM self-attention refinement.
- [ ] Implement N1.7 as a focused extension of N1.6, defaulting to official-style 32 layers/1536 width and optional four-layer 2048-hidden VLM refinement.
- [ ] Keep RTC as an explicitly unsupported compatibility surface instead of silently approximating it.
- [ ] Run N1.7 tests and static compilation.

### Task 4: Regression and handoff

**Files:**
- Verify: `starVLA/model/modules/action_model/GR00T_ActionHeader.py`
- Verify: `starVLA/model/modules/action_model/cross_attention_dit.py`
- Verify: `tests/model/modules/action_model/test_gr00t_n16_n17_action_heads.py`

- [ ] Run the focused new test module.
- [ ] Run relevant existing action-model tests, or document any environment dependency that prevents execution.
- [ ] Run `python -m compileall` on changed Python modules.
- [ ] Run `git diff --check` and inspect the final diff for accidental edits to the old head.
- [ ] Commit the implementation as a separate commit after verification.
