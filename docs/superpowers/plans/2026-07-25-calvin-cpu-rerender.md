# CALVIN CPU Rerender Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CPU-only CALVIN rerender runner with deterministic random segment sampling, correct scene routing, and configurable multiprocessing.

**Architecture:** Keep the existing frame-to-H5/video/UVD writer as the rendering core, fix its environment construction to load the scene selected by each source interval, and add a process-level orchestrator that assigns selected segments to isolated worker directories. The root manifest records the exact source segment, scene, worker, and output file for downstream LeRobot conversion.

**Tech Stack:** Python 3.8+, argparse, multiprocessing, NumPy, OmegaConf/Hydra, PyBullet, h5py, imageio, pytest.

## Global Constraints

- CPU-only: `use_egl=False`; no CUDA/EGL environment selection.
- `--num-random-segments 0` means all complete language segments.
- Default worker count is 16; explicit `--workers` is allowed up to available CPUs.
- Scene selection comes from `scene_info.npy`, never from the D split config alone.
- Existing user modifications outside the Calvin rerender files must remain untouched.

---

### Task 1: Pure planning helpers

**Files:**
- Create: `examples/simBenchmarks/calvin/data_preparation/rerender_calvin_cpu.py`
- Test: temporary `/tmp/test_calvin_rerender_cpu.py` during TDD, removed after verification

**Interfaces:**
- `select_random_segments(segments, count, seed) -> list[dict]`
- `scene_for_frame(scene_intervals, frame_id) -> str`
- `balance_segments(segments, workers) -> list[list[dict]]`

- [ ] Write failing tests for deterministic selection, scene interval routing, and balanced frame totals.
- [ ] Run the tests and confirm they fail because the module is not present.
- [ ] Implement the three pure helpers with explicit validation.
- [ ] Run the tests and confirm they pass.

### Task 2: CPU environment and worker renderer

**Files:**
- Modify: `examples/simBenchmarks/calvin/data_preparation/rerender_calvin_episodes.py`
- Modify: `examples/simBenchmarks/calvin/data_preparation/rerender_calvin_segments.py`
- Modify: `examples/simBenchmarks/calvin/data_preparation/rerender_calvin_cpu.py`

**Interfaces:**
- `make_env(config_dir, scene_name=None) -> PlayTableSimEnv`
- worker entry point consumes `(dataset_root, split, worker_segments, worker_output, comparison)` and writes isolated H5/summary files.

- [ ] Load the split config for shared robot/camera settings, then override the scene with `conf/scene/calvin_scene_{A|B|C|D}.yaml` and the correct environment data path.
- [ ] Remove tactile rendering, set `use_egl=False`, and instantiate one environment per scene per worker.
- [ ] Reuse `render_episode` so action/state, RGB, depth, camera calibration, and UVD remain unchanged.
- [ ] Attach source segment index, task text, and scene to each worker summary.

### Task 3: CLI orchestration and manifest

**Files:**
- Modify: `examples/simBenchmarks/calvin/data_preparation/rerender_calvin_cpu.py`

**Interfaces:**
- CLI supports `--dataset-root`, `--output-root`, `--source-split`, `--num-random-segments`, `--seed`, `--workers`, `--comparison`, and `--overwrite`.

- [ ] Validate raw metadata and build complete language segments.
- [ ] Resolve scene intervals and select segments deterministically.
- [ ] Balance work by frame count and launch bounded CPU processes.
- [ ] Validate one complete H5 per selected segment and write root `summary.json`.
- [ ] Exit nonzero on worker failure or output-count/range mismatch.

### Task 4: Verification

**Files:**
- No permanent test files required.

- [ ] Run temporary unit tests.
- [ ] Run `--help` and a no-render planning check.
- [ ] Run two real CPU smoke tests with 5 random segments per dataset and 16 workers, without comparison videos first.
- [ ] Inspect summaries and verify no GPU/EGL process or temporary test file remains.
