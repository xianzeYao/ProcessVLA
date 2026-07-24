# CALVIN CPU Rerender Design

## Goal

Provide one CPU-only rerender entry point for CALVIN `ABC_D` and `ABCD_D` raw
sets. It must support deterministic random selection of language segments,
correct A/B/C/D scene routing, configurable multiprocessing, and the existing
H5 RGB-D/UVD output contract used by the LeRobot converter.

## Data flow

1. Load `ep_start_end_ids.npy`, `scene_info.npy`, frame NPZ files, and
   `lang_annotations/auto_lang_ann.npy` from the selected split.
2. Build complete language segments and discard segments whose source frames
   are missing or cross a source episode.
3. Select `--num-random-segments` with a local seed; `0` means all complete
   segments.
4. Resolve the scene independently for every selected segment using the
   source frame interval in `scene_info.npy`.
5. Balance selected segments across CPU workers by frame count. Each worker
   creates one CALVIN environment per scene it needs and renders only its own
   output directory.
6. Write worker summaries and one root `summary.json` manifest. The existing
   recursive H5 discovery in `build_calvin_rerender_lerobot.py` remains the
   conversion boundary.

## CLI

```text
--dataset-root PATH
--output-root PATH
--source-split {training,validation}
--num-random-segments N       # 0 means all
--seed INT
--workers INT                 # default 16
--comparison                  # optional comparison MP4/JPEG output
--overwrite
```

The runner is CPU-only: it sets `use_egl=False`, never sets CUDA/EGL device
variables, and does not require a GPU to be idle.

## Safety and validation

- Reject invalid worker counts, missing raw metadata, missing frame files, and
  incomplete language segments instead of silently producing partial output.
- Keep each worker's files isolated to prevent H5 name collisions.
- Verify that every selected segment produces one H5 file covering exactly its
  source range before writing the success manifest.
- Unit-test selection, scene routing, and frame-count balancing without
  starting PyBullet; run a small real CPU smoke test after implementation.
