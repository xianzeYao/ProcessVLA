# LIBERO V3 Extreme-Task Trace Audit

## Goal

Use the existing QwenGR00TCoTV3 checkpoint to inspect whether its three
left-finger/right-finger/wrist UVD traces are accurate and action-aligned on
representative closed-loop LIBERO rollouts.  For every standard LIBERO suite,
select the two highest-scoring and two lowest-scoring tasks from the completed
50-trial V3 evaluation, rerun five fixed-seed episodes per selected task, and
produce synchronized action/trace/depth-error visualizations.

This is a checkpoint diagnostic.  It does not change training, losses, model
architecture, or the checkpoint.

## Ranking and episode selection

- Rank tasks from the completed standard LIBERO V3 worker logs under
  `.../libero/20260814_071939/logs`.
- Parse each task's 50 `Success:` results and retain task id, language, success
  count, and success rate.
- Within each suite, select the two highest and two lowest success rates.
- Resolve ties deterministically by task id.
- For a best task, prefer the first originally successful initial-state index.
- For a worst task, prefer the first originally failed initial-state index; if
  the task had no failure, use its first successful index.
- Hold that representative initial-state index fixed and rerun it with the
  five deterministic policy/environment seeds `7,8,9,10,11`. Seed lists are
  recorded in the manifest and may be overridden explicitly by the CLI.
- Record the original result and the rerun result separately because the
  original policy server did not guarantee an identical Torch sampling stream.

The initial run covers 16 tasks and five rollouts per task, for 80 rollouts in
total.  Selection logic is kept independent of the ranking source so a
completed LIBERO-Plus result can later drive the same audit without changing
rendering or metrics.

## Geometry inference path

The V3 action head and geometry decoders already share one Qwen forward, but
the normal `predict_action` response currently returns actions only.  Add an
optional `return_geometry` request that, when enabled, decodes and returns:

- current dense depth;
- future dense depth;
- flattened V3 UVD tokens;
- UVD normalized times and landmark ids/order metadata.

Action-only requests retain their existing response and execution path.  The
diagnostic path must reuse the same hidden split used to produce the action;
it must not run a second Qwen forward.

## Online rollout and ground truth

Run the selected task and initial-state index in the normal LIBERO simulator
with the same RGB preprocessing and action chunk cadence as standard
evaluation.  At each action refresh anchor:

1. request the action chunk and V3 geometry from the policy server;
2. canonicalize time-major UVD into `[time, landmark, 3]` for
   left/right/wrist;
3. record the predicted trace and executed 7D action chunk;
4. record simulator body positions for the three configured landmark bodies
   after every executed action;
5. project the realized body positions into the flipped agent-view camera used
   by the model;
6. align realized positions to the checkpoint's four trajectory times across
   the eight-step action horizon.

If an episode ends before a complete future horizon, unavailable future points
are marked invalid rather than padded.  Projection validity and in-frame
validity are recorded separately.  Point depth is camera-frame `z`, matching
the training sidecar contract.

## Metrics

Report metrics per episode, suite, selected-task rank group, time point, and
landmark:

- UV ADE and FDE in model-input pixels;
- absolute point-depth MAE/RMSE in millimetres;
- adjacent `delta-d` MAE in millimetres;
- approach/retreat direction accuracy with a configurable depth-change dead
  zone, default 2 mm;
- gripper-center UV/depth error derived from the two finger points;
- left-right span error after camera-frame XYZ back-projection;
- wrist-to-contact-center direction error;
- persistence baselines that repeat the anchor point across the future times;
- current/future dense-depth MAE as secondary context;
- rollout success, step count, and action-chunk latency.

At every action-refresh anchor, compute three independent values for each of
left, right, and wrist: UV ADE is the mean valid horizon-point Euclidean pixel
error; point-depth MAE is the mean valid horizon-point absolute camera-z
error; adjacent `delta-d` MAE is the mean valid error over consecutive horizon
segments.  These nine anchor values form the per-episode landmark curves.

Geometry-derived span and direction are computed in camera-frame XYZ, not by
subtracting raw UVD coordinates.

## Visual artifacts

Produce one MP4 and one static summary PNG for every selected fixed-seed
rollout.  Also produce one five-seed aggregate summary figure per selected
task.  Each video frame uses a stable dashboard containing:

- agent-view RGB with predicted and realized LRW trace overlays;
- a compact wrist-view inset;
- full-episode executed action curves for x/y/z/roll/pitch/yaw/gripper with a
  moving time cursor;
- per-landmark predicted versus realized depth for the active chunk;
- three fixed-axis full-episode error panels for UV ADE, point-depth MAE, and
  adjacent `delta-d` MAE; every panel contains separate left, right, and wrist
  curves plus a moving current-anchor cursor;
- success state, task rank, task success rate, current step, and valid-point
  counts.

Colors remain stable across every artifact: left, right, and wrist use fixed
landmark colors; prediction and ground truth use distinct line styles.

Write:

- `selection.json` with the parsed task table, 16 selected tasks, their fixed
  initial-state indices, and the five audit seeds;
- one resumable raw NPZ/JSON record per episode;
- 80 MP4s, 80 per-rollout summary PNGs, and 16 five-seed task summaries;
- one suite contact sheet per suite;
- `episodes.jsonl`, `summary.json`, and `summary.csv`;
- a run manifest containing checkpoint, source evaluation, seeds, commands,
  software settings, and artifact paths.

## Reliability and scope controls

- Fix NumPy and Torch seeds for every rerun, use the same five-seed set for all
  selected tasks, and reuse each task's representative initial-state index
  across its five seeds and any retries.
- Cache raw rollout records before rendering so visualization can be regenerated
  without inference.
- Resume completed episodes by validating their manifest and raw record.
- Do not mutate or stop the running LIBERO-Plus evaluation.  Real generation
  starts only when sufficient GPU memory is available.
- If a selected rerun outcome differs from the original outcome, keep it and
  label the mismatch; do not silently search until a desired success/failure is
  obtained.
- Do not claim held-out expert-trace accuracy from these results.  Realized
  rollout traces measure policy self-consistency and closed-loop geometry on
  evaluation states.

## Verification

- Unit-test worker-log parsing, deterministic extreme selection, and preferred
  episode-index selection.
- Unit-test V3 time-major canonicalization and future-step alignment, including
  early termination masks.
- Unit-test projection/flip conventions against visible simulator landmarks.
- Unit-test UV/depth/delta-depth/persistence metrics on synthetic examples.
- Unit-test that action-only server responses are unchanged and
  `return_geometry` performs one shared backbone forward.
- Smoke-test one short selected episode before starting all suites.
- Validate that every generated MP4 decodes, has nonzero frames, and matches its
  raw record; validate exactly four selected tasks and 20 rollouts per suite,
  16 selected tasks and 80 rollouts in total.

## Non-goals

- No V3.1 representation change.
- No token intervention or causal patching in this first audit.
- No RPY recovery claim.
- No retraining.
- No automatic replacement of selected episodes based on attractive outcomes.
