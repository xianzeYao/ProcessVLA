# LIBERO V3 Post-Training Parallel Evaluation Design

## Objective

After the active eight-GPU LIBERO CoT V3 training run finishes successfully,
automatically evaluate its final exported model on both standard LIBERO and
LIBERO-plus without manual intervention.

## Trigger and checkpoint contract

- Monitor the active Accelerate launcher process for the current run.
- Do not start evaluation while training still owns the GPUs.
- After the launcher exits, require both the 60,000-step checkpoint and
  `final_model/pytorch_model.pt` to exist and be non-empty.
- Evaluate the explicit final-model path rather than resolving the numerically
  latest intermediate checkpoint.
- If training exits before producing the required artifacts, record failure and
  do not launch either evaluation.

## Parallel evaluation layout

Start both existing launchers concurrently after the checkpoint gate passes:

- Standard LIBERO uses GPUs 0,1,2,3, one suite per GPU, with 50 trials per task.
- LIBERO-plus uses GPUs 0,1,2,3,4,5,6,7, eight disjoint task workers, with one
  trial per task.
- GPUs 0-3 therefore host one policy server from each evaluation. The launchers
  use distinct port ranges (6694-6697 and 9883-9890) and distinct output trees.
- Disable rollout video saving for both evaluations.

## Resource and failure handling

- Before launching, wait until the training process has exited and GPU memory
  has been released.
- Record a top-level watcher log plus the normal per-server and per-worker logs
  produced by each evaluation launcher.
- Run the two evaluation launchers as independent background children and wait
  for both. Failure of one must not terminate the other.
- Record each exit status and a final combined status. Do not report aggregate
  success unless the corresponding launcher's `overall_results.json` exists.

## Output locations

- Standard LIBERO:
  `<training_run>_eval/libero/<timestamp>/`
- LIBERO-plus:
  `<training_run>_eval/libero_plus/<timestamp>/`
- Watcher log:
  `<training_run>/posttrain_parallel_eval.log`

## Verification

- Dry-run both existing launchers with the explicit checkpoint and intended GPU
  lists before arming the watcher.
- Confirm the watcher process is detached and points at the active training PID.
- Confirm no evaluation process starts before the final-model gate is satisfied.
