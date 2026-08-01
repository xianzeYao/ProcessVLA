# RoboCasa Variable-GPU Evaluation Design

## Goal

Allow the existing RoboCasa-GR1 multi-GPU evaluator to run with any sensible
number of unique GPU workers instead of requiring exactly four GPUs. The
evaluation protocol remains 24 tasks with the configured number of episodes
per task.

## Scope

- Accept between 1 and 24 unique GPU identifiers.
- Keep one policy server and one sequential simulator worker per GPU entry.
- Keep deterministic round-robin task assignment:
  `worker_id = task_index % num_workers`.
- Keep task definitions, episode counts, seeds, action settings, result files,
  aggregation, and metric definitions unchanged.
- Preserve the current four-GPU behavior exactly.

Duplicate GPU identifiers are rejected. Running two policy servers on one GPU
is outside this change because it changes memory contention and makes the
meaning of “GPU worker count” ambiguous.

## Implementation

Two existing fixed-size checks will share the same policy:

1. `run_multigpu_eval.sh` validates that the parsed GPU list contains 1–24
   unique entries before launching servers.
2. `robocasa_eval_protocol.py` applies the same validation when building the
   manifest, so direct Python callers cannot bypass it.

No scheduling loop needs to change: the shell launcher and manifest builder
already derive worker IDs, ports, and task assignment from the GPU-list length.

## Error Handling

The launcher and manifest builder fail before starting a policy server when:

- the GPU list is empty;
- more than 24 GPU entries are supplied; or
- a GPU identifier occurs more than once.

Error messages report the invalid worker count or duplicate identifier.

## Verification

Automated tests cover:

- eight unique GPUs produce eight workers and assign exactly three of the 24
  tasks to each worker;
- existing four-GPU manifests remain valid;
- empty, over-24, and duplicate GPU lists are rejected.

A shell dry run with `GPUS=0,1,2,3,4,5,6,7` must produce eight server/worker
assignments, 24 task assignments, and no launched processes.
