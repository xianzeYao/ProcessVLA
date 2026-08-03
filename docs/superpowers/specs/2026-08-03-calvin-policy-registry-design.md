# Calvin Policy Registry Fix

## Problem

CALVIN training uses `starVLA.dataloader.calvin_lerobot_datasets.CalvinDataConfig`,
but policy serving reconstructs the training transform through the centralized
`DATASET_NAMED_MIXTURES` and `ROBOT_TYPE_CONFIG_MAP` registries. The CALVIN
mixtures are not registered there. In addition, registry discovery only checks
`examples/<name>/train_files/data_registry`, while the repository stores
benchmark registries under deeper paths such as
`examples/simBenchmarks/<name>/train_files/data_registry`.

The result is that all CALVIN policy servers terminate before binding their
ports with `data_mix='calvin_task_ABC_D' not in DATASET_NAMED_MIXTURES`.

## Design

1. Make centralized registry discovery recursively find directories whose
   relative suffix is `train_files/data_registry` anywhere below `examples`.
   Preserve deterministic ordering and ignore unrelated directories named
   `data_registry`.
2. Add a CALVIN benchmark registry that reuses the existing
   `CalvinDataConfig` class and registers both training mixtures:
   `calvin_task_ABC_D` and `calvin_task_ABCD_D`.
3. Map both mixtures to the `franka` robot type so policy serving selects the
   same action/state keys and tensor-only relative-action transform used during
   training.
4. Do not change checkpoint files, saved statistics, training configs, CALVIN
   environment code, or evaluation observation construction.

## Validation

Use a subprocess-based registry test to avoid Python module-cache effects. It
must fail before the fix because `calvin_task_ABC_D` is absent, then pass after
the fix by asserting:

- nested benchmark registry directories are discovered;
- both CALVIN mixture names exist;
- both resolve uniquely to robot type `franka`;
- `ROBOT_TYPE_CONFIG_MAP['franka']` is `CalvinDataConfig`;
- constructing `PolicyNormProcessor` from the trained 80k checkpoint succeeds.

Finally, launch one policy server on a free GPU/port and verify that it reaches
the listening state. This smoke test is stopped after the port opens; the full
eight-GPU evaluation remains user-controlled.

## Safety

All existing uncommitted training-config and evaluation-script changes remain
untouched. The implementation is limited to registry discovery, one CALVIN
registry module, and its focused regression test.
