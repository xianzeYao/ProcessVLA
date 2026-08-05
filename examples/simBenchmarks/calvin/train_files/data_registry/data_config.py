"""CALVIN benchmark data config and training mixtures."""

from starVLA.dataloader.calvin_lerobot_datasets import CalvinDataConfig


ROBOT_TYPE_CONFIG_MAP = {
    "calvin_franka": CalvinDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "calvin_task_ABC_D": [("calvin_task_ABC_D", 1.0, "calvin_franka")],
    "calvin_task_ABCD_D": [("calvin_task_ABCD_D", 1.0, "calvin_franka")],
}
