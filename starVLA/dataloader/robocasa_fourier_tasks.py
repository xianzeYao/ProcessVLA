"""Canonical 24-task scope for the StarVLA GR1 Fourier fine-tuning corpus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FOURIER_MIXTURE_NAME = "fourier_gr1_unified_1000"
FOURIER_ROBOT_TYPE = "fourier_gr1_arms_waist"
FOURIER_DATASET_SUFFIX = "_GR1ArmsAndWaistFourierHands_1000"
FOURIER_LANGUAGE_PREFIX = "unlocked_waist: "


@dataclass(frozen=True)
class FourierTask:
    basename: str

    @property
    def source_dataset_name(self) -> str:
        return f"gr1_unified.{self.basename}"

    @property
    def official_dataset_name(self) -> str:
        return f"{self.source_dataset_name}{FOURIER_DATASET_SUFFIX}"

    @property
    def hdf5_filename(self) -> str:
        return f"{self.basename}.hdf5"


FOURIER_TASKS = (
    FourierTask("PnPBottleToCabinetClose"),
    FourierTask("PnPCanToDrawerClose"),
    FourierTask("PnPCupToDrawerClose"),
    FourierTask("PnPMilkToMicrowaveClose"),
    FourierTask("PnPPotatoToMicrowaveClose"),
    FourierTask("PnPWineToCabinetClose"),
    FourierTask("PosttrainPnPNovelFromCuttingboardToBasketSplitA"),
    FourierTask("PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA"),
    FourierTask("PosttrainPnPNovelFromCuttingboardToPanSplitA"),
    FourierTask("PosttrainPnPNovelFromCuttingboardToPotSplitA"),
    FourierTask("PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA"),
    FourierTask("PosttrainPnPNovelFromPlacematToBasketSplitA"),
    FourierTask("PosttrainPnPNovelFromPlacematToBowlSplitA"),
    FourierTask("PosttrainPnPNovelFromPlacematToPlateSplitA"),
    FourierTask("PosttrainPnPNovelFromPlacematToTieredshelfSplitA"),
    FourierTask("PosttrainPnPNovelFromPlateToBowlSplitA"),
    FourierTask("PosttrainPnPNovelFromPlateToCardboardboxSplitA"),
    FourierTask("PosttrainPnPNovelFromPlateToPanSplitA"),
    FourierTask("PosttrainPnPNovelFromPlateToPlateSplitA"),
    FourierTask("PosttrainPnPNovelFromTrayToCardboardboxSplitA"),
    FourierTask("PosttrainPnPNovelFromTrayToPlateSplitA"),
    FourierTask("PosttrainPnPNovelFromTrayToPotSplitA"),
    FourierTask("PosttrainPnPNovelFromTrayToTieredbasketSplitA"),
    FourierTask("PosttrainPnPNovelFromTrayToTieredshelfSplitA"),
)


def fourier_dataset_names() -> list[str]:
    return [task.official_dataset_name for task in FOURIER_TASKS]


def fourier_mixture_entries() -> list[tuple[str, float, str]]:
    return [(task.official_dataset_name, 1.0, FOURIER_ROBOT_TYPE) for task in FOURIER_TASKS]


def task_for_source_dataset(source_dataset_name: str) -> FourierTask:
    for task in FOURIER_TASKS:
        if task.source_dataset_name == source_dataset_name:
            return task
    raise KeyError(f"Unknown Fourier RoboCasa source task: {source_dataset_name}")


def canonicalize_fourier_instruction(remark: Any) -> str:
    """Convert Teleop episode remarks to the official X task language form.

    Teleop-Sim stores the fine-grained object instruction in ``remarks`` without
    the embodiment prefix, while the official Fourier metadata prepends
    ``unlocked_waist:``.  Keeping this conversion at the metadata boundary
    avoids changing the generic LeRobot language loader.
    """
    if isinstance(remark, bytes):
        remark = remark.decode("utf-8")
    text = str(remark).strip()
    if not text:
        raise ValueError("Teleop episode remarks must contain a non-empty instruction")
    if text.startswith(FOURIER_LANGUAGE_PREFIX):
        return text
    return f"{FOURIER_LANGUAGE_PREFIX}{text}"


def build_episode_instruction_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Build deterministic task metadata and episode-to-task-index mappings."""
    tasks = [{"task_index": 0, "task": ""}]
    task_indices: dict[str, int] = {}
    episode_to_task: dict[int, int] = {}
    for row in sorted(rows, key=lambda item: int(item["episode_index"])):
        instruction = canonicalize_fourier_instruction(row.get("remarks", ""))
        task_index = task_indices.get(instruction)
        if task_index is None:
            task_index = len(tasks)
            task_indices[instruction] = task_index
            tasks.append({"task_index": task_index, "task": instruction})
        episode_to_task[int(row["episode_index"])] = task_index
    return tasks, episode_to_task
