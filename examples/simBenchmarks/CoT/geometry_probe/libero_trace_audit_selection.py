"""Select reproducible LIBERO trace-audit cases from completed worker logs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable, Sequence


_TASK_RECORD = re.compile(r"^\s*Task:\s*(?P<language>.+?)\s*$")
_SUCCESS_RECORD = re.compile(r"\bSuccess:\s*(?P<outcome>True|False)\s*$")
_CURRENT_RATE_RECORD = re.compile(
    r"\bCurrent task success rate:\s*(?P<rate>[0-9]+(?:\.[0-9]+)?)\s*$"
)
_RATE_TOLERANCE = 0.005
_DEFAULT_SEEDS = (7, 8, 9, 10, 11)


@dataclass(frozen=True)
class TaskEvaluation:
    """Ordered outcomes for one completed task in one LIBERO suite."""

    suite: str
    task_id: int
    language: str
    outcomes: tuple[bool, ...]

    @property
    def success_rate(self) -> float:
        if not self.outcomes:
            raise ValueError("task evaluation has no outcomes")
        return sum(self.outcomes) / len(self.outcomes)


@dataclass(frozen=True)
class AuditCase:
    """One deterministic re-evaluation request for a selected task."""

    suite: str
    task_id: int
    language: str
    rank_group: str
    initial_state_index: int
    seed: int
    original_success: bool


def parse_worker_log(path: str | Path, suite: str) -> list[TaskEvaluation]:
    """Parse complete task blocks from a standard LIBERO worker log.

    A block is complete only when its repeated Task/Success episode records are
    followed by a ``Current task success rate`` record. The task id is the
    zero-based completed-block order, not an episode counter from the log.
    """

    evaluations: list[TaskEvaluation] = []
    block_language: str | None = None
    pending_language: str | None = None
    outcomes: list[bool] = []

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        task_match = _TASK_RECORD.match(line)
        success_match = _SUCCESS_RECORD.search(line)
        rate_match = _CURRENT_RATE_RECORD.search(line)

        if task_match:
            language = task_match.group("language")
            if pending_language is not None:
                raise ValueError(f"line {line_number}: Task record lacks a Success record")
            if block_language is None:
                block_language = language
            elif language != block_language:
                raise ValueError(f"line {line_number}: task language changed within a block")
            pending_language = language
            continue

        if success_match:
            if block_language is None or pending_language is None:
                raise ValueError(f"line {line_number}: Success record has no Task record")
            outcomes.append(success_match.group("outcome") == "True")
            pending_language = None
            continue

        if rate_match:
            if block_language is None or pending_language is not None or not outcomes:
                raise ValueError(f"line {line_number}: incomplete task block before success rate")
            logged_rate = float(rate_match.group("rate"))
            if not 0.0 <= logged_rate <= 1.0:
                raise ValueError(f"line {line_number}: invalid task success rate {logged_rate}")
            outcome_rate = sum(outcomes) / len(outcomes)
            if not math.isclose(logged_rate, outcome_rate, abs_tol=_RATE_TOLERANCE):
                raise ValueError(
                    f"line {line_number}: logged rate {logged_rate} does not match outcomes "
                    f"({outcome_rate})"
                )
            evaluations.append(
                TaskEvaluation(suite, len(evaluations), block_language, tuple(outcomes))
            )
            block_language = None
            outcomes = []

    if block_language is not None or pending_language is not None:
        raise ValueError("worker log ends with an incomplete task block")
    if not evaluations:
        raise ValueError("worker log contains no completed task blocks")
    return evaluations


def select_extreme_tasks(
    rows: Sequence[TaskEvaluation], count: int = 2
) -> dict[str, list[TaskEvaluation]]:
    """Return disjoint best and worst tasks for one suite.

    Best tasks sort by descending rate then ascending task id. Worst tasks sort
    by ascending rate then ascending task id after removing the selected best
    tasks. This best-first precedence keeps all four selections distinct even
    when a suite is saturated at the same success rate.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    if not rows:
        raise ValueError("cannot select tasks from no evaluations")
    suites = {row.suite for row in rows}
    if len(suites) != 1:
        raise ValueError("select_extreme_tasks accepts evaluations from one suite")
    task_ids = [row.task_id for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task ids must be unique within a suite")
    if any(not row.language or not row.outcomes for row in rows):
        raise ValueError("every task needs one language and at least one outcome")
    if len(rows) < 2 * count:
        raise ValueError("not enough distinct tasks for non-overlapping best and worst groups")

    best = sorted(rows, key=lambda row: (-row.success_rate, row.task_id))[:count]
    best_task_ids = {row.task_id for row in best}
    remaining = [row for row in rows if row.task_id not in best_task_ids]
    worst = sorted(remaining, key=lambda row: (row.success_rate, row.task_id))[:count]
    return {"best": best, "worst": worst}


def build_audit_cases(
    rows: Iterable[TaskEvaluation],
    seeds: Sequence[int] = _DEFAULT_SEEDS,
    extremes: int = 2,
) -> list[AuditCase]:
    """Build fixed-state, five-seed audit cases for every represented suite."""

    if not seeds:
        raise ValueError("at least one audit seed is required")
    by_suite: dict[str, list[TaskEvaluation]] = defaultdict(list)
    for row in rows:
        by_suite[row.suite].append(row)
    if not by_suite:
        raise ValueError("cannot build audit cases from no evaluations")

    cases: list[AuditCase] = []
    for suite in sorted(by_suite):
        selected = select_extreme_tasks(by_suite[suite], count=extremes)
        for rank_group in ("best", "worst"):
            for row in selected[rank_group]:
                representative_index = _representative_index(row, rank_group)
                original_success = row.outcomes[representative_index]
                cases.extend(
                    AuditCase(
                        suite=row.suite,
                        task_id=row.task_id,
                        language=row.language,
                        rank_group=rank_group,
                        initial_state_index=representative_index,
                        seed=seed,
                        original_success=original_success,
                    )
                    for seed in seeds
                )
    return cases


def validate_audit_cases(cases: Sequence[AuditCase]) -> None:
    """Require the standard four-suite, 16-task, 80-case audit shape."""

    if len(cases) != 80:
        raise ValueError(f"expected 80 audit cases, got {len(cases)}")
    by_suite: dict[str, list[AuditCase]] = defaultdict(list)
    for case in cases:
        by_suite[case.suite].append(case)
    if len(by_suite) != 4:
        raise ValueError(f"expected four suites, got {len(by_suite)}")

    selected_tasks = 0
    expected_seeds = set(_DEFAULT_SEEDS)
    for suite, suite_cases in by_suite.items():
        if len(suite_cases) != 20:
            raise ValueError(f"suite {suite!r} must contain 20 cases")
        by_task: dict[tuple[int, str], list[AuditCase]] = defaultdict(list)
        for case in suite_cases:
            if case.rank_group not in {"best", "worst"}:
                raise ValueError(f"suite {suite!r} has unknown rank group {case.rank_group!r}")
            by_task[(case.task_id, case.rank_group)].append(case)
        if len(by_task) != 4:
            raise ValueError(f"suite {suite!r} must contain four selected tasks")
        task_ids = {task_id for task_id, _ in by_task}
        if len(task_ids) != 4:
            raise ValueError(f"suite {suite!r} selected a task in multiple rank groups")
        if {group for _, group in by_task} != {"best", "worst"}:
            raise ValueError(f"suite {suite!r} must contain best and worst tasks")
        if sum(group == "best" for _, group in by_task) != 2:
            raise ValueError(f"suite {suite!r} must contain two best tasks")
        if sum(group == "worst" for _, group in by_task) != 2:
            raise ValueError(f"suite {suite!r} must contain two worst tasks")
        for task_cases in by_task.values():
            if len(task_cases) != 5 or {case.seed for case in task_cases} != expected_seeds:
                raise ValueError("every selected task must contain seeds 7, 8, 9, 10, and 11")
            if len({case.initial_state_index for case in task_cases}) != 1:
                raise ValueError("all seeds for one task must share an initial state index")
            if len({case.language for case in task_cases}) != 1:
                raise ValueError("all seeds for one task must share one language")
            if len({case.original_success for case in task_cases}) != 1:
                raise ValueError("all seeds for one task must share one original outcome")
        selected_tasks += len(by_task)
    if selected_tasks != 16:
        raise ValueError(f"expected 16 selected tasks, got {selected_tasks}")


def _representative_index(row: TaskEvaluation, rank_group: str) -> int:
    desired_outcome = rank_group == "best"
    for index, outcome in enumerate(row.outcomes):
        if outcome == desired_outcome:
            return index
    return 0
