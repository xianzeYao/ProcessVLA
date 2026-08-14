from __future__ import annotations

import os
from pathlib import Path

import pytest

from examples.simBenchmarks.CoT.geometry_probe.libero_trace_audit_selection import (
    AuditCase,
    TaskEvaluation,
    build_audit_cases,
    parse_worker_log,
    select_extreme_tasks,
    validate_audit_cases,
)


def _worker_log(task_blocks: list[tuple[str, tuple[bool, ...]]]) -> str:
    lines: list[str] = ["08/14 [07:22:37]  INFO     | Task suite: libero_test"]
    for language, outcomes in task_blocks:
        for index, outcome in enumerate(outcomes, start=1):
            lines.extend(
                [
                    f"Task: {language}",
                    f"08/14 [07:23:{index:02d}]  INFO     | Success: {outcome}",
                ]
            )
        lines.append(
            "08/14 [07:30:00]  INFO     | Current task success rate: "
            f"{sum(outcomes) / len(outcomes):.2f}"
        )
    return "\n".join(lines) + "\n"


def _rows(suite: str = "libero_test") -> list[TaskEvaluation]:
    return [
        TaskEvaluation(suite, 0, "task zero", (False, True, True, True)),
        TaskEvaluation(suite, 1, "task one", (False, True, True, True)),
        TaskEvaluation(suite, 2, "task two", (False, True, False, True)),
        TaskEvaluation(suite, 3, "task three", (True, False, False, False)),
        TaskEvaluation(suite, 4, "task four", (False, False, False, False)),
    ]


def test_parse_worker_log_preserves_completed_block_order_and_outcomes(
    tmp_path: Path,
) -> None:
    first = tuple(index % 3 != 0 for index in range(50))
    second = tuple(index % 4 == 0 for index in range(50))
    path = tmp_path / "libero_test.worker.log"
    path.write_text(_worker_log([("first language", first), ("second language", second)]))

    rows = parse_worker_log(path, "libero_test")

    assert [(row.task_id, row.language) for row in rows] == [
        (0, "first language"),
        (1, "second language"),
    ]
    assert rows[0].outcomes == first
    assert rows[1].outcomes == second


@pytest.mark.parametrize(
    "contents",
    [
        "Task: unfinished\nSuccess: True\n",
        "Task: first language\nTask: second language\nSuccess: True\n"
        "Current task success rate: 1.0\n",
        "Success: True\nCurrent task success rate: 1.0\n",
        "Task: a language\nSuccess: True\nCurrent task success rate: 0.0\n",
    ],
)
def test_parse_worker_log_rejects_incomplete_or_inconsistent_blocks(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "broken.worker.log"
    path.write_text(contents)

    with pytest.raises(ValueError):
        parse_worker_log(path, "libero_test")


def test_select_extreme_tasks_breaks_rate_ties_by_task_id() -> None:
    selected = select_extreme_tasks(_rows(), count=2)

    assert [row.task_id for row in selected["best"]] == [0, 1]
    assert [row.task_id for row in selected["worst"]] == [4, 3]


def test_select_extreme_tasks_uses_best_first_for_saturated_suites() -> None:
    rows = [
        TaskEvaluation("libero_test", 0, "task zero", (False, False, True, True, True)),
        TaskEvaluation("libero_test", 1, "task one", (True, True, True, True, True)),
        TaskEvaluation("libero_test", 2, "task two", (True, True, True, True, True)),
        TaskEvaluation("libero_test", 3, "task three", (True, True, True, True, True)),
    ]

    selected = select_extreme_tasks(rows, count=2)

    assert [row.task_id for row in selected["best"]] == [1, 2]
    assert [row.task_id for row in selected["worst"]] == [0, 3]


def test_select_extreme_tasks_rejects_insufficient_distinct_tasks() -> None:
    with pytest.raises(ValueError):
        select_extreme_tasks(_rows()[:3])
    with pytest.raises(ValueError):
        select_extreme_tasks(_rows(), count=3)


def test_build_audit_cases_uses_one_representative_state_per_task() -> None:
    cases = build_audit_cases(_rows(), seeds=(7, 8, 9, 10, 11), extremes=2)

    best_zero = [case for case in cases if case.task_id == 0]
    worst_four = [case for case in cases if case.task_id == 4]
    assert len(best_zero) == 5
    assert {case.initial_state_index for case in best_zero} == {1}
    assert {case.original_success for case in best_zero} == {True}
    assert {case.seed for case in best_zero} == {7, 8, 9, 10, 11}
    assert {case.rank_group for case in best_zero} == {"best"}
    assert {case.initial_state_index for case in worst_four} == {0}
    assert {case.original_success for case in worst_four} == {False}
    assert {case.rank_group for case in worst_four} == {"worst"}


def test_validate_audit_cases_requires_four_suites_sixteen_tasks_and_eighty_cases() -> None:
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    cases: list[AuditCase] = []
    for suite in suites:
        cases.extend(build_audit_cases(_rows(suite)))

    validate_audit_cases(cases)

    with pytest.raises(ValueError):
        validate_audit_cases(cases[:-1])


def test_standard_worker_logs_contract_when_fixture_path_is_injected() -> None:
    log_dir = os.environ.get("LIBERO_TRACE_AUDIT_LOG_DIR")
    if log_dir is None:
        pytest.skip("set LIBERO_TRACE_AUDIT_LOG_DIR to run against standard worker logs")
    root = Path(log_dir)
    filenames = {
        "libero_spatial": "libero_spatial_gpu0.worker.log",
        "libero_object": "libero_object_gpu1.worker.log",
        "libero_goal": "libero_goal_gpu2.worker.log",
        "libero_10": "libero_10_gpu3.worker.log",
    }

    rows = [
        row
        for suite, filename in filenames.items()
        for row in parse_worker_log(root / filename, suite)
    ]
    cases = build_audit_cases(rows)

    assert {row.suite for row in rows} == set(filenames)
    assert all(len(row.outcomes) == 50 for row in rows)
    validate_audit_cases(cases)
