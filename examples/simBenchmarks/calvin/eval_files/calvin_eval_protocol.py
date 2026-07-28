"""Pure helpers for reproducible, split-aware CALVIN evaluation."""

from __future__ import annotations

from typing import Any, Iterable


_SPLIT_ALIASES = {
    "ABCD_D": "ABCD_D",
    "ABCD->D": "ABCD_D",
    "ABC_D": "ABC_D",
    "ABC->D": "ABC_D",
}


def normalize_split(split: str) -> str:
    """Normalize paper notation and repository notation for a split."""
    try:
        return _SPLIT_ALIASES[split.strip().upper()]
    except KeyError as exc:
        valid = ", ".join(sorted(_SPLIT_ALIASES))
        raise ValueError(f"Unsupported CALVIN split {split!r}; expected one of: {valid}") from exc


def build_worker_ranges(total_sequences: int, num_workers: int) -> list[tuple[int, int]]:
    """Return contiguous, non-overlapping ranges covering every sequence."""
    if total_sequences < 0:
        raise ValueError("total_sequences must be non-negative")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")

    base, remainder = divmod(total_sequences, num_workers)
    ranges = []
    start = 0
    for worker_id in range(num_workers):
        size = base + (1 if worker_id < remainder else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def summarize_chain_results(results: Iterable[int]) -> dict[str, Any]:
    """Compute CALVIN chain metrics from per-sequence task counts."""
    values = [int(value) for value in results]
    num_sequences = len(values)
    success_counts = [sum(value >= length for value in values) for length in range(1, 6)]
    denominator = max(num_sequences, 1)
    return {
        "num_sequences": num_sequences,
        "results": values,
        "success_counts": success_counts,
        "success_rates": [count / denominator for count in success_counts],
        "average_tasks_completed": sum(values) / denominator,
    }


def aggregate_worker_payloads(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge workers by sequence order and recompute metrics globally."""
    ordered = sorted(payloads, key=lambda payload: int(payload["sequence_start"]))
    expected_start = 0
    all_results: list[int] = []
    for payload in ordered:
        start = int(payload["sequence_start"])
        end = int(payload["sequence_end"])
        results = [int(value) for value in payload["results"]]
        if start != expected_start:
            raise ValueError(f"worker ranges are not contiguous: expected {expected_start}, got {start}")
        if end - start != len(results):
            raise ValueError(f"worker range [{start}, {end}) has {len(results)} results")
        all_results.extend(results)
        expected_start = end
    return summarize_chain_results(all_results)
