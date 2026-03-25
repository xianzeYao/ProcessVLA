import argparse
import json
import pathlib
from collections import defaultdict


def summarize(results_path: pathlib.Path) -> None:
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    rows = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print(f"No rows found in {results_path}")
        return

    total = len(rows)
    success = sum(1 for row in rows if row.get("success", False))
    fail = total - success
    crash = sum(1 for row in rows if row.get("returncode") not in (None, 0))
    timeout = sum(1 for row in rows if row.get("error") and "timeout" in str(row.get("error")).lower())

    print(f"results_file: {results_path}")
    print(f"total_episodes: {total}")
    print(f"successes: {success}")
    print(f"failures: {fail}")
    print(f"success_rate: {100.0 * success / total:.2f}%")
    print(f"crash_like_failures: {crash}")
    print(f"timeouts: {timeout}")

    per_task = defaultdict(list)
    for row in rows:
        per_task[int(row["task_id"])].append(row)

    print("\nper_task:")
    for task_id in sorted(per_task):
        task_rows = per_task[task_id]
        task_total = len(task_rows)
        task_success = sum(1 for row in task_rows if row.get("success", False))
        task_crash = sum(1 for row in task_rows if row.get("returncode") not in (None, 0))
        task_timeout = sum(1 for row in task_rows if row.get("error") and "timeout" in str(row.get("error")).lower())
        print(
            f"  task_id={task_id}: "
            f"{task_success}/{task_total} = {100.0 * task_success / task_total:.2f}% "
            f"(crash_like={task_crash}, timeout={task_timeout})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl", type=pathlib.Path)
    args = parser.parse_args()
    summarize(args.results_jsonl)


if __name__ == "__main__":
    main()
