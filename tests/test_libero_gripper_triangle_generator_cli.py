from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.gripper_triangle import gripper_triangle_path


def valid_payload() -> dict[str, np.ndarray]:
    return {
        "world_xyz": np.asarray(
            [
                [[-0.02, 0.0, 1.0], [0.02, 0.0, 1.0], [0.0, 0.05, 1.0]],
                [[-0.01, 0.0, 2.0], [0.01, 0.0, 2.0], [0.0, 0.05, 2.0]],
            ],
            dtype=np.float32,
        ),
        "agentview_uvd_pixels": np.asarray(
            [
                [[30.0, 16.0, 1.0], [34.0, 16.0, 1.0], [32.0, 21.0, 1.0]],
                [[31.5, 16.0, 2.0], [32.5, 16.0, 2.0], [32.0, 18.5, 2.0]],
            ],
            dtype=np.float32,
        ),
        "agentview_projection_valid": np.ones((2, 3), dtype=np.bool_),
        "agentview_in_frame": np.ones((2, 3), dtype=np.bool_),
    }


class GripperTriangleGeneratorCliTest(unittest.TestCase):
    def test_episode_range_is_end_exclusive_then_limited(self) -> None:
        from examples.modelExtensions.CoT.scripts.build_libero_gripper_triangle_sidecars import (
            select_episode_ids,
        )

        self.assertEqual(
            select_episode_ids(10, episode_start=2, episode_end=7, max_episodes=3),
            [2, 3, 4],
        )
        self.assertEqual(
            select_episode_ids(4, episode_start=1, episode_end=None, max_episodes=None),
            [1, 2, 3],
        )

    def test_validate_only_reports_missing_before_metadata_then_finalizes(self) -> None:
        from examples.modelExtensions.CoT.scripts.build_libero_gripper_triangle_sidecars import (
            run,
            write_sidecar_atomic,
        )

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = base / "datasets"
            suite = dataset_root / "libero_goal_no_noops_1.0.0_lerobot"
            (suite / "meta").mkdir(parents=True)
            info_path = suite / "meta" / "info.json"
            original_info = {
                "total_episodes": 2,
                "total_frames": 4,
                "features": {
                    "observation.images.image": {
                        "dtype": "video",
                        "shape": [32, 64, 3],
                    }
                },
            }
            info_path.write_text(json.dumps(original_info), encoding="utf-8")
            for episode_id in range(2):
                parquet = suite / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
                parquet.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {"episode_index": [episode_id, episode_id], "frame_index": [0, 1]}
                ).to_parquet(parquet)
            write_sidecar_atomic(
                gripper_triangle_path(suite, 0),
                valid_payload(),
                frame_count=2,
                width=64,
                height=32,
                overwrite=False,
            )
            failed_report = base / "failed.json"

            with self.assertRaisesRegex(RuntimeError, "1 sidecar.*failed validation"):
                run(
                    [
                        "--dataset-root",
                        str(dataset_root),
                        "--suite",
                        "libero_goal",
                        "--validate-only",
                        "--report-path",
                        str(failed_report),
                    ]
                )

            failure = json.loads(failed_report.read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["totals"]["selected_episodes"], 2)
            self.assertEqual(failure["totals"]["selected_frames"], 4)
            self.assertEqual(failure["missing"], ["libero_goal/episode_000001"])
            self.assertEqual(json.loads(info_path.read_text(encoding="utf-8")), original_info)

            write_sidecar_atomic(
                gripper_triangle_path(suite, 1),
                valid_payload(),
                frame_count=2,
                width=64,
                height=32,
                overwrite=False,
            )
            success_report = base / "success.json"
            success = run(
                [
                    "--dataset-root",
                    str(dataset_root),
                    "--suite",
                    "libero_goal",
                    "--validate-only",
                    "--report-path",
                    str(success_report),
                ]
            )

        self.assertEqual(success["status"], "success")
        self.assertEqual(success["totals"]["validated_episodes"], 2)
        self.assertEqual(success["suites"]["libero_goal"]["episode_count"], 2)
        self.assertEqual(success["suites"]["libero_goal"]["frame_count"], 4)
        self.assertEqual(success["missing"], [])
        self.assertEqual(success["corrupt"], [])


if __name__ == "__main__":
    unittest.main()
