from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from starVLA.dataloader.gr00t_lerobot.gripper_triangle import (
    gripper_triangle_path,
    load_gripper_triangle_sidecar,
)


class GripperTriangleGenerationIntegrationTest(unittest.TestCase):
    def test_generation_replays_source_states_writes_report_and_resumes(self) -> None:
        from examples.modelExtensions.CoT.scripts.build_libero_gripper_triangle_sidecars import (
            run,
        )

        class FakeModel:
            body_ids = {
                "gripper0_finger_joint1_tip": 2,
                "gripper0_finger_joint2_tip": 0,
                "gripper0_right_gripper": 1,
            }

            def body_name2id(self, name: str) -> int:
                return self.body_ids[name]

        class FakeData:
            body_xpos = np.zeros((3, 3), dtype=np.float64)

        class FakeSim:
            model = FakeModel()
            data = FakeData()

        class FakeEnv:
            sim = FakeSim()

            def regenerate_obs_from_state(self, state: np.ndarray) -> None:
                offset = float(state[0])
                self.sim.data.body_xpos = np.asarray(
                    [
                        [0.02 + offset, 0.0, 1.0],
                        [offset, 0.05, 1.0],
                        [-0.02 + offset, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )

            def close(self) -> None:
                pass

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            dataset_root = base / "datasets"
            suite = dataset_root / "libero_goal_no_noops_1.0.0_lerobot"
            (suite / "meta").mkdir(parents=True)
            (suite / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "total_episodes": 1,
                        "total_frames": 2,
                        "features": {
                            "observation.images.image": {
                                "dtype": "video",
                                "shape": [32, 64, 3],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            source_hdf5 = base / "source.hdf5"
            source_hdf5.touch()

            class FakeDataGroup:
                attrs: dict[str, object] = {}

            class FakeH5:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def __getitem__(self, key: str):
                    if key == "data":
                        return FakeDataGroup()
                    if key == "data/demo_0/states":
                        return np.asarray([[0.0], [0.01]], dtype=np.float64)
                    raise KeyError(key)

            def fake_h5_open(_path: Path, _mode: str):
                return FakeH5()
            camera_rel = Path("camera/chunk-000/episode_000000.npz")
            camera_path = suite / camera_rel
            camera_path.parent.mkdir(parents=True)
            k = np.asarray(
                [[100.0, 0.0, 32.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            np.savez_compressed(
                camera_path,
                agentview_K=np.repeat(k[None], 2, axis=0),
                agentview_T_world_camera=np.repeat(
                    np.eye(4, dtype=np.float32)[None], 2, axis=0
                ),
            )
            parquet = suite / "data" / "chunk-000" / "episode_000000.parquet"
            parquet.parent.mkdir(parents=True)
            pd.DataFrame(
                {
                    "source.hdf5_path": [str(source_hdf5)] * 2,
                    "source.hdf5_demo_id": ["demo_0"] * 2,
                    "source.hdf5_index": [0, 1],
                    "observation.camera.params_path": [str(camera_rel)] * 2,
                }
            ).to_parquet(parquet)
            report_path = base / "generation.json"

            result = run(
                [
                    "--dataset-root",
                    str(dataset_root),
                    "--suite",
                    "libero_goal",
                    "--report-path",
                    str(report_path),
                ],
                env_factory=lambda _attrs, _width, _height: FakeEnv(),
                h5_open=fake_h5_open,
            )

            sidecar = load_gripper_triangle_sidecar(
                gripper_triangle_path(suite, 0), frame_count=2, width=64, height=32
            )
            np.testing.assert_allclose(
                sidecar.world_xyz,
                [
                    [[-0.02, 0.0, 1.0], [0.02, 0.0, 1.0], [0.0, 0.05, 1.0]],
                    [[-0.01, 0.0, 1.0], [0.03, 0.0, 1.0], [0.01, 0.05, 1.0]],
                ],
                atol=1e-7,
            )
            np.testing.assert_allclose(
                sidecar.agentview_uvd_pixels[0],
                [[30.0, 16.0, 1.0], [34.0, 16.0, 1.0], [32.0, 21.0, 1.0]],
                atol=1e-6,
            )
            self.assertEqual(result["totals"]["written_episodes"], 1)
            self.assertEqual(result["totals"]["validated_episodes"], 1)
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), result)

            resumed = run(
                [
                    "--dataset-root",
                    str(dataset_root),
                    "--suite",
                    "libero_goal",
                    "--report-path",
                    str(report_path),
                ],
                env_factory=lambda *_args: self.fail(
                    "a valid existing sidecar must skip simulator construction"
                ),
                h5_open=lambda *_args: self.fail(
                    "a valid existing sidecar must skip HDF5 construction"
                ),
            )

        self.assertEqual(resumed["totals"]["skipped_episodes"], 1)
        self.assertEqual(resumed["totals"]["written_episodes"], 0)


if __name__ == "__main__":
    unittest.main()
