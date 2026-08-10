import unittest

import torch

from examples.simBenchmarks.CoT.geometry_probe.paired_probe import (
    strip_action_model_for_geometry,
)


class _Framework(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.action_model = torch.nn.Linear(16, 16)

    def predict_geometry(self, examples):
        return examples


class GeometryOnlyCheckpointTest(unittest.TestCase):
    def test_removes_only_unused_action_model_and_reports_freed_bytes(self):
        framework = _Framework()
        expected = sum(
            parameter.numel() * parameter.element_size()
            for parameter in framework.action_model.parameters()
        )

        removed = strip_action_model_for_geometry(framework)

        self.assertEqual(removed, expected)
        self.assertFalse(hasattr(framework, "action_model"))
        self.assertTrue(hasattr(framework, "backbone"))
        self.assertTrue(callable(framework.predict_geometry))


if __name__ == "__main__":
    unittest.main()
