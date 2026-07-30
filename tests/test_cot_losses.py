
import unittest

import torch

from starVLA.model.modules.cot_losses import (
    masked_smooth_l1_loss,
    uvd_adjacent_relative_loss,
    uvd_regression_loss,
)


class CotLossTest(unittest.TestCase):
    def test_masked_depth_loss_ignores_invalid_pixels(self):
        pred = torch.tensor([[[[100.0, 1.0]]]])
        target = torch.tensor([[[[0.0, 2.0]]]])
        valid = torch.tensor([[[[False, True]]]])

        loss = masked_smooth_l1_loss(pred, target, valid)

        self.assertAlmostEqual(float(loss), 0.5, places=6)

    def test_uvd_loss_ignores_invalid_trace_points(self):
        pred = torch.tensor([[[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]])
        target = torch.zeros_like(pred)
        valid = torch.tensor([[True, False]])

        loss = uvd_regression_loss(pred, target, valid)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_adjacent_relative_loss_uses_same_hand_time_deltas(self):
        target = torch.tensor(
            [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [1.0, 0.0, 0.0], [11.0, 0.0, 0.0]]]
        )
        pred = torch.tensor(
            [[[5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [6.0, 0.0, 0.0], [11.0, 0.0, 0.0]]]
        )
        valid = torch.ones(1, 4, dtype=torch.bool)

        loss = uvd_adjacent_relative_loss(pred, target, valid, hand_count=2)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_adjacent_relative_loss_matches_hand_computed_smooth_l1(self):
        target = torch.zeros(1, 3, 3)
        pred = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
        )
        valid = torch.ones(1, 3, dtype=torch.bool)

        loss = uvd_adjacent_relative_loss(pred, target, valid, hand_count=1)

        self.assertAlmostEqual(float(loss), 1.0 / 3.0, places=6)

    def test_adjacent_relative_loss_requires_both_segment_endpoints_valid(self):
        target = torch.zeros(1, 3, 3)
        pred = torch.tensor(
            [[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        )
        valid = torch.tensor([[True, False, True]])

        loss = uvd_adjacent_relative_loss(pred, target, valid, hand_count=1)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

if __name__ == "__main__":
    unittest.main()
