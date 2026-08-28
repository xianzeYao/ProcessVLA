
import unittest

import torch

from starVLA.model.modules.cot_losses import (
    masked_smooth_l1_loss,
    uvd_adjacent_relative_loss,
    uvd_regression_loss,
    uvd_triangle_shape_loss,
)


class CotLossTest(unittest.TestCase):
    def test_masked_depth_loss_ignores_invalid_pixels(self):
        pred = torch.tensor([[[[100.0, 1.0]]]])
        target = torch.tensor([[[[0.0, 2.0]]]])
        valid = torch.tensor([[[[False, True]]]])

        loss = masked_smooth_l1_loss(pred, target, valid)

        self.assertAlmostEqual(float(loss), 0.5, places=6)

    def test_masked_depth_loss_ignores_nan_at_invalid_pixels(self):
        pred = torch.tensor([[[[1.0, 100.0]]]], requires_grad=True)
        target = torch.tensor([[[[2.0, float("nan")]]]])
        valid = torch.tensor([[[[True, False]]]])

        loss = masked_smooth_l1_loss(pred, target, valid)
        loss.backward()

        self.assertAlmostEqual(float(loss), 0.5, places=6)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(float(pred.grad[0, 0, 0, 1]), 0.0)

    def test_uvd_loss_ignores_invalid_trace_points(self):
        pred = torch.tensor([[[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]])
        target = torch.zeros_like(pred)
        valid = torch.tensor([[True, False]])

        loss = uvd_regression_loss(pred, target, valid)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_uvd_loss_ignores_nan_at_invalid_trace_points(self):
        pred = torch.zeros(1, 2, 3, requires_grad=True)
        target = torch.tensor(
            [[[0.0, 0.0, 0.0], [float("nan"), float("nan"), float("nan")]]]
        )
        valid = torch.tensor([[True, False]])

        loss = uvd_regression_loss(pred, target, valid)
        loss.backward()

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(float(pred.grad[0, 1].abs().sum()), 0.0)

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

    def test_adjacent_relative_loss_ignores_nan_at_invalid_endpoints(self):
        pred = torch.zeros(1, 3, 3, requires_grad=True)
        target = torch.zeros_like(pred).detach()
        target[0, 1] = float("nan")
        valid = torch.tensor([[True, False, True]])

        loss = uvd_adjacent_relative_loss(
            pred,
            target,
            valid,
            hand_count=1,
        )
        loss.backward()

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_triangle_shape_loss_ignores_nan_in_incomplete_triangles(self):
        pred = torch.zeros(1, 6, 3, requires_grad=True)
        target = torch.zeros_like(pred).detach()
        target[0, 4] = float("nan")
        valid = torch.ones(1, 6, dtype=torch.bool)
        valid[0, 4] = False

        loss = uvd_triangle_shape_loss(pred, target, valid)
        loss.backward()

        self.assertEqual(float(loss), 0.0)
        self.assertTrue(torch.isfinite(pred.grad).all())

if __name__ == "__main__":
    unittest.main()
