from __future__ import annotations

import torch

from starVLA.model.modules.cot_losses import (
    uvd_adjacent_relative_loss,
    uvd_triangle_shape_loss,
)


def triangle_tokens() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [-1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
                [-0.5, 0.0, 1.0],
                [0.5, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ]
    )


def test_temporal_loss_tracks_each_of_three_landmarks_independently() -> None:
    target = triangle_tokens()
    translated = target + torch.tensor([4.0, -2.0, 0.5])
    valid = torch.ones(1, 6, dtype=torch.bool)
    assert float(
        uvd_adjacent_relative_loss(translated, target, valid, hand_count=3)
    ) == 0.0

    moved_right_at_t1 = translated.clone()
    moved_right_at_t1[0, 4, 0] += 1.0
    assert float(
        uvd_adjacent_relative_loss(moved_right_at_t1, target, valid, hand_count=3)
    ) > 0.0


def test_shape_loss_is_translation_invariant_and_detects_triangle_changes() -> None:
    target = triangle_tokens()
    valid = torch.ones(1, 6, dtype=torch.bool)

    translated = target + torch.tensor([4.0, -2.0, 0.5])
    assert float(uvd_triangle_shape_loss(translated, target, valid)) == 0.0

    aperture_changed = target.clone()
    aperture_changed[0, 1, 0] += 1.0
    assert float(uvd_triangle_shape_loss(aperture_changed, target, valid)) > 0.0

    wrist_changed = target.clone()
    wrist_changed[0, 5, 1] += 1.0
    assert float(uvd_triangle_shape_loss(wrist_changed, target, valid)) > 0.0


def test_shape_loss_requires_complete_triangle_and_returns_differentiable_zero() -> None:
    target = triangle_tokens()
    pred = (target + 10.0).requires_grad_()
    valid = torch.ones(1, 6, dtype=torch.bool)
    valid[0, 2] = False
    valid[0, 3] = False

    loss = uvd_triangle_shape_loss(pred, target, valid)
    loss.backward()

    assert float(loss) == 0.0
    assert pred.grad is not None
    assert torch.equal(pred.grad, torch.zeros_like(pred))
