import numpy as np
import pytest

from starVLA.dataloader.gr00t_lerobot.cot_geometry_v4 import (
    V4HorizonSpec,
    sample_forward_uvd_indices,
)


def test_forward_horizons_have_equal_point_counts_and_double_coarse_span():
    spec = V4HorizonSpec(
        action_horizon=8,
        local_uvd_num_points=8,
        coarse_uvd_num_points=8,
        coarse_uvd_stride=2,
        terminal_repeat=True,
    )

    np.testing.assert_array_equal(spec.local_indices(3, 30), np.arange(4, 12))
    np.testing.assert_array_equal(spec.coarse_indices(3, 30), np.arange(5, 20, 2))


def test_forward_horizons_repeat_terminal_without_padding_slots():
    spec = V4HorizonSpec(8, 8, 8, 2, True)

    np.testing.assert_array_equal(spec.local_indices(6, 7), np.full(8, 7))
    np.testing.assert_array_equal(spec.coarse_indices(6, 7), np.full(8, 7))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"local_uvd_num_points": 7}, "local.*action_horizon"),
        ({"coarse_uvd_num_points": 9}, "coarse.*action_horizon"),
        ({"coarse_uvd_stride": 1}, "stride.*2"),
        ({"terminal_repeat": False}, "terminal_repeat.*true"),
    ],
)
def test_v4_horizon_contract_fails_fast(kwargs, message):
    values = {
        "action_horizon": 8,
        "local_uvd_num_points": 8,
        "coarse_uvd_num_points": 8,
        "coarse_uvd_stride": 2,
        "terminal_repeat": True,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        V4HorizonSpec(**values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 2, "terminal": 1, "num_points": 2, "stride": 1},
        {"start": 0, "terminal": 1, "num_points": 0, "stride": 1},
        {"start": 0, "terminal": 1, "num_points": 2, "stride": 0},
    ],
)
def test_forward_sampler_rejects_invalid_ranges(kwargs):
    with pytest.raises(ValueError):
        sample_forward_uvd_indices(**kwargs)
