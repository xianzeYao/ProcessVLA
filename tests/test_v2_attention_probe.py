from __future__ import annotations

import numpy as np
import pytest
import torch

from examples.simBenchmarks.CoT.attention_probe.v2_attention_probe import (
    aggregate_attention_records,
    attention_to_patch_map,
    build_condition_groups,
    contiguous_token_runs,
    infer_patch_grid,
    modality_attention_mass,
    selected_attention_probabilities,
)


def test_contiguous_token_runs_keeps_two_camera_spans_separate():
    token_ids = torch.tensor([9, 7, 7, 7, 2, 7, 7, 4])

    runs = contiguous_token_runs(token_ids, token_id=7)

    assert [run.tolist() for run in runs] == [[1, 2, 3], [5, 6]]


@pytest.mark.parametrize(
    ("token_count", "expected"),
    [(256, (16, 16)), (240, (15, 16)), (6, (2, 3))],
)
def test_infer_patch_grid_prefers_near_square_factorization(token_count, expected):
    assert infer_patch_grid(token_count) == expected


def test_selected_attention_probabilities_matches_masked_softmax():
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    allowed = torch.tensor([[[[True, False, True], [True, True, False]]]])

    actual = selected_attention_probabilities(
        query,
        key,
        query_indices=torch.tensor([1]),
        attention_mask=allowed,
    )

    expected = torch.softmax(torch.tensor([0.0, 1.0]) / np.sqrt(2.0), dim=0)
    assert actual.shape == (1, 1, 1, 3)
    torch.testing.assert_close(actual[0, 0, 0, :2], expected)
    assert actual[0, 0, 0, 2].item() == 0.0


def test_modality_mass_is_normalized_over_valid_grouped_keys():
    attention = torch.tensor([[0.10, 0.20, 0.30, 0.15, 0.25]])
    groups = {
        "agentview": np.asarray([0, 1]),
        "wrist": np.asarray([2]),
        "geometry": np.asarray([3, 4]),
    }

    mass = modality_attention_mass(attention, groups)

    assert mass == pytest.approx(
        {"agentview": 0.30, "wrist": 0.30, "geometry": 0.40}
    )


def test_aggregate_attention_records_preserves_layer_and_step_axes():
    records = [
        {"layer": 0, "call": 0, "attention": torch.full((2, 3), 1.0)},
        {"layer": 2, "call": 0, "attention": torch.full((2, 3), 3.0)},
        {"layer": 0, "call": 1, "attention": torch.full((2, 3), 5.0)},
    ]

    summary = aggregate_attention_records(records)

    assert summary["mean"].shape == (2, 3)
    torch.testing.assert_close(summary["mean"], torch.full((2, 3), 3.0))
    assert summary["layers"].tolist() == [0, 2, 0]
    assert summary["calls"].tolist() == [0, 0, 1]


def test_build_condition_groups_tracks_two_views_and_appended_geometry():
    input_ids = torch.tensor([9, 7, 7, 2, 7, 7, 7, 3])
    native_mask = torch.tensor([True, True, True, True, True, True, True, False])

    groups = build_condition_groups(
        input_ids,
        image_token_id=7,
        native_attention_mask=native_mask,
        depth_query_count=2,
        uvd_token_count=3,
        include_depth=True,
    )

    assert groups["agentview"].tolist() == [1, 2]
    assert groups["wrist"].tolist() == [4, 5, 6]
    assert groups["language"].tolist() == [0, 3]
    assert groups["depth_current"].tolist() == [8, 9]
    assert groups["depth_future"].tolist() == [10, 11]
    assert groups["uvd"].tolist() == [12, 13, 14]


def test_attention_to_patch_map_averages_queries_and_normalizes():
    attention = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 0.0], [0.0, 3.0, 2.0, 1.0, 0.0]]
    )

    patch_map = attention_to_patch_map(
        attention,
        key_indices=np.asarray([1, 2, 3, 4]),
        patch_grid=(2, 2),
    )

    assert patch_map.shape == (2, 2)
    np.testing.assert_allclose(patch_map, [[1.0, 1.0], [1.0, 0.0]])
