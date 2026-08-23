import numpy as np
import pytest
import torch

from starVLA.model.modules.geometric_cot_v4 import (
    GeometryTokenEmbedding,
    GeometryTokenLayout,
    append_geometry_slots,
    build_geometry_full_attention_mask,
    pack_coarse_uvd_targets_time_major,
    pack_local_uvd_targets_time_major,
)


def test_v4_layout_orders_coarse_before_local():
    layout = GeometryTokenLayout(
        depth_query_count=2,
        local_uvd_points_per_hand=3,
        coarse_uvd_points_per_hand=3,
        hand_count=2,
    )

    slices = layout.sequence_slices(native_token_count=4)

    assert layout.geometry_token_count == 16
    assert slices.depth_current == slice(4, 6)
    assert slices.depth_future == slice(6, 8)
    assert slices.uvd_coarse == slice(8, 14)
    assert slices.uvd_local == slice(14, 20)


def test_local_reads_all_coarse_but_coarse_cannot_read_local():
    layout = GeometryTokenLayout(1, 2, 2, 1)
    native_count = 2
    attention = torch.ones(
        1,
        native_count + layout.geometry_token_count,
        dtype=torch.bool,
    )

    mask = build_geometry_full_attention_mask(attention, layout)[0, 0]
    slices = layout.sequence_slices(native_count)

    assert mask[slices.uvd_local.start, slices.uvd_coarse].all()
    assert not mask[
        slices.uvd_coarse.stop - 1,
        slices.uvd_local.start,
    ]


def test_scales_use_independent_per_slot_queries_and_time_mlps():
    layout = GeometryTokenLayout(1, 3, 3, 1)
    module = GeometryTokenEmbedding(hidden_dim=8, layout=layout)

    assert module.coarse_queries.shape == (1, 3, 8)
    assert module.local_queries.shape == (1, 3, 8)
    assert module.coarse_queries is not module.local_queries
    assert module.coarse_time_embedding is not module.local_time_embedding
    assert not torch.equal(module.coarse_queries[:, 0], module.coarse_queries[:, 1])
    assert not torch.equal(module.local_queries[:, 0], module.local_queries[:, 1])

    output = module(batch_size=2)
    assert output.shape == (2, layout.geometry_token_count, 8)


def test_append_geometry_slots_uses_v4_geometry_count():
    layout = GeometryTokenLayout(2, 3, 3, 1)
    inputs = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "mm_token_type_ids": torch.ones(2, 4, dtype=torch.long),
    }

    output = append_geometry_slots(inputs, layout, placeholder_token_id=0)

    assert output["input_ids"].shape == (2, 4 + layout.geometry_token_count)
    assert output["attention_mask"][:, -layout.geometry_token_count :].all()
    assert not output["mm_token_type_ids"][
        :, -layout.geometry_token_count :
    ].any()


def make_examples(hand_count):
    local = np.arange(3 * hand_count * 3, dtype=np.float32).reshape(
        3,
        hand_count,
        3,
    )
    coarse = local + 100.0
    if hand_count == 1:
        local = local[:, 0]
        coarse = coarse[:, 0]
        valid = np.asarray([True, False, True])
    else:
        valid = np.asarray(
            [[True, False], [True, True], [False, True]],
            dtype=np.bool_,
        )
    return [
        {
            "uvd": local,
            "uvd_valid_mask": valid,
            "uvd_time": np.asarray([0.25, 0.5, 1.0], dtype=np.float32),
            "uvd_coarse": coarse,
            "uvd_coarse_valid_mask": valid,
            "uvd_coarse_time": np.asarray(
                [0.25, 0.5, 1.0],
                dtype=np.float32,
            ),
        }
    ]


@pytest.mark.parametrize("hand_count", [1, 2])
def test_named_packers_are_strict_and_time_major(hand_count):
    layout = GeometryTokenLayout(1, 3, 3, hand_count)
    examples = make_examples(hand_count)

    local = pack_local_uvd_targets_time_major(
        examples,
        layout,
        device=torch.device("cpu"),
    )
    coarse = pack_coarse_uvd_targets_time_major(
        examples,
        layout,
        device=torch.device("cpu"),
    )

    assert local.target.shape == (1, 3 * hand_count, 3)
    assert coarse.target.shape == (1, 3 * hand_count, 3)
    torch.testing.assert_close(coarse.target, local.target + 100.0)
    torch.testing.assert_close(
        local.times[0],
        torch.tensor([0.25, 0.5, 1.0]).repeat_interleave(hand_count),
    )
    torch.testing.assert_close(
        local.hand_ids[0],
        torch.arange(hand_count).repeat(3),
    )


def test_named_packer_rejects_short_targets_instead_of_padding():
    layout = GeometryTokenLayout(1, 3, 3, 1)
    examples = make_examples(1)
    examples[0]["uvd"] = examples[0]["uvd"][:2]
    examples[0]["uvd_valid_mask"] = examples[0]["uvd_valid_mask"][:2]
    examples[0]["uvd_time"] = examples[0]["uvd_time"][:2]

    with pytest.raises(ValueError, match="exactly 3 time points"):
        pack_local_uvd_targets_time_major(
            examples,
            layout,
            device=torch.device("cpu"),
        )
