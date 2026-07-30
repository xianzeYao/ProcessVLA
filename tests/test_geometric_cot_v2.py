import numpy as np
import pytest
import torch

from starVLA.model.modules.geometric_cot_v2 import (
    GeometryTokenEmbedding,
    GeometryTokenLayout,
    append_geometry_slots,
    build_geometry_full_attention_mask,
    pack_uvd_targets_time_major,
)


def test_single_hand_mask_keeps_native_causal_and_depth_groups_full():
    layout = GeometryTokenLayout(depth_query_count=2, uvd_points_per_hand=3, hand_count=1)
    native_mask = torch.ones(1, 3, dtype=torch.bool)
    appended_mask = torch.cat([native_mask, torch.ones(1, layout.geometry_token_count, dtype=torch.bool)], dim=1)

    allowed = build_geometry_full_attention_mask(appended_mask, layout)[0, 0]

    # Native V/L remains causal and cannot read appended geometry.
    assert allowed[2].tolist() == [True, True, True, False, False, False, False, False, False, False]
    # Current-depth slots [3,4] read V/L and their complete group, but not future/UVD.
    assert allowed[3].tolist() == [True, True, True, True, True, False, False, False, False, False]
    assert allowed[4].tolist() == [True, True, True, True, True, False, False, False, False, False]
    # Future-depth slots [5,6] read V/L, current depth, and their complete group.
    assert allowed[5].tolist() == [True, True, True, True, True, True, True, False, False, False]
    assert allowed[6].tolist() == [True, True, True, True, True, True, True, False, False, False]
    # Single-hand UVD remains temporally causal.
    assert allowed[7].tolist() == [True, True, True, True, True, True, True, True, False, False]
    assert allowed[8].tolist() == [True, True, True, True, True, True, True, True, True, False]


def test_dual_hand_mask_is_time_major_block_causal():
    layout = GeometryTokenLayout(depth_query_count=1, uvd_points_per_hand=2, hand_count=2)
    appended_mask = torch.ones(1, 2 + layout.geometry_token_count, dtype=torch.bool)

    allowed = build_geometry_full_attention_mask(appended_mask, layout)[0, 0]
    # Sequence: V0,V1,Dc,Df,L0,R0,L1,R1.
    assert layout.sequence_slices(native_token_count=2).uvd == slice(4, 8)
    assert allowed[4, 5]  # L0 reads R0 in a full-attention layer.
    assert allowed[5, 4]  # R0 reads L0.
    assert not allowed[4, 6]  # Time zero cannot read time one.
    assert not allowed[5, 7]
    assert allowed[6, 4] and allowed[6, 5] and allowed[6, 7]
    assert allowed[7, 4] and allowed[7, 5] and allowed[7, 6]


def test_full_attention_mask_never_reads_padding_or_invalid_uvd_keys():
    layout = GeometryTokenLayout(depth_query_count=1, uvd_points_per_hand=2, hand_count=1)
    # Native key 0 is left padding and the final UVD slot is invalid.
    appended_mask = torch.tensor([[0, 1, 1, 1, 1, 0]], dtype=torch.bool)

    allowed = build_geometry_full_attention_mask(appended_mask, layout)[0, 0]

    assert not allowed[:, 0].any()
    assert not allowed[:, -1].any()


def test_append_geometry_slots_keeps_every_fixed_query_active():
    layout = GeometryTokenLayout(depth_query_count=1, uvd_points_per_hand=2, hand_count=1)
    qwen_inputs = {
        "input_ids": torch.tensor([[9, 10, 11]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
        "pixel_values": torch.randn(4, 8),
    }
    appended = append_geometry_slots(
        qwen_inputs,
        layout,
        placeholder_token_id=17,
    )

    assert appended["input_ids"].tolist() == [[9, 10, 11, 17, 17, 17, 17]]
    assert appended["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 1]]
    assert appended["mm_token_type_ids"].tolist() == [[0, 1, 0, 0, 0, 0, 0]]
    assert appended["pixel_values"] is qwen_inputs["pixel_values"]
    assert qwen_inputs["input_ids"].shape == (1, 3)


def test_pack_uvd_targets_uses_time_major_order_and_fixed_slots():
    layout = GeometryTokenLayout(depth_query_count=1, uvd_points_per_hand=3, hand_count=2)
    examples = [
        {
            "uvd": np.asarray(
                [
                    [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
                    [[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]],
                ],
                dtype=np.float32,
            ),
            "uvd_valid_mask": np.asarray([[True, True], [True, False]]),
            "uvd_time": np.asarray([0.0, 0.5], dtype=np.float32),
        }
    ]

    packed = pack_uvd_targets_time_major(examples, layout, device=torch.device("cpu"))

    assert packed.target[0].tolist() == [
        [10.0, 11.0, 12.0],
        [20.0, 21.0, 22.0],
        [30.0, 31.0, 32.0],
        [40.0, 41.0, 42.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert packed.valid[0].tolist() == [True, True, True, False, False, False]
    assert packed.times[0].tolist() == [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
    assert packed.hand_ids[0].tolist() == [0, 1, 0, 1, 0, 1]


def test_geometry_token_embedding_expands_shared_seed_in_time_major_order():
    layout = GeometryTokenLayout(depth_query_count=2, uvd_points_per_hand=3, hand_count=2)
    module = GeometryTokenEmbedding(hidden_dim=8, layout=layout)
    times = torch.tensor([[0.0, 0.0, 0.5, 0.5, 1.0, 1.0]])
    hand_ids = torch.tensor([[0, 1, 0, 1, 0, 1]])

    output = module(batch_size=1, uvd_times=times, uvd_hand_ids=hand_ids)

    assert output.shape == (1, layout.geometry_token_count, 8)
    assert not torch.equal(
        output[:, layout.geometry_uvd_slice.start],
        output[:, layout.geometry_uvd_slice.start + 1],
    )


def test_dual_hand_layout_rejects_collapsed_single_hand_target():
    layout = GeometryTokenLayout(depth_query_count=1, uvd_points_per_hand=3, hand_count=2)
    examples = [
        {
            "uvd": np.zeros((3, 3), dtype=np.float32),
            "uvd_valid_mask": np.ones((3,), dtype=np.bool_),
        }
    ]

    with pytest.raises(ValueError, match="exactly 2 hands"):
        pack_uvd_targets_time_major(examples, layout, device=torch.device("cpu"))
