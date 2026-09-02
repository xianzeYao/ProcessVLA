import numpy as np
import pytest
import torch

from starVLA.model.modules.geometric_cot_v2 import (
    GeometryTokenEmbedding,
    GeometryTokenLayout,
    SharedDepthAttentionPool,
    append_geometry_slots,
    build_depth_summary_interventions,
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


@pytest.mark.parametrize(
    ("enable_current_depth", "enable_future_depth", "expected_current", "expected_future"),
    [
        (False, True, slice(2, 2), slice(2, 4)),
        (True, False, slice(2, 4), slice(4, 4)),
    ],
)
def test_layout_omits_the_disabled_depth_branch_from_the_qwen_sequence(
    enable_current_depth,
    enable_future_depth,
    expected_current,
    expected_future,
):
    layout = GeometryTokenLayout(
        depth_query_count=2,
        uvd_points_per_hand=2,
        hand_count=1,
        enable_current_depth=enable_current_depth,
        enable_future_depth=enable_future_depth,
    )

    slices = layout.sequence_slices(native_token_count=2)
    embeddings = GeometryTokenEmbedding(hidden_dim=4, layout=layout)(batch_size=1)
    appended_mask = torch.ones(1, 2 + layout.geometry_token_count, dtype=torch.bool)
    allowed = build_geometry_full_attention_mask(appended_mask, layout)[0, 0]

    assert slices.depth_current == expected_current
    assert slices.depth_future == expected_future
    assert slices.uvd == slice(4, 6)
    assert layout.geometry_token_count == 4
    assert embeddings.shape == (1, 4, 4)
    remaining_depth = expected_future if enable_future_depth else expected_current
    assert allowed[remaining_depth.start, remaining_depth.stop - 1]
    assert not allowed[slices.uvd.start, slices.uvd.start + 1]


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


def test_shared_depth_attention_pool_returns_normalized_nonuniform_weights_and_gradients():
    pool = SharedDepthAttentionPool(hidden_dim=2)
    with torch.no_grad():
        pool.norm.weight.fill_(1.0)
        pool.norm.bias.zero_()
        pool.score.weight.copy_(torch.tensor([[1.0, -1.0]]))

    tokens = torch.tensor(
        [[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]],
        requires_grad=True,
    )
    summary, weights = pool(tokens)

    assert summary.shape == (1, 2)
    assert weights.shape == (1, 3)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))
    assert weights[0, 0] > weights[0, 2] > weights[0, 1]

    summary.square().sum().backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()
    assert pool.score.weight.grad is not None
    assert pool.score.weight.grad.abs().sum() > 0


def test_shared_depth_attention_pool_reuses_one_parameter_set_for_both_calls():
    pool = SharedDepthAttentionPool(hidden_dim=4)
    current = torch.randn(2, 8, 4)
    future = torch.randn(2, 8, 4)

    current_summary, current_weights = pool(current)
    future_summary, future_weights = pool(future)

    assert current_summary.shape == future_summary.shape == (2, 4)
    assert current_weights.shape == future_weights.shape == (2, 8)
    assert len(list(pool.parameters())) == 3


def test_depth_summary_interventions_preserve_zero_swap_and_cyclic_shuffle_semantics():
    current = torch.tensor([[1.0], [2.0], [3.0]])
    future = torch.tensor([[10.0], [20.0], [30.0]])

    variants = build_depth_summary_interventions(current, future)

    assert torch.equal(variants["normal"][0], current)
    assert torch.equal(variants["normal"][1], future)
    assert torch.equal(variants["zero"][0], torch.zeros_like(current))
    assert torch.equal(variants["zero"][1], torch.zeros_like(future))
    assert torch.equal(variants["swap"][0], future)
    assert torch.equal(variants["swap"][1], current)
    assert torch.equal(variants["shuffle"][0], torch.tensor([[3.0], [1.0], [2.0]]))
    assert torch.equal(variants["shuffle"][1], torch.tensor([[30.0], [10.0], [20.0]]))


def test_depth_summary_interventions_skip_shuffle_for_single_sample():
    variants = build_depth_summary_interventions(torch.ones(1, 2), torch.zeros(1, 2))

    assert set(variants) == {"normal", "zero", "swap"}


@pytest.mark.parametrize("enabled_branch", ["current", "future"])
def test_depth_summary_interventions_keep_a_disabled_branch_absent(enabled_branch):
    summary = torch.tensor([[1.0], [2.0]])
    current = summary if enabled_branch == "current" else None
    future = summary if enabled_branch == "future" else None

    variants = build_depth_summary_interventions(current, future)

    assert set(variants) == {"normal", "zero", "shuffle"}
    disabled_index = 1 if enabled_branch == "current" else 0
    enabled_index = 0 if enabled_branch == "current" else 1
    assert all(values[disabled_index] is None for values in variants.values())
    assert torch.equal(variants["normal"][enabled_index], summary)
    assert torch.equal(variants["zero"][enabled_index], torch.zeros_like(summary))
    assert torch.equal(
        variants["shuffle"][enabled_index],
        torch.tensor([[2.0], [1.0]]),
    )
