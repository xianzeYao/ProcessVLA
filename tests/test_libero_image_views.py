import pytest


def test_agentview_mode_keeps_only_first_image_as_a_list():
    from starVLA.libero_image_views import select_libero_image_views

    primary = object()
    wrist = object()

    assert select_libero_image_views([primary, wrist], "agentview") == [primary]


def test_unknown_view_mode_is_rejected():
    from starVLA.libero_image_views import select_libero_image_views

    with pytest.raises(ValueError, match="view_mode"):
        select_libero_image_views([object(), object()], "wrist")


def test_libero_cot_data_config_uses_only_primary_video_in_agentview_mode():
    from starVLA.dataloader.cot_lerobot_datasets import LiberoCoTDataConfig

    config = LiberoCoTDataConfig(image_views="agentview")

    assert config.video_keys == ["video.primary_image"]


def test_agentview_mode_rejects_an_empty_image_list():
    from starVLA.libero_image_views import select_libero_image_views

    with pytest.raises(ValueError, match="at least one"):
        select_libero_image_views([], "agentview")


def test_all_mode_preserves_both_images_and_training_video_order():
    from starVLA.dataloader.cot_lerobot_datasets import LiberoCoTDataConfig
    from starVLA.libero_image_views import select_libero_image_views

    primary, wrist = object(), object()

    assert select_libero_image_views([primary, wrist], "all") == [primary, wrist]
    assert LiberoCoTDataConfig().video_keys == ["video.primary_image", "video.wrist_image"]
