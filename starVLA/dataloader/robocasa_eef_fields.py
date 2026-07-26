"""Field-selection helpers for RoboCasa EEF geometry."""

from __future__ import annotations

from collections.abc import Iterable


def select_robocasa_eef_world_columns(columns: Iterable[str]) -> tuple[str, str]:
    """Select bilateral EEF world-position columns, never pinch markers.

    New rerenders expose the explicit raw robot EEF fields. Older rerenders
    only have the legacy ``left/right_eef_pos`` fields, which are the same XYZ
    point for the current Fourier XML and remain a safe compatibility fallback.
    """

    available = set(columns)
    explicit = (
        "observation.left_robot_eef_pos",
        "observation.right_robot_eef_pos",
    )
    legacy = (
        "observation.left_eef_pos",
        "observation.right_eef_pos",
    )
    if set(explicit) <= available:
        return explicit
    if set(legacy) <= available:
        return legacy
    raise KeyError(
        "RoboCasa episode must provide bilateral EEF fields; "
        f"missing explicit={sorted(set(explicit) - available)}, "
        f"missing legacy={sorted(set(legacy) - available)}"
    )


def select_robocasa_uvd_world_columns(columns: Iterable[str]) -> tuple[str, str]:
    """Select bilateral thumb-index pinch fields for active UVD supervision.

    The active CoT target is deliberately strict: an old EEF-only rerender must
    fail loudly instead of silently changing the supervision point.
    """

    available = set(columns)
    thumb_index = (
        "observation.left_thumb_index_pinch_pos",
        "observation.right_thumb_index_pinch_pos",
    )
    if set(thumb_index) <= available:
        return thumb_index
    raise KeyError(
        "RoboCasa UVD supervision requires thumb-index fields; "
        f"missing={sorted(set(thumb_index) - available)}"
    )
