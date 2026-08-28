"""Shared LIBERO policy-image view selection."""


def select_libero_image_views(images, view_mode: str):
    """Return the ordered image list expected by a LIBERO policy."""
    if view_mode not in {"all", "agentview"}:
        raise ValueError(f"unknown view_mode={view_mode!r}; expected 'all' or 'agentview'")
    if not images:
        raise ValueError("LIBERO policy input requires at least one image")
    if view_mode == "agentview":
        return list(images[:1])
    return list(images)
