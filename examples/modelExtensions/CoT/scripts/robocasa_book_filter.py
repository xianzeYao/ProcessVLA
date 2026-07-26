"""DIAL-compatible filtering for the missing RoboCasa ``book`` asset.

This module deliberately operates on a copy of the replay state.  It is used
only immediately before the state is handed to RoboCasa's ``reset_to``; raw
HDF5 data, LeRobot data, and the installed asset tree remain unchanged.
"""

from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any


BOOK_KEYWORD = "book"


def _text(value: Any) -> str:
    """Convert HDF5/NumPy scalar strings to ordinary text."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8")
    if hasattr(value, "tobytes"):
        try:
            return value.tobytes().decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            pass
    return str(value)


def _has_book(value: Any) -> bool:
    try:
        return BOOK_KEYWORD in json.dumps(value, ensure_ascii=False, default=str).lower()
    except (TypeError, ValueError):
        return BOOK_KEYWORD in str(value).lower()


def _remove_book_dicts(value: Any) -> Any:
    """Recursively remove list entries whose serialized dict mentions book.

    This matches the useful part of DIAL's ``remove_dicts_with_keyword``:
    dictionaries inside metadata lists are removed as units, while enclosing
    metadata dictionaries are retained and recursively traversed.
    """
    if isinstance(value, list):
        return [
            _remove_book_dicts(item)
            for item in value
            if not (isinstance(item, dict) and _has_book(item))
        ]
    if isinstance(value, dict):
        return {key: _remove_book_dicts(item) for key, item in value.items()}
    return value


def remove_book_metadata(ep_meta: Any) -> Any:
    """Return ``ep_meta`` with list entries referring to ``book`` removed."""
    if ep_meta is None:
        return None
    if isinstance(ep_meta, (str, bytes, bytearray, memoryview)) or hasattr(ep_meta, "tobytes"):
        parsed = json.loads(_text(ep_meta))
        filtered = _remove_book_dicts(parsed)
        return json.dumps(filtered, ensure_ascii=False)
    return _remove_book_dicts(copy.deepcopy(ep_meta))


def remove_book_nodes(xml: Any) -> str:
    """Remove book-related asset and worldbody nodes from a MuJoCo XML model.

    The scope mirrors DIAL: remove matching direct children under ``asset`` and
    matching object bodies under ``worldbody``.  Matching direct geoms in a
    retained body are removed as well, which handles mixed object bodies.
    """
    root = ET.fromstring(_text(xml))

    for asset in root.iter("asset"):
        for child in list(asset):
            if _has_book(ET.tostring(child, encoding="unicode")):
                asset.remove(child)

    for worldbody in root.iter("worldbody"):
        for child in list(worldbody):
            if child.tag != "body":
                continue
            if _has_book(ET.tostring(child, encoding="unicode")):
                worldbody.remove(child)
                continue
            for geom in list(child):
                if geom.tag == "geom" and _has_book(ET.tostring(geom, encoding="unicode")):
                    child.remove(geom)

    return ET.tostring(root, encoding="unicode")


def make_ik_indicator_invisible(xml: Any) -> str:
    """Hide MuJoCo pinch-sphere sites used only as visual IK indicators."""
    root = ET.fromstring(_text(xml))
    for site in root.iter("site"):
        if "pinch_spheres" in site.get("name", "").lower():
            site.set("rgba", "0 0 0 0")
    return ET.tostring(root, encoding="unicode")


def sanitize_book_state(state: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy and sanitize only ``model``/``ep_meta`` replay fields."""
    sanitized = dict(state)
    if "ep_meta" in sanitized and sanitized["ep_meta"] is not None:
        sanitized["ep_meta"] = remove_book_metadata(sanitized["ep_meta"])
    if "model" in sanitized and sanitized["model"] is not None:
        sanitized["model"] = make_ik_indicator_invisible(remove_book_nodes(sanitized["model"]))
    return sanitized


def make_book_safe_reset_to(original_reset_to: Callable[[Any, dict[str, Any]], Any]) -> Callable[[Any, dict[str, Any]], Any]:
    """Wrap RoboCasa's reset callback with the local book workaround."""
    def reset_to(env: Any, state: dict[str, Any]) -> Any:
        return original_reset_to(env, sanitize_book_state(state))

    return reset_to
