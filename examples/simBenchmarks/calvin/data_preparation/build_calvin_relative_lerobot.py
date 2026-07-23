"""Build Calvin LeRobot output with explicit relative-action metadata."""

from . import build_calvin_starvla_lerobot as converter


converter.CALVIN_MODALITY_FILE = "modality_calvin_lerobot_relative.json"


if __name__ == "__main__":
    converter.main()
