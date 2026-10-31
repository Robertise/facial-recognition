"""
config.py – innovation settings: feature flags from the environment, every
threshold and parameter from innovation/config.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(os.environ.get("INNOVATION_CONFIG", Path(__file__).with_name("config.json")))


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    """INNOVATION_ENABLED=1 mounts the innovation API and creates its tables."""
    return _flag("INNOVATION_ENABLED")


def research_mode() -> bool:
    """INNOVATION_RESEARCH_MODE=1 unlocks file-mode analysis and data collection."""
    return _flag("INNOVATION_RESEARCH_MODE")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("challenge", "asr", "fusion", "thresholds", "media"):
        if key not in cfg:
            raise ValueError(f"innovation config is missing '{key}'")
    return cfg
