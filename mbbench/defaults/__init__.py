"""Default training and generation configurations."""

from importlib.resources import files
from typing import Any

import yaml


def load_default_config(name: str) -> dict[str, Any]:
    payload = yaml.safe_load(files(__name__).joinpath(name).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Default configuration root must be a mapping: {name}")
    return payload


DEFAULT_MAX_IN_MEMORY_MB = 512.0
