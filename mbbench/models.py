from __future__ import annotations

import sys
from pathlib import Path

from .bundle import ModelBundle
from .pretrained import (
    OFFICIAL_MODEL_URLS,
    download_model as download_model_archive,
    model_cache_dir,
    official_model_name,
    prompt_install_directory,
    resolve_model_directory,
)
from .simulator import Simulator


def _official_selection(model: str) -> tuple[str, str, bool]:
    if model not in {"ae", "vae", "ae-system"}:
        raise ValueError("model must be 'ae', 'vae', or 'ae-system'.")
    if model == "ae-system":
        return "ae-system", "ae", True
    return official_model_name(model, False), model, False


def load_model(
    path: str | Path | None = None,
    *,
    model: str = "ae",
    device: str = "auto",
) -> Simulator:
    """Load a custom bundle or an installed official model."""

    if path is not None:
        return Simulator.from_directory(path, device)
    _, base_model, enabled = _official_selection(model)
    root = resolve_model_directory(base_model, None, enabled)
    return Simulator.from_directory(root, device)


def download_model(
    model: str = "ae",
    *,
    install_dir: str | Path | None = None,
    url: str | None = None,
    device: str = "auto",
) -> Simulator:
    """Download an official model and return a ready simulator."""

    name, _, _ = _official_selection(model)
    resolved_url = url or OFFICIAL_MODEL_URLS[name]
    if not resolved_url:
        raise ValueError(
            f"No official download URL is registered for model {name!r}; provide url."
        )
    default_dir = model_cache_dir(name)
    if install_dir is not None:
        target = Path(install_dir).expanduser().resolve()
    elif sys.stdin.isatty() and sys.stdout.isatty():
        target = prompt_install_directory(default_dir)
    else:
        target = default_dir
    bundle = download_model_archive(url=resolved_url, target_dir=target)
    return Simulator.from_directory(bundle.root, device)


def list_conditions(model: Simulator | ModelBundle) -> dict[str, list[str]]:
    """List accepted values for every model condition field."""

    bundle = model.bundle if isinstance(model, Simulator) else model
    if not isinstance(bundle, ModelBundle):
        raise TypeError("model must be a Simulator or ModelBundle.")
    return {
        field: bundle.condition_values(field)
        for field in bundle.config["condition_fields"]
    }
