from __future__ import annotations

import random
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import torch
from .yaml_io import load_yaml, write_yaml


def select_device(value: str = "auto") -> torch.device:
    """Resolve auto, CPU, or a CUDA device using the installed PyTorch build."""

    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation cannot use CUDA."
        )
    return device


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def capture_rng_state() -> dict[str, Any]:
    """Capture random generators needed to continue at the next epoch."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore random generators saved at an epoch boundary."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def require_bool(value: object, name: str) -> bool:
    """Return a Boolean value or reject ambiguous truthy inputs."""

    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be True or False.")
    return bool(value)


def new_output_dir(path: str | Path, *, overwrite: bool = False) -> Path:
    """Create an output directory without silently replacing prior results."""

    overwrite = require_bool(overwrite, "overwrite")
    output = Path(path).expanduser().resolve()
    if (
        output == Path.home().resolve()
        or output == Path(output.anchor)
        or output.parent == output
        or (output / ".git").exists()
    ):
        raise ValueError(f"Refusing to use a broad output directory: {output}")
    if output.exists() and not overwrite:
        if not output.is_dir():
            raise FileExistsError(
                f"Output path is an existing file: {output}. Use --overwrite to replace it."
            )
        if any(output.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output}. Use --overwrite to replace files."
            )
    if output.exists() and overwrite:
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    output.mkdir(parents=True, exist_ok=True)
    return output


@contextmanager
def staged_output_dir(path: str | Path, *, overwrite: bool = False):
    """Stage output beside its destination before replacement."""

    overwrite = require_bool(overwrite, "overwrite")
    output = Path(path).expanduser().resolve()
    if (
        output == Path.home().resolve()
        or output == Path(output.anchor)
        or output.parent == output
        or (output / ".git").exists()
    ):
        raise ValueError(f"Refusing to use a broad output directory: {output}")
    if output.exists() and not overwrite:
        if not output.is_dir():
            raise FileExistsError(
                f"Output path is an existing file: {output}. Use --overwrite to replace it."
            )
        if any(output.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {output}. Use --overwrite to replace it."
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        yield staging
        if output.exists():
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
        staging.replace(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def write_training_progress(
    progress_dir: str | Path, message: str, status: Mapping[str, Any]
) -> None:
    """Append a concise training message and replace the machine-readable current status."""

    directory = Path(progress_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with (directory / "train_progress.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")
    (directory / "latest_status.json").write_text(
        json.dumps({**status, "updated_at": timestamp}, indent=2) + "\n",
        encoding="utf-8",
    )
