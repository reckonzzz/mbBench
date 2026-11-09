from __future__ import annotations

import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from ._download import check_archive_path, copy_download, extract_zip
from .bundle import ModelBundle


_ZENODO_MODEL_FILES = "https://zenodo.org/api/records/24080099/files"
OFFICIAL_MODEL_URLS: dict[str, str] = {
    name: f"{_ZENODO_MODEL_FILES}/mbbench-{name}.zip/content"
    for name in ("ae", "vae", "ae-system")
}
_OFFICIAL_ARCHIVE_SHA256 = {
    OFFICIAL_MODEL_URLS["ae"]: "d3d9683acfd8aaa71b4434a75b7e96f49f30203117fe44dd629a0f6e3047495b",
    OFFICIAL_MODEL_URLS["vae"]: "929b8c2f3c03089282262240097e71c70a900ee52a3afcc8bd73df81dea3c11c",
    OFFICIAL_MODEL_URLS["ae-system"]: "52b8bf52a3e241431e331ff0936d7997f4460ed01454ca16a2d1eb10336a083c",
}


def model_cache_dir(name: str) -> Path:
    return Path.home() / ".cache" / "mbbench" / "models" / name


def official_model_name(model: str, use_body_system: bool) -> str:
    if model == "vae" and use_body_system:
        raise ValueError("The official VAE does not include Body-System conditioning.")
    return "ae-system" if use_body_system else model


def prompt_install_directory(default_dir: str | Path) -> Path:
    """Ask for a model installation directory, accepting the displayed default."""

    default = Path(default_dir).expanduser().resolve()
    answer = input(f"Install the model in [{default}]: ").strip()
    return Path(answer).expanduser().resolve() if answer else default


def _missing_model_message(name: str, model: str, target: Path) -> str:
    return (
        f"Official model {name!r} is not installed in {target}. "
        f"Run 'mbbench download-model --model {model}', or use "
        "'--model custom --model-dir PATH'."
    )


def resolve_model_directory(
    model: str, model_dir: str | Path | None, use_body_system: bool
) -> Path:
    if model == "custom":
        if model_dir is None:
            raise ValueError("--model custom requires --model-dir.")
        return ModelBundle.load(model_dir).root
    name = official_model_name(model, use_body_system)
    target = model_cache_dir(name)
    if target.is_dir():
        return ModelBundle.load(target).root

    url = OFFICIAL_MODEL_URLS[name]
    message = _missing_model_message(name, model, target)
    if not url:
        raise FileNotFoundError(
            message + " This release has no registered official URL yet."
        )
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise FileNotFoundError(message)
    answer = (
        input(f"Official model {name!r} was not found. Download it now? [y/N]: ")
        .strip()
        .lower()
    )
    if answer not in {"y", "yes"}:
        raise FileNotFoundError(message)
    install_dir = prompt_install_directory(target)
    return download_model(url=url, target_dir=install_dir).root


def _extract_model_archive(archive: Path, directory: Path) -> None:
    if zipfile.is_zipfile(archive):
        extract_zip(archive, directory, label="Model archive")
        return

    with tarfile.open(archive, "r:gz") as bundle_archive:
        members = bundle_archive.getmembers()
        for member in members:
            if not (member.isfile() or member.isdir()):
                raise ValueError("Model archive must contain only files and directories.")
            check_archive_path(directory, member.name, label="Model archive")
        bundle_archive.extractall(directory, members)


def download_model(
    *, url: str, target_dir: str | Path, timeout: int = 120,
    progress: bool | None = None,
) -> ModelBundle:
    """Download a ZIP or tar.gz model archive and validate its required files."""

    target = Path(target_dir).expanduser().resolve()
    if target.exists():
        return ModelBundle.load(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / "model.archive"
        with urlopen(url, timeout=timeout) as response:
            copy_download(
                response, archive, label=f"model archive for {target.name}",
                expected_sha256=_OFFICIAL_ARCHIVE_SHA256.get(url), progress=progress,
            )
        extracted = temporary_root / "extracted"
        extracted.mkdir()
        _extract_model_archive(archive, extracted)
        roots = [path for path in extracted.iterdir() if path.is_dir()]
        source = roots[0] if len(roots) == 1 else extracted
        ModelBundle.load(source)
        try:
            source.replace(target)
        except OSError:
            if not target.is_dir():
                raise
    return ModelBundle.load(target)
