from __future__ import annotations

import csv
import gzip
import gzip
import json
import shutil
import tempfile
import warnings
import zipfile
from http.client import IncompleteRead
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._download import copy_download
from .benchmark_bundle import BenchmarkBundle, _load_benchmark_bundle
from .common import load_yaml, require_bool, staged_output_dir


OFFICIAL_DATASET_RECORD_URL = "https://zenodo.org/api/records/24076644"
CATALOG_FILENAME = "mbbench_dataset_catalog.json"
CATALOG_SCHEMA = "mbbench_dataset_catalog_v1"
REQUIRED_BUNDLE_FILES = (
    "benchmark.yaml",
    "counts.csv.gz",
    "relative_abundance.csv.gz",
    "latent_composition.csv.gz",
    "metadata.csv.gz",
    "feature_truth.csv.gz",
    "sample_truth.csv.gz",
    "batch_neutral_biological_reference.csv.gz",
)


def dataset_cache_dir() -> Path:
    return Path.home() / ".cache" / "mbbench" / "datasets"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def _cached_json(url: str, name: str) -> dict[str, Any]:
    cache = dataset_cache_dir() / "catalog"
    payload_path = cache / f"{name}.json"
    state_path = cache / f"{name}.http.json"
    state = _read_json(state_path) if state_path.is_file() else {}
    headers = {"Accept-Encoding": "gzip"}
    if state.get("url") == url:
        if state.get("etag"):
            headers["If-None-Match"] = str(state["etag"])
        if state.get("last_modified"):
            headers["If-Modified-Since"] = str(state["last_modified"])
    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            body = response.read()
            if response.headers.get("Content-Encoding", "").lower() == "gzip":
                body = gzip.decompress(body)
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Remote JSON root must be an object: {url}")
            _write_json(payload_path, payload)
            _write_json(
                state_path,
                {
                    "url": url,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                },
            )
            return payload
    except HTTPError as error:
        if error.code == 304 and payload_path.is_file():
            return _read_json(payload_path)
        if payload_path.is_file():
            warnings.warn(
                f"Could not refresh the mbBench dataset catalog ({error}); using the cached copy.",
                stacklevel=2,
            )
            return _read_json(payload_path)
        raise RuntimeError(
            f"Could not download mbBench dataset metadata: {error}"
        ) from error
    except (OSError, EOFError, IncompleteRead) as error:
        if payload_path.is_file():
            warnings.warn(
                f"Could not refresh the mbBench dataset catalog ({error}); using the cached copy.",
                stacklevel=2,
            )
            return _read_json(payload_path)
        raise RuntimeError(
            "Could not reach the mbBench dataset catalog. Download the dataset ZIP "
            "manually and use mbbench.dataset(archive=PATH) or "
            "'mbbench import-dataset ARCHIVE'."
        ) from error


def _catalog_url(record: Mapping[str, Any]) -> str:
    files = record.get("files")
    if isinstance(files, Mapping):
        entries = files.get("entries", [])
    else:
        entries = files or []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("key") or entry.get("filename")
        if name != CATALOG_FILENAME:
            continue
        links = entry.get("links", {})
        if isinstance(links, Mapping):
            url = links.get("content") or links.get("download") or links.get("self")
            if url:
                return str(url)
    raise ValueError(
        f"The official dataset record does not contain {CATALOG_FILENAME}."
    )


def _validate_catalog(catalog: Mapping[str, Any]) -> None:
    if catalog.get("schema_version") != CATALOG_SCHEMA:
        raise ValueError(
            f"Unsupported dataset catalog schema: {catalog.get('schema_version')!r}."
        )
    latest = catalog.get("latest")
    releases = catalog.get("releases")
    if not isinstance(latest, str) or not latest:
        raise ValueError("Dataset catalog requires a nonempty latest release.")
    if not isinstance(releases, list) or not releases:
        raise ValueError("Dataset catalog requires at least one release.")
    versions: set[str] = set()
    for release in releases:
        if not isinstance(release, Mapping):
            raise ValueError("Every dataset release must be an object.")
        version = release.get("version")
        datasets = release.get("datasets")
        if not isinstance(version, str) or not version or version in versions:
            raise ValueError(
                "Dataset release versions must be unique nonempty strings."
            )
        versions.add(version)
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(f"Dataset release {version!r} contains no datasets.")
        identities: set[tuple[str, int]] = set()
        for entry in datasets:
            if not isinstance(entry, Mapping):
                raise ValueError("Every dataset entry must be an object.")
            preset = entry.get("preset")
            seed = entry.get("seed")
            archive = entry.get("archive")
            url = entry.get("url")
            if not isinstance(preset, str) or not preset:
                raise ValueError("Every dataset entry requires a preset name.")
            if not isinstance(seed, int):
                raise ValueError(f"Dataset {preset!r} requires an integer seed.")
            if not isinstance(archive, str) or not archive.endswith(".zip"):
                raise ValueError(f"Dataset {preset!r} requires a ZIP archive name.")
            if not isinstance(url, str) or not url:
                raise ValueError(f"Dataset {preset!r} requires a download URL.")
            identity = (preset, seed)
            if identity in identities:
                raise ValueError(
                    f"Dataset release {version!r} repeats preset {preset!r}, seed {seed}."
                )
            identities.add(identity)
    if latest not in versions:
        raise ValueError(f"Latest dataset release {latest!r} is not listed.")


def _load_catalog() -> dict[str, Any]:
    if not OFFICIAL_DATASET_RECORD_URL:
        raise RuntimeError(
            "The official mbBench dataset record has not been published yet. "
            "Use mbbench.dataset(archive=PATH) for a manually downloaded ZIP."
        )
    record = _cached_json(OFFICIAL_DATASET_RECORD_URL, "record")
    links = record.get("links", {})
    latest_url = links.get("latest") if isinstance(links, Mapping) else None
    if latest_url and str(latest_url) != OFFICIAL_DATASET_RECORD_URL:
        record = _cached_json(str(latest_url), "latest_record")
    catalog = _cached_json(_catalog_url(record), "datasets")
    _validate_catalog(catalog)
    return catalog


def _release(catalog: Mapping[str, Any], version: str) -> Mapping[str, Any]:
    selected = catalog["latest"] if version == "latest" else version
    for release in catalog["releases"]:
        if release["version"] == selected:
            return release
    available = ", ".join(str(item["version"]) for item in catalog["releases"])
    raise ValueError(
        f"Unknown dataset version {version!r}. Available versions: {available}."
    )


def _entries(version: str = "latest") -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    catalog = _load_catalog()
    release = _release(catalog, version)
    entries = [
        {
            **dict(entry),
            "version": release["version"],
            "version_doi": entry.get("artifact_version_doi") or release.get("version_doi"),
            "record_id": entry.get("artifact_record_id") or release.get("record_id"),
        }
        for entry in release["datasets"]
    ]
    return release, entries


def list_datasets(*, version: str = "latest") -> list[dict[str, Any]]:
    """List pre-generated benchmark datasets in an official release."""

    _, entries = _entries(version)
    return entries


def show_dataset(preset: str, *, version: str = "latest") -> list[dict[str, Any]]:
    """Return every catalog entry for one preset."""

    entries = list_datasets(version=version)
    matches = [entry for entry in entries if entry["preset"] == preset]
    if not matches:
        available = sorted({entry["preset"] for entry in entries})
        raise ValueError(
            f"Unknown pre-generated dataset {preset!r}. Available presets: {', '.join(available)}."
        )
    return matches


def _select_dataset(preset: str, seed: int, version: str) -> dict[str, Any]:
    _, entries = _entries(version)
    for entry in entries:
        if entry["preset"] == preset and entry["seed"] == seed:
            return entry
    available = sorted((entry["preset"], entry["seed"]) for entry in entries)
    choices = ", ".join(f"{name} (seed {value})" for name, value in available)
    raise ValueError(
        f"No pre-generated dataset matches preset {preset!r}, seed {seed}. "
        f"Available datasets: {choices}."
    )


def _bundle_root(root: Path) -> Path:
    if (root / "benchmark.yaml").is_file():
        return root
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "benchmark.yaml").is_file()
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("Dataset archive must contain one mbBench benchmark bundle.")


def _csv_header_and_first_row(path: Path) -> tuple[list[str], list[str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        first = next(reader, [])
    if not header or not first or len(first) != len(header):
        raise ValueError(f"Benchmark table is empty or malformed: {path.name}")
    return header, first


def _validate_bundle_layout(root: Path) -> Path:
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(
            f"Benchmark dataset is incomplete; missing: {', '.join(missing)}"
        )
    config = load_yaml(root / "benchmark.yaml")
    if not config:
        raise ValueError("benchmark.yaml must contain a configuration mapping.")
    counts_header, counts_first = _csv_header_and_first_row(root / "counts.csv.gz")
    if len(counts_header) < 2 or counts_header[0] != "sample_id":
        raise ValueError("counts.csv.gz requires sample_id followed by taxon columns.")
    for name in (
        "relative_abundance.csv.gz",
        "latent_composition.csv.gz",
        "batch_neutral_biological_reference.csv.gz",
    ):
        header, first = _csv_header_and_first_row(root / name)
        if header != counts_header or first[0] != counts_first[0]:
            raise ValueError(
                f"{name} does not match the count-table columns or sample IDs."
            )
    for name in ("metadata.csv.gz", "sample_truth.csv.gz"):
        header, first = _csv_header_and_first_row(root / name)
        if (
            "sample_id" not in header
            or first[header.index("sample_id")] != counts_first[0]
        ):
            raise ValueError(f"{name} does not match the first count-table sample ID.")
    feature_header, _ = _csv_header_and_first_row(root / "feature_truth.csv.gz")
    if "taxon" not in feature_header:
        raise ValueError("feature_truth.csv.gz requires a taxon column.")
    return root.resolve()


def _extract_archive(archive: Path, target: Path, overwrite: bool) -> Path:
    overwrite = require_bool(overwrite, "overwrite")
    if target.exists():
        if not overwrite:
            return _validate_bundle_layout(target)
    with staged_output_dir(target, overwrite=overwrite) as staging:
        extracted = staging / ".archive"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as bundle_archive:
            for member in bundle_archive.infolist():
                resolved = (extracted / member.filename).resolve()
                if extracted not in resolved.parents and resolved != extracted:
                    raise ValueError("Dataset archive contains an invalid path.")
            bundle_archive.extractall(extracted)
        source = _bundle_root(extracted)
        _validate_bundle_layout(source)
        for item in source.iterdir():
            item.replace(staging / item.name)
        shutil.rmtree(extracted)
    return target.resolve()


def install_dataset_archive(
    archive: str | Path,
    *,
    install_dir: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Install a local dataset archive without loading all tables."""

    source = Path(archive).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset archive does not exist: {source}")
    target = (
        Path(install_dir).expanduser().resolve()
        if install_dir is not None
        else dataset_cache_dir() / "manual" / source.stem
    )
    return _extract_archive(source, target, overwrite)


def import_dataset(
    archive: str | Path,
    *,
    install_dir: str | Path | None = None,
    overwrite: bool = False,
) -> BenchmarkBundle:
    """Install and load a benchmark dataset from a local ZIP archive."""

    source = Path(archive).expanduser().resolve()
    root = install_dataset_archive(source, install_dir=install_dir, overwrite=overwrite)
    return _load_benchmark_bundle(
        root,
        source={"type": "manual_archive", "archive": str(source)},
    )


def download_dataset(
    preset: str,
    *,
    seed: int = 42,
    version: str = "latest",
    install_dir: str | Path | None = None,
    overwrite: bool = False,
) -> BenchmarkBundle:
    """Download and fully load an official pre-generated benchmark dataset."""

    root, entry = install_dataset(
        preset,
        seed=seed,
        version=version,
        install_dir=install_dir,
        overwrite=overwrite,
    )
    release = str(entry["version"])
    return _load_benchmark_bundle(
        root,
        source={
            "type": "zenodo_download",
            "requested_version": version,
            "resolved_version": release,
            "version_doi": entry.get("version_doi"),
            "record_id": entry.get("record_id"),
            "preset": preset,
            "seed": seed,
        },
    )


def install_dataset(
    preset: str,
    *,
    seed: int = 42,
    version: str = "latest",
    install_dir: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Install an official dataset without loading all tables."""

    overwrite = require_bool(overwrite, "overwrite")
    entry = _select_dataset(preset, seed, version)
    release = str(entry["version"])
    target = (
        Path(install_dir).expanduser().resolve()
        if install_dir is not None
        else dataset_cache_dir() / release / f"{preset}_seed{seed}"
    )
    if target.exists() and not overwrite:
        root = _validate_bundle_layout(target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            archive = Path(temporary) / str(entry["archive"])
            try:
                with urlopen(str(entry["url"]), timeout=120) as response:
                    copy_download(response, archive, label=archive.name)
            except (OSError, HTTPError, URLError) as error:
                raise RuntimeError(
                    f"Could not download {entry['archive']}. Download the ZIP manually "
                    "and use mbbench.dataset(archive=PATH) or "
                    "'mbbench import-dataset ARCHIVE'."
                ) from error
            root = _extract_archive(archive, target, overwrite=True)
    return root, entry


def dataset(
    *,
    preset: str | None = None,
    archive: str | Path | None = None,
    path: str | Path | None = None,
    seed: int = 42,
    version: str = "latest",
    install_dir: str | Path | None = None,
    overwrite: bool = False,
) -> BenchmarkBundle:
    """Fully load an official, archived, or extracted benchmark dataset."""

    overwrite = require_bool(overwrite, "overwrite")
    sources = [preset is not None, archive is not None, path is not None]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one of preset, archive, or path.")
    if preset is not None:
        return download_dataset(
            preset,
            seed=seed,
            version=version,
            install_dir=install_dir,
            overwrite=overwrite,
        )
    if archive is not None:
        return import_dataset(archive, install_dir=install_dir, overwrite=overwrite)
    root = Path(path).expanduser().resolve()
    return _load_benchmark_bundle(
        root, source={"type": "local_directory", "path": str(root)}
    )
