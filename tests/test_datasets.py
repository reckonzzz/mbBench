from __future__ import annotations

import gzip
import json
import shutil
from email.message import Message
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd
import pytest

from mbbench.benchmark_bundle import load_benchmark_bundle, write_benchmark_bundle
from mbbench.datasets import (
    CATALOG_SCHEMA,
    _cached_json,
    dataset,
    download_dataset,
    install_dataset,
    list_datasets,
    show_dataset,
)


def _benchmark_bundle(path: Path) -> Path:
    sample_ids = ["sample_1", "sample_2"]
    taxa = ["g1", "g2", "g3"]

    def abundance(values):
        frame = pd.DataFrame(values, columns=taxa)
        frame.insert(0, "sample_id", sample_ids)
        return frame

    counts = abundance([[10, 20, 70], [0, 40, 60]])
    relative = abundance([[0.1, 0.2, 0.7], [0.0, 0.4, 0.6]])
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group": ["control", "case"],
            "batch": ["batch_1", "batch_1"],
            "raw_total_reads": [100, 100],
        }
    )
    feature_truth = pd.DataFrame(
        {
            "taxon": taxa,
            "is_direct_biological_target": [False, True, False],
        }
    )
    sample_truth = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group_index": [0, 1],
            "batch_index": [0, 0],
        }
    )
    write_benchmark_bundle(
        path,
        tables={
            "counts": counts,
            "relative_abundance": relative,
            "latent_composition": relative.copy(),
            "metadata": metadata,
            "feature_truth": feature_truth,
            "sample_truth": sample_truth,
            "batch_neutral_reference": relative.copy(),
        },
        config={"schema_version": "mbbench_benchmark_v1", "preset": "tiny"},
    )
    return path


def _archive(bundle: Path, destination: Path) -> Path:
    archive = shutil.make_archive(
        str(destination.with_suffix("")),
        "zip",
        root_dir=bundle.parent,
        base_dir=bundle.name,
    )
    return Path(archive)


def _catalog(archive: Path, *, version: str = "2026.1") -> dict:
    return {
        "schema_version": CATALOG_SCHEMA,
        "latest": version,
        "releases": [
            {
                "version": version,
                "version_doi": "10.5281/zenodo.1",
                "record_id": "1",
                "datasets": [
                    {
                        "preset": "tiny",
                        "seed": 42,
                        "archive": archive.name,
                        "url": archive.as_uri(),
                        "n_samples": 2,
                        "n_batches": 1,
                    }
                ],
            }
        ],
    }


def test_dataset_loads_directory_and_manual_zip(tmp_path, monkeypatch):
    import mbbench.datasets as datasets

    monkeypatch.setattr(datasets, "dataset_cache_dir", lambda: tmp_path / "cache")
    bundle = _benchmark_bundle(tmp_path / "bundle")
    local = dataset(path=bundle)
    assert local.source == {"type": "local_directory", "path": str(bundle.resolve())}
    assert local.counts.shape == (2, 4)
    assert load_benchmark_bundle(bundle).source is None
    with pytest.raises(TypeError):
        load_benchmark_bundle(bundle, source={"type": "user"})

    archive = _archive(bundle, tmp_path / "tiny.zip")
    imported = dataset(archive=archive)
    assert imported.source["type"] == "manual_archive"
    assert imported.relative_abundance.iloc[:, 1:].sum(axis=1).eq(1).all()
    assert imported.root == (tmp_path / "cache" / "manual" / "tiny").resolve()

    with pytest.raises(ValueError, match="exactly one"):
        dataset(path=bundle, archive=archive)


@pytest.mark.parametrize("artifact_record_id", [None, "2"])
def test_catalog_listing_and_download_always_load_full_bundle(tmp_path, monkeypatch, artifact_record_id):
    import mbbench.datasets as datasets

    monkeypatch.setattr(datasets, "dataset_cache_dir", lambda: tmp_path / "cache")
    bundle = _benchmark_bundle(tmp_path / "bundle")
    archive = _archive(bundle, tmp_path / "tiny.zip")
    catalog = _catalog(archive)
    if artifact_record_id is not None:
        catalog["releases"][0]["datasets"][0].update(
            artifact_record_id=artifact_record_id,
            artifact_version_doi=f"10.5281/zenodo.{artifact_record_id}",
        )
    monkeypatch.setattr(datasets, "_load_catalog", lambda: catalog)

    entries = list_datasets()
    assert entries[0]["version"] == "2026.1"
    assert show_dataset("tiny")[0]["n_samples"] == 2

    installed, installed_entry = install_dataset("tiny")
    assert installed == (tmp_path / "cache" / "2026.1" / "tiny_seed42").resolve()
    assert installed_entry["seed"] == 42

    downloaded = download_dataset("tiny")
    assert downloaded.source["requested_version"] == "latest"
    assert downloaded.source["resolved_version"] == "2026.1"
    expected_record_id = artifact_record_id or "1"
    assert downloaded.source["record_id"] == expected_record_id
    assert downloaded.source["version_doi"] == f"10.5281/zenodo.{expected_record_id}"
    np.testing.assert_array_equal(
        downloaded.counts.iloc[:, 1:], [[10, 20, 70], [0, 40, 60]]
    )


def test_cli_install_path_does_not_load_full_bundle(tmp_path, monkeypatch):
    import mbbench.datasets as datasets

    monkeypatch.setattr(datasets, "dataset_cache_dir", lambda: tmp_path / "cache")
    bundle = _benchmark_bundle(tmp_path / "bundle")
    archive = _archive(bundle, tmp_path / "tiny.zip")
    monkeypatch.setattr(datasets, "_load_catalog", lambda: _catalog(archive))
    monkeypatch.setattr(
        datasets,
        "_load_benchmark_bundle",
        lambda *args, **kwargs: pytest.fail("install-only path loaded the full bundle"),
    )
    root, _ = datasets.install_dataset("tiny")
    assert (root / "benchmark.yaml").is_file()
    manual = datasets.install_dataset_archive(archive, install_dir=tmp_path / "manual")
    assert (manual / "counts.csv.gz").is_file()


class _Response:
    def __init__(self, payload: dict, headers: dict[str, str]):
        self.payload = json.dumps(payload).encode("utf-8")
        self.headers = Message()
        for name, value in headers.items():
            self.headers[name] = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


@pytest.mark.parametrize("network_error", [URLError("offline"), IncompleteRead(b"{", 20)])
def test_catalog_check_is_automatic_and_uses_cached_copy(tmp_path, monkeypatch, network_error):
    import mbbench.datasets as datasets

    monkeypatch.setattr(datasets, "dataset_cache_dir", lambda: tmp_path / "cache")
    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        assert request.get_header("Accept-encoding") == "gzip"
        if len(calls) == 1:
            response = _Response({"release": "one"}, {"ETag": '"v1"', "Content-Encoding": "gzip"})
            response.payload = gzip.compress(response.payload)
            return response
        assert request.get_header("If-none-match") == '"v1"'
        raise HTTPError(request.full_url, 304, "Not Modified", Message(), None)

    monkeypatch.setattr(datasets, "urlopen", urlopen)
    assert _cached_json("https://example.test/catalog", "test") == {"release": "one"}
    assert _cached_json("https://example.test/catalog", "test") == {"release": "one"}

    def offline(request, timeout):
        raise network_error

    monkeypatch.setattr(datasets, "urlopen", offline)
    with pytest.warns(UserWarning, match="cached copy"):
        assert _cached_json("https://example.test/catalog", "test") == {
            "release": "one"
        }
