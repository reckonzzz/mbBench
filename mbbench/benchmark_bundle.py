from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Mapping, TextIO

import numpy as np
import pandas as pd

from .common import load_yaml, staged_output_dir, write_yaml


@dataclass(frozen=True)
class BenchmarkBundle:
    """A fully loaded benchmark dataset.

    The object contains count, relative-abundance, latent-composition, metadata,
    feature-truth, sample-truth, batch-neutral-reference, and optional paired
    neutral-count tables, plus the resolved configuration, root, and source.
    """

    root: Path
    config: dict
    counts: pd.DataFrame
    relative_abundance: pd.DataFrame
    latent_composition: pd.DataFrame
    metadata: pd.DataFrame
    feature_truth: pd.DataFrame
    sample_truth: pd.DataFrame
    batch_neutral_reference: pd.DataFrame
    batch_neutral_counts: pd.DataFrame | None
    source: dict[str, object] | None = None


_SAMPLE_TABLE_FILES = {
    "counts": "counts.csv.gz",
    "relative_abundance": "relative_abundance.csv.gz",
    "latent_composition": "latent_composition.csv.gz",
    "metadata": "metadata.csv.gz",
    "sample_truth": "sample_truth.csv.gz",
    "batch_neutral_reference": "batch_neutral_biological_reference.csv.gz",
    "batch_neutral_counts": "batch_neutral_counts.csv.gz",
}


def _taxa(frame: pd.DataFrame) -> list[str]:
    if "sample_id" not in frame:
        raise ValueError("Benchmark abundance tables require a sample_id column.")
    taxa = [str(column) for column in frame.columns if column != "sample_id"]
    if any(not taxon.strip() for taxon in taxa) or len(taxa) != len(set(taxa)):
        raise ValueError("Benchmark taxon names must be nonempty and unique.")
    return taxa


def _validate_tables(
    tables: Mapping[str, pd.DataFrame], config: Mapping[str, object]
) -> None:
    required = {
        "counts",
        "relative_abundance",
        "latent_composition",
        "metadata",
        "feature_truth",
        "sample_truth",
        "batch_neutral_reference",
    }
    missing = sorted(required - set(tables))
    if missing:
        raise ValueError(f"Benchmark bundle is missing tables: {', '.join(missing)}")
    counts = tables["counts"]
    relative = tables["relative_abundance"]
    latent = tables["latent_composition"]
    reference = tables["batch_neutral_reference"]
    taxa = _taxa(counts)
    if (
        not taxa
        or _taxa(relative) != taxa
        or _taxa(latent) != taxa
        or _taxa(reference) != taxa
    ):
        raise ValueError("Benchmark abundance tables must have the same ordered taxa.")
    sample_id_values = counts["sample_id"]
    if (
        sample_id_values.isna().any()
        or sample_id_values.astype(str).str.strip().eq("").any()
        or sample_id_values.astype(str).duplicated().any()
    ):
        raise ValueError("Benchmark sample IDs must be nonempty and unique.")
    sample_ids = sample_id_values.astype(str).tolist()
    for name in (
        "relative_abundance",
        "latent_composition",
        "metadata",
        "sample_truth",
        "batch_neutral_reference",
    ):
        frame = tables[name]
        if frame["sample_id"].astype(str).tolist() != sample_ids:
            raise ValueError(f"{name} sample IDs or row order do not match counts.")
    count_values = counts[taxa].to_numpy(dtype=np.float64)
    if np.any(count_values < 0) or not np.array_equal(
        count_values, np.rint(count_values)
    ):
        raise ValueError("Benchmark counts must be nonnegative integers.")
    depths = count_values.sum(axis=1)
    if np.any(depths <= 0):
        raise ValueError("Every benchmark count row must have positive depth.")
    expected_relative = count_values / depths[:, None]
    relative_values = relative[taxa].to_numpy(dtype=np.float64)
    if not np.allclose(relative_values, expected_relative, atol=1e-12, rtol=0):
        raise ValueError("relative_abundance must equal counts divided by row depth.")
    for name, frame in (
        ("latent_composition", latent),
        ("batch_neutral_reference", reference),
    ):
        values = frame[taxa].to_numpy(dtype=np.float64)
        if (
            np.any(values < 0)
            or not np.isfinite(values).all()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-10)
        ):
            raise ValueError(f"{name} must be finite, nonnegative, and closed.")
    feature_truth = tables["feature_truth"]
    if (
        "taxon" not in feature_truth
        or feature_truth["taxon"].astype(str).tolist() != taxa
    ):
        raise ValueError("feature_truth taxon order does not match abundance tables.")
    neutral_counts = tables.get("batch_neutral_counts")
    if neutral_counts is not None:
        if (
            _taxa(neutral_counts) != taxa
            or neutral_counts["sample_id"].astype(str).tolist() != sample_ids
        ):
            raise ValueError("batch_neutral_counts does not match the primary tables.")
        neutral_values = neutral_counts[taxa].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(neutral_values).all()
            or np.any(neutral_values < 0)
            or not np.array_equal(neutral_values, np.rint(neutral_values))
        ):
            raise ValueError("batch_neutral_counts must contain nonnegative integers.")
    internal = config.get("resolved_internal", {})
    if isinstance(internal, Mapping) and internal.get("strict_target_audit"):
        audit = internal["strict_target_audit"]
        if not isinstance(audit, Mapping):
            raise ValueError("strict_target_audit must be a mapping.")
        if float(audit.get("maximum_non_target_change", 1)) > 1e-12:
            raise ValueError("Strict target non-target invariant failed.")
        if float(audit.get("maximum_target_mass_change", 1)) > 1e-10:
            raise ValueError("Strict target mass invariant failed.")
        if int(audit.get("support_change_count", 1)) != 0:
            raise ValueError("Strict target support invariant failed.")


def _validate_sample_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    required = {
        "counts",
        "relative_abundance",
        "latent_composition",
        "metadata",
        "sample_truth",
        "batch_neutral_reference",
    }
    missing = sorted(required - set(tables))
    if missing:
        raise ValueError(f"Benchmark chunk is missing tables: {', '.join(missing)}")
    counts = tables["counts"]
    taxa = _taxa(counts)
    sample_ids = counts["sample_id"].astype(str).tolist()
    if (
        counts["sample_id"].isna().any()
        or counts["sample_id"].astype(str).str.strip().eq("").any()
        or counts["sample_id"].astype(str).duplicated().any()
    ):
        raise ValueError("Benchmark sample IDs must be nonempty and unique.")
    for name in required - {"counts"}:
        frame = tables[name]
        if frame["sample_id"].astype(str).tolist() != sample_ids:
            raise ValueError(f"{name} sample IDs or row order do not match counts.")
    for name in ("relative_abundance", "latent_composition", "batch_neutral_reference"):
        if _taxa(tables[name]) != taxa:
            raise ValueError(
                "Benchmark abundance tables must have the same ordered taxa."
            )
    count_values = counts[taxa].to_numpy(dtype=np.float64)
    depths = count_values.sum(axis=1)
    if (
        np.any(count_values < 0)
        or not np.array_equal(count_values, np.rint(count_values))
        or np.any(depths <= 0)
    ):
        raise ValueError(
            "Benchmark counts must be nonnegative integers with positive depth."
        )
    relative = tables["relative_abundance"][taxa].to_numpy(dtype=np.float64)
    if not np.allclose(relative, count_values / depths[:, None], atol=1e-12, rtol=0):
        raise ValueError("relative_abundance must equal counts divided by row depth.")
    for name in ("latent_composition", "batch_neutral_reference"):
        values = tables[name][taxa].to_numpy(dtype=np.float64)
        if (
            np.any(values < 0)
            or not np.isfinite(values).all()
            or not np.allclose(values.sum(axis=1), 1, atol=1e-10)
        ):
            raise ValueError(f"{name} must be finite, nonnegative, and closed.")
    neutral = tables.get("batch_neutral_counts")
    if neutral is not None:
        if (
            _taxa(neutral) != taxa
            or neutral["sample_id"].astype(str).tolist() != sample_ids
        ):
            raise ValueError("batch_neutral_counts does not match the primary tables.")
        neutral_values = neutral[taxa].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(neutral_values).all()
            or np.any(neutral_values < 0)
            or not np.array_equal(neutral_values, np.rint(neutral_values))
        ):
            raise ValueError("batch_neutral_counts must contain nonnegative integers.")


class BenchmarkStreamWriter:
    """Append validated sample chunks to one benchmark bundle."""

    def __init__(self, root: str | Path, *, paired_neutral_counts: bool) -> None:
        self.root = Path(root)
        names = list(_SAMPLE_TABLE_FILES)
        if not paired_neutral_counts:
            names.remove("batch_neutral_counts")
        self._handles: dict[str, TextIO] = {
            name: gzip.open(
                self.root / _SAMPLE_TABLE_FILES[name],
                "wt",
                encoding="utf-8",
                newline="",
            )
            for name in names
        }
        self._first = True
        self.rows = 0

    def write(self, tables: Mapping[str, pd.DataFrame]) -> None:
        _validate_sample_tables(tables)
        if set(tables) != set(self._handles):
            raise ValueError(
                "Benchmark chunk tables do not match the selected output files."
            )
        lengths = {len(table) for table in tables.values()}
        if len(lengths) != 1:
            raise ValueError("Benchmark chunk tables must have the same row count.")
        for name, handle in self._handles.items():
            tables[name].to_csv(handle, index=False, header=self._first)
        self.rows += lengths.pop()
        self._first = False

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()

    def __enter__(self) -> "BenchmarkStreamWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def write_benchmark_bundle(
    output_dir: str | Path,
    *,
    tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, object],
    overwrite: bool = False,
) -> Path:
    _validate_tables(tables, config)
    with staged_output_dir(output_dir, overwrite=overwrite) as staging:
        tables["counts"].to_csv(staging / "counts.csv.gz", index=False)
        tables["relative_abundance"].to_csv(
            staging / "relative_abundance.csv.gz", index=False
        )
        tables["latent_composition"].to_csv(
            staging / "latent_composition.csv.gz", index=False
        )
        tables["metadata"].to_csv(staging / "metadata.csv.gz", index=False)
        tables["feature_truth"].to_csv(staging / "feature_truth.csv.gz", index=False)
        tables["sample_truth"].to_csv(staging / "sample_truth.csv.gz", index=False)
        tables["batch_neutral_reference"].to_csv(
            staging / "batch_neutral_biological_reference.csv.gz", index=False
        )
        if tables.get("batch_neutral_counts") is not None:
            tables["batch_neutral_counts"].to_csv(
                staging / "batch_neutral_counts.csv.gz", index=False
            )
        write_yaml(staging / "benchmark.yaml", config)
    return Path(output_dir).expanduser().resolve()


def _load_benchmark_bundle(
    path: str | Path, *, source: Mapping[str, object] | None = None
) -> BenchmarkBundle:
    root = Path(path).expanduser().resolve()
    config = load_yaml(root / "benchmark.yaml")
    tables = {
        "counts": pd.read_csv(root / "counts.csv.gz"),
        "relative_abundance": pd.read_csv(root / "relative_abundance.csv.gz"),
        "latent_composition": pd.read_csv(root / "latent_composition.csv.gz"),
        "metadata": pd.read_csv(root / "metadata.csv.gz"),
        "feature_truth": pd.read_csv(root / "feature_truth.csv.gz"),
        "sample_truth": pd.read_csv(root / "sample_truth.csv.gz"),
        "batch_neutral_reference": pd.read_csv(
            root / "batch_neutral_biological_reference.csv.gz"
        ),
    }
    neutral_path = root / "batch_neutral_counts.csv.gz"
    if neutral_path.is_file():
        tables["batch_neutral_counts"] = pd.read_csv(neutral_path)
    _validate_tables(tables, config)
    return BenchmarkBundle(
        root=root,
        config=config,
        counts=tables["counts"],
        relative_abundance=tables["relative_abundance"],
        latent_composition=tables["latent_composition"],
        metadata=tables["metadata"],
        feature_truth=tables["feature_truth"],
        sample_truth=tables["sample_truth"],
        batch_neutral_reference=tables["batch_neutral_reference"],
        batch_neutral_counts=tables.get("batch_neutral_counts"),
        source=None if source is None else dict(source),
    )


def load_benchmark_bundle(path: str | Path) -> BenchmarkBundle:
    """Load and validate an extracted benchmark dataset."""

    return _load_benchmark_bundle(path)
