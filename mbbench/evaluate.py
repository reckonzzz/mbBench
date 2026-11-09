from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .common import staged_output_dir


def _read_abundance(path: str | Path) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    feature_columns = [
        column for column in frame if column not in {"sample_id", "Sample"}
    ]
    values = (
        frame[feature_columns]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float64)
    )
    if (
        not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values.sum(axis=1) <= 0)
    ):
        raise ValueError(f"Invalid abundance table: {path}")
    return values / values.sum(axis=1, keepdims=True), feature_columns


def _aitchison(values: np.ndarray, pseudocount: float) -> np.ndarray:
    logged = np.log(values + pseudocount)
    return logged - logged.mean(axis=1, keepdims=True)


def _mean_distance(
    left: np.ndarray, right: np.ndarray, metric: str, block_size: int = 512
) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(left), block_size):
        distances = cdist(left[start : start + block_size], right, metric=metric)
        total += float(distances.sum(dtype=np.float64))
        count += distances.size
    return total / count


def _energy_distance(real: np.ndarray, synthetic: np.ndarray, metric: str) -> float:
    between = _mean_distance(real, synthetic, metric)
    within_real = _mean_distance(real, real, metric)
    within_synthetic = _mean_distance(synthetic, synthetic, metric)
    return float(max(2 * between - within_real - within_synthetic, 0.0))


def evaluate_generation(
    *,
    real_abundance: str | Path,
    synthetic_abundance: str | Path,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, float]:
    """Compute summary distances between real and generated compositions."""

    real, real_taxa = _read_abundance(real_abundance)
    synthetic, synthetic_taxa = _read_abundance(synthetic_abundance)
    if real_taxa != synthetic_taxa:
        raise ValueError(
            "Real and synthetic abundance tables must have identical taxon columns and order."
        )
    real_prevalence = (real > 0).mean(axis=0)
    synthetic_prevalence = (synthetic > 0).mean(axis=0)
    positive = np.r_[real[real > 0], synthetic[synthetic > 0]]
    pseudocount = max(float(positive.min()) * 0.5, 1e-12)
    metrics = {
        "real_shannon_mean": float(
            (-(real * np.log(real.clip(min=1e-300)))).sum(axis=1).mean()
        ),
        "synthetic_shannon_mean": float(
            (-(synthetic * np.log(synthetic.clip(min=1e-300)))).sum(axis=1).mean()
        ),
        "real_richness_mean": float((real > 0).sum(axis=1).mean()),
        "synthetic_richness_mean": float((synthetic > 0).sum(axis=1).mean()),
        "prevalence_mae": float(np.abs(real_prevalence - synthetic_prevalence).mean()),
        "mean_abundance_mae": float(
            np.abs(real.mean(axis=0) - synthetic.mean(axis=0)).mean()
        ),
        "bray_curtis_energy": _energy_distance(real, synthetic, "braycurtis"),
        "aitchison_energy": _energy_distance(
            _aitchison(real, pseudocount),
            _aitchison(synthetic, pseudocount),
            "euclidean",
        ),
        "aitchison_pseudocount": pseudocount,
    }
    if output_dir is not None:
        with staged_output_dir(output_dir, overwrite=overwrite) as output:
            (output / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
            pd.DataFrame(
                {
                    "taxon": real_taxa,
                    "real_prevalence": real_prevalence,
                    "synthetic_prevalence": synthetic_prevalence,
                    "real_mean_abundance": real.mean(axis=0),
                    "synthetic_mean_abundance": synthetic.mean(axis=0),
                }
            ).to_csv(output / "taxon_summary.csv", index=False)
    return metrics
