from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples


def close_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("Abundance values must form a finite nonnegative matrix.")
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Every abundance row must have positive mass.")
    return matrix / totals


def clr(values: np.ndarray, pseudocount: float) -> np.ndarray:
    closed = close_rows(values)
    logged = np.log(closed + pseudocount)
    return logged - logged.mean(axis=1, keepdims=True)


def _paired_distances(
    candidate: np.ndarray, reference: np.ndarray, pseudocount: float
) -> dict[str, np.ndarray]:
    candidate = close_rows(candidate)
    reference = close_rows(reference)
    if candidate.shape != reference.shape:
        raise ValueError("Candidate and reference matrices must have the same shape.")
    error = clr(candidate, pseudocount) - clr(reference, pseudocount)
    squared = np.square(error).sum(axis=1)
    return {
        "aitchison": np.sqrt(squared),
        "clr_rms": np.sqrt(squared / candidate.shape[1]),
        "bray_curtis": np.abs(candidate - reference).sum(axis=1) / 2.0,
        "hellinger": np.sqrt(
            np.square(np.sqrt(candidate) - np.sqrt(reference)).sum(axis=1)
        )
        / np.sqrt(2.0),
    }


def paired_reference_metrics(
    observed: np.ndarray,
    corrected: np.ndarray,
    reference: np.ndarray,
    sample_ids: list[str],
    *,
    pseudocount: float,
    method: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    observed_distances = _paired_distances(observed, reference, pseudocount)
    corrected_distances = _paired_distances(corrected, reference, pseudocount)
    before = observed_distances["aitchison"]
    after = corrected_distances["aitchison"]
    unchanged = np.isclose(before, after, rtol=1e-10, atol=1e-12)
    closer = (after < before) & ~unchanged
    farther = (after > before) & ~unchanged
    improvement = before - after
    observed_rms = float(np.sqrt(np.mean(np.square(before))))
    corrected_rms = float(np.sqrt(np.mean(np.square(after))))
    pbrs_raw = (
        float(1.0 - corrected_rms**2 / observed_rms**2)
        if observed_rms > np.finfo(float).eps
        else None
    )
    metrics: dict[str, Any] = {
        "relative_pseudocount": float(pseudocount),
        "observed_rms_aitchison": observed_rms,
        "corrected_rms_aitchison": corrected_rms,
        "mean_paired_aitchison_improvement": float(np.mean(improvement)),
        "fraction_samples_closer": float(np.mean(closer)),
        "fraction_samples_unchanged": float(np.mean(unchanged)),
        "fraction_samples_farther": float(np.mean(farther)),
        "pbrs_raw": pbrs_raw,
        "pbrs": None if pbrs_raw is None else float(np.clip(pbrs_raw, 0.0, 1.0)),
    }
    for name in ("clr_rms", "bray_curtis", "hellinger"):
        metrics[f"observed_mean_{name}"] = float(
            np.mean(observed_distances[name])
        )
        metrics[f"corrected_mean_{name}"] = float(
            np.mean(corrected_distances[name])
        )
    sample_metrics = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "method": method,
            "observed_aitchison": before,
            "corrected_aitchison": after,
            "aitchison_improvement": improvement,
            "observed_clr_rms": observed_distances["clr_rms"],
            "corrected_clr_rms": corrected_distances["clr_rms"],
            "observed_bray_curtis": observed_distances["bray_curtis"],
            "corrected_bray_curtis": corrected_distances["bray_curtis"],
            "observed_hellinger": observed_distances["hellinger"],
            "corrected_hellinger": corrected_distances["hellinger"],
            "status": np.select(
                [closer, farther], ["closer", "farther"], default="unchanged"
            ),
        }
    )
    return metrics, sample_metrics


def _factor_matrix(values: pd.Series) -> np.ndarray:
    text = values.astype(str)
    levels = sorted(text.unique().tolist())
    if len(levels) < 2:
        return np.empty((len(values), 0), dtype=float)
    return np.column_stack(
        [text.eq(level).to_numpy(dtype=float) for level in levels[1:]]
    )


def _design(*variables: pd.Series) -> np.ndarray:
    columns = [np.ones(len(variables[0]), dtype=float)]
    for values in variables:
        encoded = _factor_matrix(values)
        columns.extend(encoded[:, index] for index in range(encoded.shape[1]))
    return np.column_stack(columns)


def _projection(design: np.ndarray) -> np.ndarray:
    return design @ np.linalg.pinv(design)


def _marginal_distance_r2(
    distance: np.ndarray, reduced: np.ndarray, full: np.ndarray
) -> float | None:
    n = distance.shape[0]
    center = np.eye(n) - np.ones((n, n)) / n
    gower = -0.5 * center @ np.square(distance) @ center
    total = float(np.trace(gower))
    if not np.isfinite(total) or total <= 0:
        return None
    effect = float(np.trace((_projection(full) - _projection(reduced)) @ gower))
    return float(max(0.0, effect) / total)


def _stratified_batch_asw(
    embedding: np.ndarray, batch: pd.Series, group: pd.Series
) -> tuple[float | None, str | None]:
    values: list[float] = []
    weights: list[int] = []
    for level in group.astype(str).unique():
        keep = group.astype(str).eq(level).to_numpy()
        labels = batch.loc[keep].astype(str).to_numpy()
        _, counts = np.unique(labels, return_counts=True)
        if len(counts) < 2 or counts.min() < 2:
            return None, f"batch ASW is not estimable in biological group {level!r}"
        values.append(float(np.mean(silhouette_samples(embedding[keep], labels))))
        weights.append(int(keep.sum()))
    return float(np.average(values, weights=weights)), None


def _structure_clr(
    observed_counts: np.ndarray,
    corrected: np.ndarray,
    reference: np.ndarray,
    metadata: pd.DataFrame,
    *,
    prevalence_min: float,
    maximum_relative_abundance_min: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    missing = sorted({"group", "batch"} - set(metadata.columns))
    if missing:
        raise ValueError(
            "Benchmark metadata is missing required columns: " + ", ".join(missing) + "."
        )
    observed_relative = close_rows(observed_counts)
    active = (observed_counts > 0).mean(axis=0) >= prevalence_min
    active &= observed_relative.max(axis=0) >= maximum_relative_abundance_min
    result: dict[str, Any] = {
        "structure_prevalence_min": float(prevalence_min),
        "structure_maximum_relative_abundance_min": float(
            maximum_relative_abundance_min
        ),
        "structure_n_active_taxa": int(active.sum()),
        "structure_metrics_supported": False,
        "structure_metrics_reason": None,
    }
    if active.sum() < 2:
        result["structure_metrics_reason"] = (
            "fewer than two taxa pass the R2/ASW feature filter"
        )
        return result, None
    stages = {
        "observed": np.asarray(observed_counts, dtype=np.float64)[:, active],
        "corrected": np.asarray(corrected, dtype=np.float64)[:, active],
        "batch_neutral_reference": np.asarray(reference, dtype=np.float64)[:, active],
    }
    if any(np.any(matrix.sum(axis=1) <= 0) for matrix in stages.values()):
        result["structure_metrics_reason"] = (
            "the R2/ASW feature filter leaves at least one zero-mass sample"
        )
        return result, None
    pseudocount = max(
        0.5 / float(np.median(stages["observed"].sum(axis=1))), 1e-12
    )
    return result, {
        "stage_clr": {
            stage: clr(matrix, pseudocount) for stage, matrix in stages.items()
        },
        "batch": metadata["batch"].astype(str).reset_index(drop=True),
        "group": metadata["group"].astype(str).reset_index(drop=True),
        "pseudocount": float(pseudocount),
    }


def _shared_structure_embeddings(
    stage_clr: dict[str, np.ndarray], n_samples: int
) -> tuple[dict[str, np.ndarray] | None, int, str | None]:
    keep = np.std(stage_clr["observed"], axis=0) > 0
    n_components = min(20, n_samples - 1, int(keep.sum()))
    if n_components < 2:
        return (
            None,
            n_components,
            "fewer than two nonconstant observed CLR features remain for ASW",
        )
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(stage_clr["observed"][:, keep])
    return (
        {
            stage: pca.transform(representation[:, keep])
            for stage, representation in stage_clr.items()
        },
        n_components,
        None,
    )


def community_structure_metrics(
    observed_counts: np.ndarray,
    corrected: np.ndarray,
    reference: np.ndarray,
    metadata: pd.DataFrame,
    *,
    prevalence_min: float,
    maximum_relative_abundance_min: float,
) -> dict[str, Any]:
    result, context = _structure_clr(
        observed_counts,
        corrected,
        reference,
        metadata,
        prevalence_min=prevalence_min,
        maximum_relative_abundance_min=maximum_relative_abundance_min,
    )
    if context is None:
        return result
    stage_clr = context["stage_clr"]
    batch = context["batch"]
    group = context["group"]
    biology_design = _design(group)
    batch_design = _design(batch)
    full_design = _design(group, batch)
    for stage, representation in stage_clr.items():
        distance = squareform(pdist(representation, metric="euclidean"))
        result[f"{stage}_batch_aitchison_marginal_r2"] = _marginal_distance_r2(
            distance, biology_design, full_design
        )
        result[f"{stage}_biology_aitchison_marginal_r2"] = _marginal_distance_r2(
            distance, batch_design, full_design
        )
    observed_biology = result["observed_biology_aitchison_marginal_r2"]
    corrected_biology = result["corrected_biology_aitchison_marginal_r2"]
    result["biology_aitchison_r2_change"] = (
        None
        if observed_biology is None or corrected_biology is None
        else float(corrected_biology - observed_biology)
    )
    observed_batch = result["observed_batch_aitchison_marginal_r2"]
    corrected_batch = result["corrected_batch_aitchison_marginal_r2"]
    result["batch_aitchison_r2_change"] = (
        None
        if observed_batch is None or corrected_batch is None
        else float(observed_batch - corrected_batch)
    )
    embeddings, n_components, reason = _shared_structure_embeddings(
        stage_clr, len(metadata)
    )
    if embeddings is None:
        result["structure_metrics_reason"] = reason
        return result
    for stage, embedding in embeddings.items():
        asw, reason = _stratified_batch_asw(embedding, batch, group)
        result[f"{stage}_batch_asw_stratified"] = asw
        if reason is not None:
            result["structure_metrics_reason"] = reason
    result["batch_asw_change"] = (
        None
        if result.get("observed_batch_asw_stratified") is None
        or result.get("corrected_batch_asw_stratified") is None
        else float(
            result["observed_batch_asw_stratified"]
            - result["corrected_batch_asw_stratified"]
        )
    )
    result["structure_relative_pseudocount"] = context["pseudocount"]
    result["structure_pca_components"] = int(n_components)
    result["structure_metrics_supported"] = result["structure_metrics_reason"] is None
    return result
