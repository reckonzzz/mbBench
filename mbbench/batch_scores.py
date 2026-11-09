from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from .batch_metrics import _structure_clr


SCORE_COLUMNS = (
    "batch_global_score",
    "batch_local_score",
    "batch_effect_removal_score",
    "biology_global_preservation_score",
    "biology_local_preservation_score",
    "biology_classifier_preservation_score",
    "biological_signal_preservation_score",
    "overall_score",
)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _bounded_similarity(observed: Any, corrected: Any) -> float | None:
    observed_value = _number(observed)
    corrected_value = _number(corrected)
    if observed_value is None or corrected_value is None:
        return None
    scale = max(abs(observed_value), abs(corrected_value), np.finfo(float).eps)
    return float(
        np.clip(1.0 - abs(corrected_value - observed_value) / scale, 0.0, 1.0)
    )


def _removal_fraction(observed: Any, corrected: Any) -> float | None:
    observed_value = _number(observed)
    corrected_value = _number(corrected)
    if (
        observed_value is None
        or corrected_value is None
        or observed_value <= np.finfo(float).eps
    ):
        return None
    return float(
        np.clip((observed_value - corrected_value) / observed_value, 0.0, 1.0)
    )


def _mixing_improvement(observed: Any, corrected: Any) -> float | None:
    observed_value = _number(observed)
    corrected_value = _number(corrected)
    if observed_value is None or corrected_value is None:
        return None
    denominator = 1.0 - observed_value
    if denominator <= np.finfo(float).eps:
        return None
    return float(
        np.clip((corrected_value - observed_value) / denominator, 0.0, 1.0)
    )


def cross_batch_classifier_metrics(
    representation: np.ndarray,
    labels: Any,
    batches: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    """Fit and evaluate the fixed leave-one-batch-out classifier protocol."""

    values = np.asarray(representation, dtype=np.float64)
    label_series = pd.Series(labels).astype(str).reset_index(drop=True)
    batch_series = pd.Series(batches).astype(str).reset_index(drop=True)
    if values.ndim != 2 or len(values) != len(label_series):
        raise ValueError("Classifier representation and labels have incompatible shapes.")
    if len(batch_series) != len(label_series):
        raise ValueError("Classifier batch and biological labels have different lengths.")
    if not np.isfinite(values).all():
        raise ValueError("Classifier representation contains non-finite values.")

    encoded, levels = pd.factorize(label_series, sort=True)
    counts = np.bincount(encoded)
    if len(levels) < 2 or (len(counts) and counts.min() < 2) or values.shape[1] < 1:
        return {
            "auroc": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "reason": "cross-batch prediction requires at least two replicated biological groups",
        }
    batch_values = batch_series.to_numpy()
    unique_batches = np.unique(batch_values)
    if len(unique_batches) < 2:
        return {
            "auroc": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "reason": "cross-batch prediction requires at least two batches",
        }

    probabilities = np.full((len(encoded), len(levels)), np.nan, dtype=np.float64)
    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    for batch in unique_batches:
        test = batch_values == batch
        train = ~test
        if set(np.unique(encoded[train])) != set(range(len(levels))):
            return {
                "auroc": None,
                "balanced_accuracy": None,
                "macro_f1": None,
                "reason": (
                    f"training data excluding batch {batch!r} do not contain every "
                    "biological group"
                ),
            }
        variances = np.var(values[train], axis=0)
        selected = np.argsort(variances)[-min(100, values.shape[1]) :]
        fitted = model.fit(values[np.ix_(np.flatnonzero(train), selected)], encoded[train])
        fold = fitted.predict_proba(
            values[np.ix_(np.flatnonzero(test), selected)]
        )
        probabilities[np.ix_(np.flatnonzero(test), fitted.classes_.astype(int))] = fold
    if not np.isfinite(probabilities).all():
        raise AssertionError("Leave-one-batch-out prediction did not fill every sample.")

    predictions = np.argmax(probabilities, axis=1)
    auroc = (
        roc_auc_score(encoded, probabilities[:, 1])
        if len(levels) == 2
        else roc_auc_score(encoded, probabilities, multi_class="ovr", average="macro")
    )
    return {
        "auroc": float(auroc),
        "balanced_accuracy": float(balanced_accuracy_score(encoded, predictions)),
        "macro_f1": float(f1_score(encoded, predictions, average="macro")),
        "reason": None,
    }


def publication_lisi_embeddings(
    stage_clr: Mapping[str, np.ndarray], *, seed: int
) -> dict[str, np.ndarray]:
    """Fit the publication LISI PCA on observed CLR and reuse it by stage."""

    required = {"observed", "corrected", "batch_neutral_reference"}
    if set(stage_clr) != required:
        raise ValueError("LISI requires observed, corrected, and reference CLR matrices.")
    observed = np.asarray(stage_clr["observed"], dtype=np.float64)
    keep = np.std(observed, axis=0) > 0
    components = min(20, len(observed) - 1, int(keep.sum()))
    if components < 2:
        raise ValueError("At least two observed CLR-PCA components are required for LISI.")
    pca = PCA(n_components=components, random_state=seed).fit(observed[:, keep])
    return {
        stage: pca.transform(np.asarray(values, dtype=np.float64)[:, keep])
        for stage, values in stage_clr.items()
    }


def _r_lisi_runtime() -> tuple[str, dict[str, str]]:
    rscript = shutil.which("Rscript")
    message = (
        "calculate_scores=True requires R and the R package 'lisi'. Install R, "
        "then run Rscript -e \"install.packages('remotes'); "
        "remotes::install_github('immunogenomics/LISI')\", or calculate scores "
        "with mb_evaluate_batch() in the mbBench R package."
    )
    if rscript is None:
        raise RuntimeError(message)
    check = subprocess.run(
        [
            rscript,
            "-e",
            (
                "if (!requireNamespace('lisi', quietly=TRUE)) quit(status=42); "
                "cat(as.character(getRversion()), '\\t', "
                "as.character(utils::packageVersion('lisi')), sep='')"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(message)
    parts = check.stdout.strip().split("\t")
    if len(parts) != 2:
        raise RuntimeError("R returned an invalid LISI dependency check result.")
    return rscript, {"r_version": parts[0], "lisi_version": parts[1]}


def standard_lisi_metrics(
    embeddings: Mapping[str, np.ndarray], metadata: pd.DataFrame
) -> tuple[dict[str, dict[str, float | None]], dict[str, str]]:
    """Calculate standard LISI endpoints with the original R package."""

    rscript, versions = _r_lisi_runtime()
    required = {"observed", "corrected", "batch_neutral_reference"}
    if set(embeddings) != required:
        raise ValueError("LISI requires observed, corrected, and reference embeddings.")
    if not {"batch", "group"} <= set(metadata.columns):
        raise ValueError("LISI metadata requires batch and group columns.")

    with tempfile.TemporaryDirectory(prefix="mbbench_lisi_") as temporary:
        root = Path(temporary)
        sample_ids = [f"sample_{index}" for index in range(len(metadata))]
        for stage, values in embeddings.items():
            frame = pd.DataFrame(
                np.asarray(values, dtype=np.float64),
                columns=[f"PC{index + 1}" for index in range(values.shape[1])],
            )
            frame.insert(0, "sample_id", sample_ids)
            frame.to_csv(root / f"{stage}.csv", index=False)
        labels = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "batch": metadata["batch"].astype(str).to_numpy(),
                "group": metadata["group"].astype(str).to_numpy(),
            }
        )
        labels.to_csv(root / "metadata.csv", index=False)
        output = root / "lisi.csv"
        script = files("mbbench").joinpath("r", "standard_lisi.R")
        run = subprocess.run(
            [rscript, str(script), str(root), str(output)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if run.returncode != 0:
            detail = (run.stderr or run.stdout).strip()
            raise RuntimeError(f"R LISI calculation failed: {detail}")
        table = pd.read_csv(output)

    required_columns = {
        "stage",
        "batch_ilisi_expected_adjusted_stratified",
        "biological_factor_separation_score",
    }
    missing = sorted(required_columns - set(table.columns))
    if missing:
        raise RuntimeError("R LISI output is missing columns: " + ", ".join(missing) + ".")
    if set(table["stage"].astype(str)) != required or table["stage"].duplicated().any():
        raise RuntimeError("R LISI output has invalid stage identities.")
    result: dict[str, dict[str, float | None]] = {}
    for row in table.itertuples(index=False):
        result[str(row.stage)] = {
            "batch_ilisi": _number(row.batch_ilisi_expected_adjusted_stratified),
            "biology_separation": _number(row.biological_factor_separation_score),
        }
    return result, versions


def _classifier_preservation(
    observed: Mapping[str, Any], corrected: Mapping[str, Any]
) -> float | None:
    components = [
        _bounded_similarity(observed.get(metric), corrected.get(metric))
        for metric in ("auroc", "balanced_accuracy", "macro_f1")
    ]
    return None if any(value is None for value in components) else _mean(components)


def calculate_batch_score_row(
    *,
    observed_counts: np.ndarray,
    corrected: np.ndarray,
    reference: np.ndarray,
    metadata: pd.DataFrame,
    raw_metrics: Mapping[str, Any],
    method: str,
    scenario: str,
    seed: int | None,
    prevalence_min: float,
    maximum_relative_abundance_min: float,
    batch_recovery_applicable: bool,
    biological_preservation_applicable: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Calculate one Figure 4b-compatible score row."""

    base = {
        "method": method,
        "scenario": scenario,
        "seed": seed,
        "batch_recovery_applicable": bool(batch_recovery_applicable),
        "biological_preservation_applicable": bool(
            biological_preservation_applicable
        ),
        "score_definition": "figure4b_publication_v1",
        "classifier_seed": int((42 if seed is None else seed) + 17),
    }
    empty = {column: None for column in SCORE_COLUMNS}
    structure, context = _structure_clr(
        observed_counts,
        corrected,
        reference,
        metadata,
        prevalence_min=prevalence_min,
        maximum_relative_abundance_min=maximum_relative_abundance_min,
    )
    if context is None:
        return {
            **base,
            "score_supported": False,
            "score_reason": structure["structure_metrics_reason"],
            **empty,
        }, {}
    try:
        embeddings = publication_lisi_embeddings(
            context["stage_clr"], seed=42 if seed is None else seed
        )
    except ValueError as error:
        return {
            **base,
            "score_supported": False,
            "score_reason": str(error),
            **empty,
        }, {}

    lisi, versions = standard_lisi_metrics(embeddings, metadata)
    classifier_seed = base["classifier_seed"]
    classifiers: dict[str, dict[str, Any]] = {}
    if biological_preservation_applicable:
        for stage in ("observed", "corrected"):
            classifiers[stage] = cross_batch_classifier_metrics(
                context["stage_clr"][stage],
                context["group"],
                context["batch"],
                seed=classifier_seed,
            )

    batch_global = batch_local = batch_score = None
    if batch_recovery_applicable:
        batch_global = _removal_fraction(
            raw_metrics.get("observed_batch_aitchison_marginal_r2"),
            raw_metrics.get("corrected_batch_aitchison_marginal_r2"),
        )
        observed_asw = _number(raw_metrics.get("observed_batch_asw_stratified"))
        corrected_asw = _number(raw_metrics.get("corrected_batch_asw_stratified"))
        asw_component = _mixing_improvement(
            None if observed_asw is None else 1.0 - abs(observed_asw),
            None if corrected_asw is None else 1.0 - abs(corrected_asw),
        )
        batch_lisi_values = [
            lisi[stage]["batch_ilisi"]
            for stage in ("observed", "corrected", "batch_neutral_reference")
        ]
        lisi_component = (
            _mixing_improvement(batch_lisi_values[0], batch_lisi_values[1])
            if all(value is not None for value in batch_lisi_values)
            else None
        )
        batch_local = (
            None
            if asw_component is None or lisi_component is None
            else _mean([asw_component, lisi_component])
        )
        batch_score = (
            None
            if batch_global is None or batch_local is None
            else _mean([batch_global, batch_local])
        )

    biology_global = biology_local = biology_classifier = biology_score = None
    if biological_preservation_applicable:
        biology_global = _bounded_similarity(
            raw_metrics.get("observed_biology_aitchison_marginal_r2"),
            raw_metrics.get("corrected_biology_aitchison_marginal_r2"),
        )
        biology_lisi_values = [
            lisi[stage]["biology_separation"]
            for stage in ("observed", "corrected", "batch_neutral_reference")
        ]
        biology_local = (
            _bounded_similarity(biology_lisi_values[0], biology_lisi_values[1])
            if all(value is not None for value in biology_lisi_values)
            else None
        )
        biology_classifier = _classifier_preservation(
            classifiers["observed"], classifiers["corrected"]
        )
        biology_components = [biology_global, biology_local, biology_classifier]
        biology_score = (
            None
            if any(value is None for value in biology_components)
            else _mean(biology_components)
        )

    overall = (
        None
        if batch_score is None or biology_score is None
        else float(0.4 * batch_score + 0.6 * biology_score)
    )
    scores = {
        "batch_global_score": batch_global,
        "batch_local_score": batch_local,
        "batch_effect_removal_score": batch_score,
        "biology_global_preservation_score": biology_global,
        "biology_local_preservation_score": biology_local,
        "biology_classifier_preservation_score": biology_classifier,
        "biological_signal_preservation_score": biology_score,
        "overall_score": overall,
    }
    expected = []
    if batch_recovery_applicable:
        expected.extend(
            ["batch_global_score", "batch_local_score", "batch_effect_removal_score"]
        )
    if biological_preservation_applicable:
        expected.extend(
            [
                "biology_global_preservation_score",
                "biology_local_preservation_score",
                "biology_classifier_preservation_score",
                "biological_signal_preservation_score",
            ]
        )
    if batch_recovery_applicable and biological_preservation_applicable:
        expected.append("overall_score")
    missing = [name for name in expected if scores[name] is None]
    reasons = [
        item.get("reason")
        for item in classifiers.values()
        if item.get("reason") is not None
    ]
    if missing:
        reasons.append("unavailable score components: " + ", ".join(missing))
    return {
        **base,
        "score_supported": not missing,
        "score_reason": "; ".join(reasons) if reasons else None,
        **scores,
    }, versions
