from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .batch_metrics import close_rows, community_structure_metrics, paired_reference_metrics
from .benchmark_bundle import BenchmarkBundle, load_benchmark_bundle
from .common import require_bool, staged_output_dir, write_yaml
from .da_metrics import align_da_input, evaluate_calls


@dataclass(frozen=True)
class BatchEvaluation:
    """Results from one batch evaluation.

    ``metrics`` has one row per method; ``sample_metrics`` is long by method and
    sample; ``scores`` is the optional one-row-per-method score table.
    ``applicability`` and ``config`` record interpretation settings, while
    ``output_dir`` and ``files`` are empty when nothing was written.
    """

    metrics: pd.DataFrame
    sample_metrics: pd.DataFrame
    scores: pd.DataFrame | None
    applicability: dict[str, Any]
    config: dict[str, Any]
    output_dir: Path | None
    files: dict[str, Path]


@dataclass(frozen=True)
class BatchSummary:
    """Repeated batch metrics and scores.

    ``replicate_metrics`` and ``replicate_scores`` are wide repetition tables.
    ``metric_summary`` and ``score_summary`` are long tables by method,
    scenario, and metric. Output paths are empty when ``output_dir`` was omitted.
    """

    metric_summary: pd.DataFrame
    score_summary: pd.DataFrame | None
    replicate_metrics: pd.DataFrame
    replicate_scores: pd.DataFrame | None
    output_dir: Path | None
    files: dict[str, Path]


@dataclass(frozen=True)
class DAEvaluation:
    """Results from one or many DA evaluations.

    ``metrics`` has one row per method and repetition; ``taxon_results`` is long
    by method and taxon; ``summary`` is present for repeated-run evaluation.
    The remaining fields record truth applicability, configuration, and files.
    """

    metrics: pd.DataFrame
    taxon_results: pd.DataFrame
    applicability: dict[str, Any]
    config: dict[str, Any]
    output_dir: Path | None
    files: dict[str, Path]
    summary: pd.DataFrame | None = None


def _benchmark_bundle(value: Any) -> BenchmarkBundle:
    if isinstance(value, BenchmarkBundle):
        return value
    if isinstance(value, (str, Path)):
        return load_benchmark_bundle(value)
    output_dir = getattr(value, "output_dir", None)
    required = (
        "counts",
        "relative_abundance",
        "metadata",
        "feature_truth",
        "sample_truth",
        "batch_neutral_reference",
        "config",
    )
    if all(getattr(value, name, None) is not None for name in required):
        return BenchmarkBundle(
            root=Path(output_dir).resolve() if output_dir is not None else Path("."),
            config=dict(value.config),
            counts=value.counts,
            relative_abundance=value.relative_abundance,
            latent_composition=value.latent_composition,
            metadata=value.metadata,
            feature_truth=value.feature_truth,
            sample_truth=value.sample_truth,
            batch_neutral_reference=value.batch_neutral_reference,
            batch_neutral_counts=value.batch_neutral_counts,
            source={"type": "generated_result"},
        )
    if output_dir is not None:
        return load_benchmark_bundle(output_dir)
    raise TypeError(
        "benchmark must be a BenchmarkBundle, benchmark result, or bundle directory."
    )


def _identity(bundle: BenchmarkBundle) -> tuple[str, int | None]:
    scenario = str(bundle.config.get("preset") or bundle.config.get("scenario_id") or "custom")
    runtime = bundle.config.get("runtime", {})
    seed = runtime.get("seed") if isinstance(runtime, Mapping) else None
    return scenario, None if seed is None else int(seed)


def _read_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (str, Path)):
        return pd.read_csv(Path(value).expanduser())
    raise TypeError("input must be a pandas DataFrame or CSV path.")


def _method_inputs(value: Any, default_method: str) -> dict[str, Any]:
    if isinstance(value, Mapping) and not isinstance(value, pd.DataFrame):
        methods = [str(method).strip() for method in value]
        if not methods or any(not method for method in methods):
            raise ValueError("Method names must be nonempty strings.")
        if len(set(methods)) != len(methods):
            raise ValueError("Method names must be unique after conversion to strings.")
        return dict(zip(methods, value.values()))
    return {default_method: value}


def _aligned_abundance(
    value: Any, sample_ids: list[str], taxa: list[str]
) -> np.ndarray:
    frame = _read_frame(value)
    if "sample_id" in frame:
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("Batch input sample IDs must be unique.")
        frame = frame.set_index(frame["sample_id"].astype(str)).drop(columns="sample_id")
    else:
        frame.index = frame.index.astype(str)
        if frame.index.duplicated().any():
            raise ValueError("Batch input sample IDs must be unique.")
    unknown_samples = sorted(set(frame.index) - set(sample_ids))
    missing_samples = sorted(set(sample_ids) - set(frame.index))
    unknown_taxa = sorted(set(map(str, frame.columns)) - set(taxa))
    missing_taxa = sorted(set(taxa) - set(map(str, frame.columns)))
    if unknown_samples or missing_samples:
        raise ValueError(
            "Batch input sample IDs do not match the benchmark; "
            f"missing={missing_samples[:5]}, extra={unknown_samples[:5]}."
        )
    if unknown_taxa or missing_taxa:
        raise ValueError(
            "Batch input taxa do not match the benchmark; "
            f"missing={missing_taxa[:5]}, extra={unknown_taxa[:5]}."
        )
    frame.columns = frame.columns.astype(str)
    values = frame.loc[sample_ids, taxa].to_numpy(dtype=np.float64)
    close_rows(values)
    return values


def _da_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        return pd.DataFrame(
            {"taxon": value.index.astype(str), "significant": value.to_numpy()}
        )
    if isinstance(value, Mapping) and all(
        isinstance(item, (bool, np.bool_)) for item in value.values()
    ):
        return pd.DataFrame(
            {"taxon": list(map(str, value.keys())), "significant": list(value.values())}
        )
    return _read_frame(value)


def _write_batch(
    output_dir: str | Path,
    metrics: pd.DataFrame,
    sample_metrics: pd.DataFrame,
    scores: pd.DataFrame | None,
    config: Mapping[str, Any],
    overwrite: bool,
) -> tuple[Path, dict[str, Path]]:
    output = Path(output_dir).expanduser().resolve()
    with staged_output_dir(output, overwrite=overwrite) as staging:
        metrics.to_csv(staging / "batch_metrics.csv", index=False)
        sample_metrics.to_csv(staging / "batch_sample_metrics.csv.gz", index=False)
        if scores is not None:
            scores.to_csv(staging / "batch_scores.csv", index=False)
        write_yaml(staging / "evaluation.yaml", config)
    files = {
        "metrics": output / "batch_metrics.csv",
        "sample_metrics": output / "batch_sample_metrics.csv.gz",
        "config": output / "evaluation.yaml",
    }
    if scores is not None:
        files["scores"] = output / "batch_scores.csv"
    return output, files


def _write_da(
    output_dir: str | Path,
    metrics: pd.DataFrame,
    taxon_results: pd.DataFrame,
    config: Mapping[str, Any],
    overwrite: bool,
    summary: pd.DataFrame | None = None,
) -> tuple[Path, dict[str, Path]]:
    output = Path(output_dir).expanduser().resolve()
    with staged_output_dir(output, overwrite=overwrite) as staging:
        metrics.to_csv(staging / "da_metrics.csv", index=False)
        taxon_results.to_csv(staging / "da_taxon_results.csv.gz", index=False)
        if summary is not None:
            summary.to_csv(staging / "summary.csv", index=False)
        write_yaml(staging / "evaluation.yaml", config)
    files = {
        "metrics": output / "da_metrics.csv",
        "taxon_results": output / "da_taxon_results.csv.gz",
        "config": output / "evaluation.yaml",
    }
    if summary is not None:
        files["summary"] = output / "summary.csv"
    return output, files


def evaluate_batch(
    benchmark: Any,
    input: Any,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    prevalence_min: float = 0.10,
    maximum_relative_abundance_min: float = 0.002,
    calculate_scores: bool = False,
) -> BatchEvaluation:
    """Evaluate one or more corrected abundance matrices.

    ``benchmark`` may be a loaded bundle, generated benchmark, or bundle path.
    ``input`` is a sample-by-taxon DataFrame/CSV, or ``{method: table}``. Tables
    require unique sample labels and every benchmark taxon; ordering is aligned
    automatically. The returned ``BatchEvaluation`` contains one metrics row per
    method, long-form sample metrics, and ``scores`` only when
    ``calculate_scores=True``. Supplying ``output_dir`` also writes these tables.
    """

    overwrite = require_bool(overwrite, "overwrite")
    if not isinstance(calculate_scores, (bool, np.bool_)):
        raise TypeError("calculate_scores must be True or False.")
    if not 0 <= prevalence_min <= 1:
        raise ValueError("prevalence_min must lie in [0, 1].")
    if maximum_relative_abundance_min < 0:
        raise ValueError("maximum_relative_abundance_min cannot be negative.")
    bundle = _benchmark_bundle(benchmark)
    taxa = [str(column) for column in bundle.counts.columns if column != "sample_id"]
    sample_ids = bundle.counts["sample_id"].astype(str).tolist()
    observed_counts = bundle.counts[taxa].to_numpy(dtype=np.float64)
    observed = bundle.relative_abundance[taxa].to_numpy(dtype=np.float64)
    reference = bundle.batch_neutral_reference[taxa].to_numpy(dtype=np.float64)
    pseudocount = max(0.5 / float(np.median(observed_counts.sum(axis=1))), 1e-12)
    scenario, seed = _identity(bundle)
    applicability = dict(bundle.config.get("truth_applicability", {}))
    metric_rows: list[dict[str, Any]] = []
    sample_rows: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []
    score_versions: dict[str, str] = {}
    inputs = _method_inputs(input, "method")
    for method, value in inputs.items():
        corrected = _aligned_abundance(value, sample_ids, taxa)
        paired, samples = paired_reference_metrics(
            observed,
            corrected,
            reference,
            sample_ids,
            pseudocount=pseudocount,
            method=method,
        )
        structure = community_structure_metrics(
            observed_counts,
            corrected,
            reference,
            bundle.metadata,
            prevalence_min=prevalence_min,
            maximum_relative_abundance_min=maximum_relative_abundance_min,
        )
        metric_row = {
            "method": method,
            "scenario": scenario,
            "seed": seed,
            "batch_recovery_applicable": bool(
                applicability.get("batch_recovery", False)
            ),
            "biological_preservation_applicable": bool(
                applicability.get("biological_preservation", False)
            ),
            **paired,
            **structure,
        }
        metric_rows.append(metric_row)
        if calculate_scores:
            from .batch_scores import calculate_batch_score_row

            score_row, versions = calculate_batch_score_row(
                observed_counts=observed_counts,
                corrected=corrected,
                reference=reference,
                metadata=bundle.metadata,
                raw_metrics=metric_row,
                method=method,
                scenario=scenario,
                seed=seed,
                prevalence_min=prevalence_min,
                maximum_relative_abundance_min=maximum_relative_abundance_min,
                batch_recovery_applicable=metric_row["batch_recovery_applicable"],
                biological_preservation_applicable=metric_row[
                    "biological_preservation_applicable"
                ],
            )
            score_rows.append(score_row)
            if versions:
                if score_versions and versions != score_versions:
                    raise RuntimeError("R or LISI version changed during evaluation.")
                score_versions = versions
        samples.insert(2, "scenario", scenario)
        samples.insert(3, "seed", seed)
        sample_rows.append(samples)
    metrics = pd.DataFrame(metric_rows)
    sample_metrics = pd.concat(sample_rows, ignore_index=True)
    scores = pd.DataFrame(score_rows) if calculate_scores else None
    config = {
        "schema_version": "mbbench_batch_evaluation_v1",
        "benchmark": {"path": str(bundle.root), "scenario": scenario, "seed": seed},
        "methods": list(inputs),
        "paired_metrics_use_full_taxon_roster": True,
        "structure_feature_filter": {
            "prevalence_min": prevalence_min,
            "maximum_relative_abundance_min": maximum_relative_abundance_min,
            "applies_to": [
                "aitchison_marginal_r2",
                "batch_asw",
                *(
                    ["score_lisi", "score_classifier"]
                    if calculate_scores
                    else []
                ),
            ],
            "does_not_apply_to": ["paired_recovery", "da_evaluation"],
        },
        "truth_applicability": applicability,
        "calculate_scores": bool(calculate_scores),
    }
    if calculate_scores:
        config["scores"] = {
            "definition": "figure4b_publication_v1",
            "file": "batch_scores.csv",
            "r_version": score_versions.get("r_version"),
            "lisi_version": score_versions.get("lisi_version"),
            "classifier": "leave_one_batch_out_logistic_regression",
        }
    output = None
    files: dict[str, Path] = {}
    if output_dir is not None:
        output, files = _write_batch(
            output_dir, metrics, sample_metrics, scores, config, overwrite
        )
    return BatchEvaluation(
        metrics, sample_metrics, scores, applicability, config, output, files
    )


def da_result(
    benchmark: Any,
    *,
    significant_taxa: Sequence[str],
    tested_taxa: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create a complete DA decision table from significant and tested taxon names."""

    bundle = _benchmark_bundle(benchmark)
    taxa = bundle.feature_truth["taxon"].astype(str).tolist()
    significant = set(map(str, significant_taxa))
    tested = set(taxa) if tested_taxa is None else set(map(str, tested_taxa))
    unknown = sorted((significant | tested) - set(taxa))
    if unknown:
        raise ValueError(f"Unknown taxa: {', '.join(unknown[:5])}.")
    if not significant <= tested:
        raise ValueError("Every significant taxon must also be tested.")
    return pd.DataFrame(
        {
            "taxon": taxa,
            "tested": [taxon in tested for taxon in taxa],
            "significant": [taxon in significant for taxon in taxa],
        }
    )


def _da_method_inputs(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and not isinstance(value, pd.DataFrame):
        if not value:
            raise ValueError("DA input cannot be empty.")
        if all(isinstance(item, (bool, np.bool_)) for item in value.values()):
            return {"method": value}
        return _method_inputs(value, "method")
    return {"method": value}


def evaluate_da(
    benchmark: Any,
    input: Any,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> DAEvaluation:
    """Evaluate one or more taxon-level DA decision tables.

    ``input`` accepts a Series or Boolean mapping keyed by taxon, a DataFrame/CSV
    with ``taxon`` and ``significant`` columns, or ``{method: table}``. Every
    benchmark taxon must occur once. Optional columns are ``tested``, ``effect``,
    ``p_value``, and ``q_value``. The result contains one metrics row per method
    and a long-form taxon outcome table.
    """

    overwrite = require_bool(overwrite, "overwrite")
    bundle = _benchmark_bundle(benchmark)
    taxa = bundle.feature_truth["taxon"].astype(str).tolist()
    applicability = dict(bundle.config.get("truth_applicability", {}))
    scenario, seed = _identity(bundle)
    metric_rows: list[dict[str, Any]] = []
    taxon_rows: list[pd.DataFrame] = []
    inputs = _da_method_inputs(input)
    for method, value in inputs.items():
        calls = align_da_input(_da_frame(value), taxa, method=method)
        metrics, table = evaluate_calls(calls, bundle.feature_truth, applicability)
        metric_rows.append(
            {"method": method, "scenario": scenario, "seed": seed, **metrics}
        )
        table.insert(2, "scenario", scenario)
        table.insert(3, "seed", seed)
        taxon_rows.append(table)
    metrics = pd.DataFrame(metric_rows)
    taxon_results = pd.concat(taxon_rows, ignore_index=True)
    config = {
        "schema_version": "mbbench_da_evaluation_v1",
        "benchmark": {"path": str(bundle.root), "scenario": scenario, "seed": seed},
        "methods": list(inputs),
        "truth_applicability": applicability,
        "full_taxon_truth_roster_used": True,
    }
    output = None
    files: dict[str, Path] = {}
    if output_dir is not None:
        output, files = _write_da(output_dir, metrics, taxon_results, config, overwrite)
    return DAEvaluation(metrics, taxon_results, applicability, config, output, files)


def evaluate_da_many(
    manifest: str | Path | pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    bootstrap_repetitions: int = 2000,
    seed: int = 42,
) -> DAEvaluation:
    """Evaluate a DA manifest and summarize independent repetitions.

    ``manifest`` is a DataFrame or CSV with ``benchmark``, ``input``, and
    ``method`` columns. Relative paths in a CSV are resolved from that file's
    directory. The returned ``DAEvaluation.summary`` has one row per
    method/scenario.
    """

    overwrite = require_bool(overwrite, "overwrite")
    manifest_path = None if isinstance(manifest, pd.DataFrame) else Path(manifest).expanduser().resolve()
    frame = manifest.copy() if manifest_path is None else pd.read_csv(manifest_path)
    required = {"benchmark", "input", "method"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"DA manifest is missing columns: {', '.join(missing)}.")
    if frame.empty:
        raise ValueError("DA manifest cannot be empty.")
    for column in required:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"DA manifest column {column!r} cannot contain missing values.")
    if manifest_path is not None:
        base = manifest_path.parent
        for column in ("benchmark", "input"):
            frame[column] = frame[column].map(
                lambda value: str((base / str(value)).resolve())
                if not Path(str(value)).expanduser().is_absolute()
                else str(Path(str(value)).expanduser().resolve())
            )
    evaluations = [
        evaluate_da(row.benchmark, {str(row.method): row.input})
        for row in frame.itertuples(index=False)
    ]
    metrics = pd.concat([item.metrics for item in evaluations], ignore_index=True)
    if metrics.duplicated(["method", "scenario", "seed"]).any():
        raise ValueError("DA manifest repeats a method, scenario, and seed identity.")
    taxa = pd.concat([item.taxon_results for item in evaluations], ignore_index=True)
    applicability = {"varies_by_benchmark": True}
    config = {
        "schema_version": "mbbench_da_evaluation_collection_v1",
        "n_runs": int(len(metrics)),
        "manifest": None if manifest_path is None else str(manifest_path),
        "summary": {
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_seed": seed,
        },
    }
    summary = _summarize_da_frame(
        metrics, bootstrap_repetitions=bootstrap_repetitions, seed=seed
    )
    output = None
    files: dict[str, Path] = {}
    if output_dir is not None:
        output, files = _write_da(
            output_dir, metrics, taxa, config, overwrite, summary=summary
        )
    return DAEvaluation(metrics, taxa, applicability, config, output, files, summary)


def _metric_frames(value: Any, filename: str) -> pd.DataFrame:
    if isinstance(value, (BatchEvaluation, DAEvaluation)):
        return value.metrics.copy()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, (list, tuple)):
        frames = [_metric_frames(item, filename) for item in value]
        return pd.concat(frames, ignore_index=True)
    path = Path(value).expanduser().resolve()
    if path.is_file():
        if filename == "batch_metrics.csv" and path.name == "batch_scores.csv":
            raise ValueError(
                "batch_scores.csv cannot be summarized without its batch_metrics.csv. "
                "Pass the evaluation directory or batch_metrics.csv instead."
            )
        return pd.read_csv(path)
    files = sorted(path.rglob(filename))
    if not files:
        raise ValueError(f"No {filename} files were found below {path}.")
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)


def _batch_score_frames(value: Any) -> pd.DataFrame | None:
    if isinstance(value, BatchEvaluation):
        return None if value.scores is None else value.scores.copy()
    if isinstance(value, DAEvaluation) or isinstance(value, pd.DataFrame):
        return None
    if isinstance(value, (list, tuple)):
        frames = [_batch_score_frames(item) for item in value]
        present = [frame for frame in frames if frame is not None]
        if present and len(present) != len(frames):
            raise ValueError("Batch summary input mixes scored and unscored evaluations.")
        return pd.concat(present, ignore_index=True) if present else None
    path = Path(value).expanduser().resolve()
    if path.is_file():
        candidate = path if path.name == "batch_scores.csv" else path.with_name("batch_scores.csv")
        return pd.read_csv(candidate) if candidate.is_file() else None
    files = sorted(path.rglob("batch_scores.csv"))
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True) if files else None


def _require_metric_columns(
    metrics: pd.DataFrame, required: set[str], label: str
) -> None:
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"{label} metrics are missing columns: {', '.join(missing)}.")


def _wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return float(max(0, center - margin)), float(min(1, center + margin))


def _bootstrap_mean_interval(
    values: pd.Series, repetitions: int, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(clean):
        return None, None
    indices = rng.integers(0, len(clean), size=(repetitions, len(clean)))
    means = clean[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _bootstrap_rng(seed: int, *identity: object) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    state = int(seed) & 0xFFFFFFFF
    for byte in "\x1f".join(map(str, identity)).encode("utf-8"):
        state = (state * 1_664_525 + byte + 1_013_904_223) & 0xFFFFFFFF
    return np.random.default_rng(state)


def _summarize_da_frame(
    metrics: pd.DataFrame, *, bootstrap_repetitions: int, seed: int
) -> pd.DataFrame:
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive.")
    _require_metric_columns(
        metrics, {"method", "scenario", "seed", "da_truth_mode"}, "DA"
    )
    if metrics.duplicated(["method", "scenario", "seed"]).any():
        raise ValueError("DA metrics repeat a method, scenario, and seed identity.")
    rows: list[dict[str, Any]] = []
    for (method, scenario), group in metrics.groupby(["method", "scenario"], sort=True):
        group = group.assign(_seed_order=group["seed"].astype(str)).sort_values(
            "_seed_order", kind="mergesort"
        )
        truth_modes = group["da_truth_mode"].dropna().astype(str).unique()
        if len(truth_modes) != 1:
            raise ValueError(
                f"DA truth mode varies within method={method!r}, scenario={scenario!r}."
            )
        row: dict[str, Any] = {
            "method": method,
            "scenario": scenario,
            "n_repetitions": int(len(group)),
            "da_truth_mode": truth_modes[0],
        }
        for source, target in (
            ("fdp", "empirical_fdr"),
            ("direct_target_power", "mean_power"),
            ("type_i_error_rate", "mean_type_i_error_rate"),
            ("type_ii_error_rate", "mean_type_ii_error_rate"),
        ):
            values = (
                pd.to_numeric(group[source], errors="coerce").dropna()
                if source in group
                else pd.Series(dtype=float)
            )
            row[target] = float(values.mean()) if len(values) else None
            rng = _bootstrap_rng(seed, method, scenario, target)
            lower, upper = _bootstrap_mean_interval(values, bootstrap_repetitions, rng)
            row[f"{target}_ci_lower"] = lower
            row[f"{target}_ci_upper"] = upper
        indicators = group.get("fwer_indicator")
        if indicators is not None:
            valid = indicators.dropna().astype(bool)
        else:
            valid = pd.Series(dtype=bool)
        row["fwer"] = float(valid.mean()) if len(valid) else None
        lower, upper = _wilson(int(valid.sum()), int(len(valid)))
        row["fwer_ci_lower"] = lower
        row["fwer_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_da(
    input: Any,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    bootstrap_repetitions: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Return one DA summary row per method and scenario.

    ``input`` may be an evaluation, metrics DataFrame/CSV, directory containing
    ``da_metrics.csv`` files, or a list of those values. Confidence intervals use
    stable, identity-specific bootstrap streams. ``output_dir`` is optional.
    """

    overwrite = require_bool(overwrite, "overwrite")
    metrics = _metric_frames(input, "da_metrics.csv")
    summary = _summarize_da_frame(
        metrics, bootstrap_repetitions=bootstrap_repetitions, seed=seed
    )
    if output_dir is not None:
        output = Path(output_dir).expanduser().resolve()
        with staged_output_dir(output, overwrite=overwrite) as staging:
            metrics.to_csv(staging / "replicate_metrics.csv", index=False)
            summary.to_csv(staging / "summary.csv", index=False)
            write_yaml(
                staging / "summary.yaml",
                {
                    "schema_version": "mbbench_da_summary_v1",
                    "bootstrap_repetitions": bootstrap_repetitions,
                    "bootstrap_seed": seed,
                },
            )
    return summary


def summarize_batch(
    input: Any,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> BatchSummary:
    """Summarize repeated batch evaluations.

    ``input`` may be an evaluation, metrics DataFrame/CSV, directory containing
    ``batch_metrics.csv`` files, or a list. A metrics file also discovers its
    sibling ``batch_scores.csv``; a directory discovers both recursively. The
    result keeps wide repetition tables in ``replicate_metrics`` and
    ``replicate_scores`` and long method/scenario/metric tables in
    ``metric_summary`` and ``score_summary``.
    """

    overwrite = require_bool(overwrite, "overwrite")
    metrics = _metric_frames(input, "batch_metrics.csv")
    _require_metric_columns(metrics, {"method", "scenario", "seed"}, "Batch")
    if metrics.duplicated(["method", "scenario", "seed"]).any():
        raise ValueError("Batch metrics repeat a method, scenario, and seed identity.")
    metric_identifiers = {
        "method",
        "scenario",
        "seed",
        "batch_recovery_applicable",
        "biological_preservation_applicable",
        "relative_pseudocount",
        "structure_prevalence_min",
        "structure_maximum_relative_abundance_min",
        "structure_n_active_taxa",
        "structure_metrics_supported",
        "structure_relative_pseudocount",
        "structure_pca_components",
    }
    def summarize_numeric(frame: pd.DataFrame, identifiers: set[str]) -> pd.DataFrame:
        numeric = [
            column
            for column in frame.columns
            if column not in identifiers
            and pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_bool_dtype(frame[column])
        ]
        rows: list[dict[str, Any]] = []
        for (method, scenario), group in frame.groupby(
            ["method", "scenario"], sort=True
        ):
            for metric in numeric:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                if len(values):
                    rows.append(
                        {
                            "method": method,
                            "scenario": scenario,
                            "metric": metric,
                            "n_repetitions": int(len(values)),
                            "mean": float(values.mean()),
                            "standard_deviation": (
                                float(values.std(ddof=1)) if len(values) > 1 else None
                            ),
                        }
                    )
        return pd.DataFrame(rows)

    metric_summary = summarize_numeric(metrics, metric_identifiers)
    scores = _batch_score_frames(input)
    score_summary = None
    if scores is not None:
        _require_metric_columns(scores, {"method", "scenario", "seed"}, "Batch score")
        if scores.duplicated(["method", "scenario", "seed"]).any():
            raise ValueError(
                "Batch scores repeat a method, scenario, and seed identity."
            )
        identities = ["method", "scenario", "seed"]
        metric_keys = set(map(tuple, metrics[identities].astype(str).to_numpy()))
        score_keys = set(map(tuple, scores[identities].astype(str).to_numpy()))
        if metric_keys != score_keys:
            raise ValueError(
                "Batch summary input mixes scored and unscored evaluation identities."
            )
        score_identifiers = {
            "method",
            "scenario",
            "seed",
            "batch_recovery_applicable",
            "biological_preservation_applicable",
            "score_definition",
            "classifier_seed",
            "score_supported",
        }
        score_summary = summarize_numeric(scores, score_identifiers)
    output = None
    files: dict[str, Path] = {}
    if output_dir is not None:
        output = Path(output_dir).expanduser().resolve()
        with staged_output_dir(output, overwrite=overwrite) as staging:
            metrics.to_csv(staging / "replicate_metrics.csv", index=False)
            metric_summary.to_csv(staging / "summary.csv", index=False)
            if scores is not None:
                scores.to_csv(staging / "replicate_scores.csv", index=False)
                score_summary.to_csv(staging / "score_summary.csv", index=False)
            write_yaml(staging / "summary.yaml", {"schema_version": "mbbench_batch_summary_v1"})
        files = {
            "replicate_metrics": output / "replicate_metrics.csv",
            "metric_summary": output / "summary.csv",
            "config": output / "summary.yaml",
        }
        if scores is not None:
            files["replicate_scores"] = output / "replicate_scores.csv"
            files["score_summary"] = output / "score_summary.csv"
    return BatchSummary(
        metric_summary=metric_summary,
        score_summary=score_summary,
        replicate_metrics=metrics,
        replicate_scores=scores,
        output_dir=output,
        files=files,
    )
