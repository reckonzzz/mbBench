from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbbench import (
    da_result,
    evaluate_batch,
    evaluate_da,
    evaluate_da_many,
    load_benchmark_bundle,
    summarize_batch,
    summarize_da,
)
from mbbench.benchmark_bundle import write_benchmark_bundle


def _bundle(tmp_path, *, truth_mode="strict_relative_abundance_targets"):
    taxa = ["Taxon_A", "Taxon_B", "Taxon_C", "Rare_taxon"]
    rows = []
    metadata = []
    for group_index, group in enumerate(("control", "case")):
        for batch_index, batch in enumerate(("batch_1", "batch_2")):
            for replicate in range(4):
                rows.append(
                    [
                        40 + 7 * group_index + replicate,
                        25 + 9 * batch_index + replicate,
                        30 + 3 * group_index + 2 * batch_index,
                        1
                        if replicate == 0 and group_index == 0 and batch_index == 0
                        else 0,
                    ]
                )
                metadata.append((group, batch))
    counts_values = np.asarray(rows, dtype=int)
    sample_ids = [f"sample_{index:02d}" for index in range(len(rows))]
    counts = pd.DataFrame(counts_values, columns=taxa)
    counts.insert(0, "sample_id", sample_ids)
    relative_values = counts_values / counts_values.sum(axis=1, keepdims=True)
    relative = pd.DataFrame(relative_values, columns=taxa)
    relative.insert(0, "sample_id", sample_ids)
    reference_values = relative_values.copy()
    reference_values[:, 0] += 0.03
    reference_values[:, 1] -= 0.02
    reference_values[:, 2] -= 0.01
    reference = pd.DataFrame(reference_values, columns=taxa)
    reference.insert(0, "sample_id", sample_ids)
    meta = pd.DataFrame(metadata, columns=["group", "batch"])
    meta.insert(0, "sample_id", sample_ids)
    truth = pd.DataFrame(
        {
            "taxon": taxa,
            "is_direct_biological_target": [True, False, True, False],
            "biological_abundance_coefficient": [0.7, 0.0, -0.7, 0.0],
            "biological_support_coefficient": [0.0] * 4,
        }
    )
    strict = truth_mode in {"strict_relative_abundance_targets", "global_null"}
    if truth_mode == "global_null":
        truth["is_direct_biological_target"] = False
        truth["biological_abundance_coefficient"] = 0.0
    config = {
        "preset": f"test-{truth_mode}",
        "runtime": {"seed": 42},
        "truth_applicability": {
            "da_power": truth_mode != "global_null",
            "da_fdr": strict,
            "da_truth_mode": truth_mode,
            "batch_recovery": True,
            "biological_preservation": True,
        },
    }
    sample_truth = pd.DataFrame({"sample_id": sample_ids, "biological_score": 0.0})
    root = tmp_path / truth_mode
    write_benchmark_bundle(
        root,
        tables={
            "counts": counts,
            "relative_abundance": relative,
            "latent_composition": relative.copy(),
            "metadata": meta,
            "feature_truth": truth,
            "sample_truth": sample_truth,
            "batch_neutral_reference": reference,
        },
        config=config,
    )
    return load_benchmark_bundle(root)


def test_batch_identity_and_reference_invariants(tmp_path):
    bundle = _bundle(tmp_path)
    identity = evaluate_batch(bundle, {"none": bundle.relative_abundance})
    row = identity.metrics.iloc[0]
    assert row["pbrs_raw"] == pytest.approx(0.0, abs=1e-12)
    assert row["mean_paired_aitchison_improvement"] == pytest.approx(0.0)
    assert row["fraction_samples_closer"] == 0.0
    assert row["fraction_samples_unchanged"] == 1.0
    assert row["biology_aitchison_r2_change"] == pytest.approx(0.0)
    assert row["batch_asw_change"] == pytest.approx(0.0)
    assert bool(row["batch_recovery_applicable"])

    reference = evaluate_batch(bundle, {"reference": bundle.batch_neutral_reference})
    row = reference.metrics.iloc[0]
    assert row["corrected_rms_aitchison"] == pytest.approx(0.0, abs=1e-12)
    assert row["pbrs_raw"] == pytest.approx(1.0, abs=1e-12)
    assert row["fraction_samples_closer"] == 1.0


def test_batch_aligns_labels_and_rejects_missing_taxa(tmp_path):
    bundle = _bundle(tmp_path)
    shuffled = bundle.relative_abundance.sample(frac=1, random_state=2)
    shuffled = shuffled[["sample_id", "Rare_taxon", "Taxon_C", "Taxon_A", "Taxon_B"]]
    result = evaluate_batch(bundle, shuffled)
    assert result.metrics.iloc[0]["pbrs_raw"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="taxa do not match"):
        evaluate_batch(bundle, shuffled.drop(columns="Taxon_A"))


def _fixed_lisi(embeddings, metadata):
    del embeddings, metadata
    return {
        "observed": {"batch_ilisi": 0.4, "biology_separation": 0.7},
        "corrected": {"batch_ilisi": 0.4, "biology_separation": 0.7},
        "batch_neutral_reference": {
            "batch_ilisi": 0.8,
            "biology_separation": 0.7,
        },
    }, {"r_version": "test", "lisi_version": "test"}


def test_batch_scores_are_opt_in_and_separate(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "fixture")
    default = evaluate_batch(bundle, bundle.relative_abundance)
    assert default.scores is None
    assert default.config["calculate_scores"] is False

    monkeypatch.setattr("mbbench.batch_scores.standard_lisi_metrics", _fixed_lisi)
    scored = evaluate_batch(
        bundle,
        {"none": bundle.relative_abundance},
        calculate_scores=True,
        output_dir=tmp_path / "scored",
    )
    row = scored.scores.iloc[0]
    assert row["batch_effect_removal_score"] == pytest.approx(0.0, abs=1e-12)
    assert row["biological_signal_preservation_score"] == pytest.approx(1.0)
    assert row["overall_score"] == pytest.approx(0.6)
    pd.testing.assert_frame_equal(default.metrics, scored.metrics.assign(method="method"))
    assert scored.files["scores"].name == "batch_scores.csv"
    assert scored.files["scores"].is_file()


def test_python_scores_require_r_and_lisi_only_when_requested(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)
    monkeypatch.setattr("mbbench.batch_scores.shutil.which", lambda name: None)
    evaluate_batch(bundle, bundle.relative_abundance)
    with pytest.raises(RuntimeError, match="requires R and the R package 'lisi'"):
        evaluate_batch(bundle, bundle.relative_abundance, calculate_scores=True)


def test_no_batch_scenario_has_no_batch_or_overall_score(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path)
    bundle.config["truth_applicability"]["batch_recovery"] = False
    monkeypatch.setattr("mbbench.batch_scores.standard_lisi_metrics", _fixed_lisi)
    result = evaluate_batch(
        bundle, bundle.relative_abundance, calculate_scores=True
    )
    row = result.scores.iloc[0]
    assert pd.isna(row["batch_effect_removal_score"])
    assert pd.isna(row["overall_score"])
    assert row["biological_signal_preservation_score"] == pytest.approx(1.0)


def test_batch_summary_keeps_scores_separate(tmp_path):
    root = tmp_path / "runs"
    for seed in (42, 43):
        run = root / str(seed)
        run.mkdir(parents=True)
        pd.DataFrame(
            {
                "method": ["method_a"],
                "scenario": ["scenario_a"],
                "seed": [seed],
                "pbrs": [0.4 + 0.1 * (seed - 42)],
            }
        ).to_csv(run / "batch_metrics.csv", index=False)
        pd.DataFrame(
            {
                "method": ["method_a"],
                "scenario": ["scenario_a"],
                "seed": [seed],
                "overall_score": [0.6 + 0.1 * (seed - 42)],
            }
        ).to_csv(run / "batch_scores.csv", index=False)
    summary = summarize_batch(root, output_dir=tmp_path / "summary")
    assert summary.metric_summary.query("metric == 'pbrs'").iloc[0]["mean"] == pytest.approx(0.45)
    assert summary.score_summary.query("metric == 'overall_score'").iloc[0]["mean"] == pytest.approx(0.65)
    assert summary.files["score_summary"].is_file()


def test_batch_summary_rejects_partly_scored_repetitions(tmp_path):
    root = tmp_path / "runs"
    for seed in (42, 43):
        run = root / str(seed)
        run.mkdir(parents=True)
        pd.DataFrame(
            {"method": ["method_a"], "scenario": ["scenario_a"], "seed": [seed]}
        ).to_csv(run / "batch_metrics.csv", index=False)
        if seed == 42:
            pd.DataFrame(
                {
                    "method": ["method_a"],
                    "scenario": ["scenario_a"],
                    "seed": [seed],
                    "overall_score": [0.6],
                }
            ).to_csv(run / "batch_scores.csv", index=False)
    with pytest.raises(ValueError, match="mixes scored and unscored"):
        summarize_batch(root)


def test_batch_summary_rejects_score_file_as_metric_input(tmp_path):
    path = tmp_path / "batch_scores.csv"
    pd.DataFrame(
        {
            "method": ["method_a"],
            "scenario": ["scenario_a"],
            "seed": [42],
            "overall_score": [0.7],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="cannot be summarized without"):
        summarize_batch(path)


def test_evaluator_rejects_string_overwrite_without_writing(tmp_path):
    bundle = _bundle(tmp_path)
    with pytest.raises(TypeError, match="overwrite must be True or False"):
        evaluate_batch(bundle, bundle.relative_abundance, overwrite="False")


def test_structure_filter_does_not_change_paired_taxon_roster(tmp_path):
    bundle = _bundle(tmp_path)
    corrected = bundle.relative_abundance.copy()
    corrected["Rare_taxon"] += 0.02
    result = evaluate_batch(bundle, corrected)
    row = result.metrics.iloc[0]
    assert row["structure_n_active_taxa"] == 3
    assert row["corrected_rms_aitchison"] > 0
    assert result.config["paired_metrics_use_full_taxon_roster"] is True

    excluded_only = bundle.relative_abundance.copy()
    excluded_only[["Taxon_A", "Taxon_B", "Taxon_C"]] = 0.0
    excluded_only["Rare_taxon"] = 1.0
    unsupported = evaluate_batch(bundle, excluded_only).metrics.iloc[0]
    assert not bool(unsupported["structure_metrics_supported"])
    assert "zero-mass sample" in unsupported["structure_metrics_reason"]
    assert unsupported["corrected_rms_aitchison"] > 0


def test_strict_da_metrics_and_result_helper(tmp_path):
    bundle = _bundle(tmp_path)
    calls = da_result(
        bundle, significant_taxa=["Taxon_A", "Taxon_C"], tested_taxa=None
    )
    result = evaluate_da(bundle, {"perfect": calls})
    row = result.metrics.iloc[0]
    assert row["direct_target_power"] == 1.0
    assert row["type_i_error_rate"] == 0.0
    assert row["type_ii_error_rate"] == 0.0
    assert row["fdp"] == 0.0
    assert row["precision"] == 1.0

    malformed = calls.copy()
    malformed["p_value"] = [0.01, "not-a-number", 0.2, 0.8]
    with pytest.raises(ValueError, match="numeric values"):
        evaluate_da(bundle, malformed)


def test_da_applicability_for_global_null_and_direct_targets(tmp_path):
    null_bundle = _bundle(tmp_path / "null", truth_mode="global_null")
    null_calls = da_result(null_bundle, significant_taxa=["Taxon_B"])
    null_row = evaluate_da(null_bundle, null_calls).metrics.iloc[0]
    assert null_row["fdp"] == 1.0
    assert bool(null_row["fwer_indicator"])
    assert pd.isna(null_row["direct_target_power"])

    direct_bundle = _bundle(tmp_path / "direct", truth_mode="direct_targets_only")
    direct_calls = da_result(
        direct_bundle, significant_taxa=["Taxon_A", "Taxon_B"]
    )
    direct_row = evaluate_da(direct_bundle, direct_calls).metrics.iloc[0]
    assert direct_row["direct_target_power"] == pytest.approx(0.5)
    assert direct_row["outside_target_fraction_descriptive"] == pytest.approx(0.5)
    assert pd.isna(direct_row["fdp"])
    assert pd.isna(direct_row["type_i_error_rate"])


def test_summarize_da_accepts_many_repetitions_without_a_python_list(tmp_path):
    rows = []
    for seed in range(100):
        rows.append(
            {
                "method": "method_a",
                "scenario": "strict",
                "seed": seed,
                "da_truth_mode": "strict_relative_abundance_targets",
                "fdp": 0.2 if seed < 25 else 0.0,
                "direct_target_power": 0.5,
                "type_i_error_rate": 0.01,
                "type_ii_error_rate": 0.5,
                "fwer_indicator": seed < 10,
            }
        )
    root = tmp_path / "evaluations" / "all"
    root.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / "da_metrics.csv", index=False)
    summary = summarize_da(tmp_path / "evaluations", bootstrap_repetitions=100)
    row = summary.iloc[0]
    assert row["n_repetitions"] == 100
    assert row["empirical_fdr"] == pytest.approx(0.05)
    assert row["mean_power"] == pytest.approx(0.5)
    assert row["fwer"] == pytest.approx(0.1)


def test_da_bootstrap_intervals_do_not_depend_on_other_methods_or_row_order():
    def rows(method, values):
        return [
            {
                "method": method,
                "scenario": "strict",
                "seed": index,
                "da_truth_mode": "strict_relative_abundance_targets",
                "fdp": value,
                "direct_target_power": 1 - value,
                "type_i_error_rate": value / 2,
                "type_ii_error_rate": value / 3,
                "fwer_indicator": value > 0.25,
            }
            for index, value in enumerate(values)
        ]

    target = pd.DataFrame(rows("b", [0.0, 0.1, 0.4, 0.8, 0.2]))
    other = pd.DataFrame(rows("a", [0.9, 0.8, 0.7, 0.6, 0.5]))
    alone = summarize_da(target, bootstrap_repetitions=200, seed=42).iloc[0]
    together = summarize_da(
        pd.concat([other, target], ignore_index=True),
        bootstrap_repetitions=200,
        seed=42,
    ).query("method == 'b'").iloc[0]
    reversed_rows = summarize_da(
        target.iloc[::-1].reset_index(drop=True),
        bootstrap_repetitions=200,
        seed=42,
    ).iloc[0]
    interval_columns = [column for column in alone.index if "_ci_" in column]
    pd.testing.assert_series_equal(
        alone[interval_columns], together[interval_columns], check_names=False
    )
    pd.testing.assert_series_equal(
        alone[interval_columns], reversed_rows[interval_columns], check_names=False
    )


def test_bundle_rejects_duplicate_sample_ids(tmp_path):
    bundle = _bundle(tmp_path / "valid")
    tables = {
        "counts": bundle.counts.copy(),
        "relative_abundance": bundle.relative_abundance.copy(),
        "latent_composition": bundle.latent_composition.copy(),
        "metadata": bundle.metadata.copy(),
        "feature_truth": bundle.feature_truth.copy(),
        "sample_truth": bundle.sample_truth.copy(),
        "batch_neutral_reference": bundle.batch_neutral_reference.copy(),
    }
    for name, table in tables.items():
        if name != "feature_truth":
            table.loc[1, "sample_id"] = table.loc[0, "sample_id"]
    with pytest.raises(ValueError, match="sample IDs must be nonempty and unique"):
        write_benchmark_bundle(tmp_path / "duplicate", tables=tables, config=bundle.config)


def test_da_manifest_resolves_relative_paths_and_writes_summary(tmp_path):
    bundle = _bundle(tmp_path / "fixture")
    calls = da_result(bundle, significant_taxa=["Taxon_A", "Taxon_C"])
    calls.to_csv(tmp_path / "calls.csv", index=False)
    pd.DataFrame(
        {
            "benchmark": [str(bundle.root.relative_to(tmp_path))],
            "input": ["calls.csv"],
            "method": ["method_a"],
        }
    ).to_csv(tmp_path / "manifest.csv", index=False)

    result = evaluate_da_many(
        tmp_path / "manifest.csv", output_dir=tmp_path / "evaluation"
    )
    assert result.summary is not None
    assert result.summary.iloc[0]["empirical_fdr"] == pytest.approx(0.0)
    assert result.files["summary"].is_file()
    assert result.config["summary"]["bootstrap_seed"] == 42
