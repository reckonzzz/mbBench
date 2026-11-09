from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mbbench import list_conditions, load_model
from mbbench._benchmark import benchmark, generate_benchmark
from mbbench.benchmark_bundle import load_benchmark_bundle
from mbbench.benchmark_config import resolve_preset
from mbbench.diffusion import DiffusionConfig, LatentDenoiser
from mbbench.generation import generate, generate_from_condition_table
from mbbench.simulator import Simulator, _select_donor_pool
from mbbench.tokenizer import HurdleAE, TokenizerConfig


def _tiny_bundle(path, *, custom_condition=False, generic_condition_only=False):
    path.mkdir()
    if generic_condition_only:
        vocabularies = {
            "Cohort": {
                "type": "categorical",
                "values": ["<UNK>", "case", "control"],
            }
        }
    else:
        vocabularies = {
            "Phenotype_Tokens": {"type": "multilabel", "tokens": ["Healthy"]},
            "Body_Site_Coarse": {
                "type": "categorical",
                "values": ["<UNK>", "stool_feces"],
            },
        }
    if custom_condition and not generic_condition_only:
        vocabularies["Cohort"] = {
            "type": "categorical",
            "values": ["<UNK>", "case", "control"],
        }
    bio_categorical = (
        {"Cohort": 3} if generic_condition_only else {"Body_Site_Coarse": 2}
    )
    if custom_condition and not generic_condition_only:
        bio_categorical["Cohort"] = 3
    tokenizer_config = TokenizerConfig(
        input_dim=3,
        bio_latent_dim=4,
        tech_latent_dim=2,
        project_latent_dim=2,
        encoder_hidden_dim=8,
        decoder_hidden_dim=8,
        condition_embedding_dim=2,
        dropout=0.0,
        bio_categorical_field_sizes=bio_categorical,
        bio_multilabel_field_dims=(
            {} if generic_condition_only else {"Phenotype_Tokens": 1}
        ),
    )
    if generic_condition_only:
        primary_fields = ("Cohort",)
    elif custom_condition:
        primary_fields = ("Phenotype_Tokens", "Body_Site_Coarse", "Cohort")
    else:
        primary_fields = ("Phenotype_Tokens", "Body_Site_Coarse")
    diffusion_config = DiffusionConfig(
        latent_dim=4,
        token_dim=4,
        num_latent_tokens=2,
        num_layers=1,
        num_heads=1,
        mlp_ratio=2.0,
        num_steps=2,
        condition_hidden_dim=8,
        dropout=0.0,
        primary_fields=primary_fields,
    )
    torch.manual_seed(7)
    tokenizer = HurdleAE(tokenizer_config)
    denoiser = LatentDenoiser(
        diffusion_config,
        bio_categorical,
        {} if generic_condition_only else {"Phenotype_Tokens": 1},
    )
    torch.save(
        {
            "state_dict": tokenizer.state_dict(),
            "model_type": "ae",
            "tokenizer_config": tokenizer_config.__dict__,
        },
        path / "tokenizer.pt",
    )
    torch.save(
        {
            "state_dict": denoiser.state_dict(),
            "model_config": diffusion_config.__dict__,
            "tokenizer_model_type": "ae",
            "latent_stats": {"mean": [0.0] * 4, "std": [1.0] * 4},
        },
        path / "diffusion.pt",
    )
    (path / "model.yaml").write_text(
        yaml.safe_dump(
            {
                "model_type": "ae",
                "condition_fields": list(primary_fields),
                "supports_body_system": False,
                "supported_output_modes": ["relative"],
            }
        ),
        encoding="utf-8",
    )
    (path / "vocabularies.json").write_text(json.dumps(vocabularies), encoding="utf-8")
    pd.DataFrame({"taxon": ["g1", "g2", "g3"]}).to_csv(path / "taxa.csv", index=False)
    combinations = (
        {"Cohort": ["case", "control"]}
        if generic_condition_only
        else {
            "Phenotype_Tokens": ["Healthy"],
            "Body_Site_Coarse": ["stool_feces"],
        }
    )
    if custom_condition and not generic_condition_only:
        combinations["Cohort"] = ["case"]
    pd.DataFrame(combinations).to_csv(path / "condition_combinations.csv", index=False)


def test_fixed_seed_generation_and_benchmark(tmp_path):
    model_dir = tmp_path / "model"
    _tiny_bundle(model_dir)
    simulator = Simulator.from_directory(model_dir, "cpu")
    first = simulator.generate(n_samples=4, seed=42)
    second = simulator.generate(n_samples=4, seed=42)
    np.testing.assert_allclose(
        first.abundance.iloc[:, 1:], second.abundance.iloc[:, 1:]
    )
    np.testing.assert_allclose(first.abundance.iloc[:, 1:].sum(axis=1), 1.0)

    scenario = resolve_preset(
        "da-strict-common-moderate",
        {
            "design.n_samples": 8,
            "design.n_batches": 2,
            "design.minimum_cell_count": 2,
            "biology.target_taxa": ["g1", "g2"],
            "observation.model": "multinomial",
            "observation.depth_mode": "fixed",
            "observation.library_size": 100,
            "observation.depth_min": 100,
            "observation.depth_max": 100,
            "runtime.batch_size": 4,
        },
    )
    result = generate_benchmark(simulator, scenario, output_dir=tmp_path / "benchmark")
    assert result.counts.iloc[:, 1:].sum(axis=1).eq(100).all()
    assert int(result.feature_truth["is_direct_biological_target"].sum()) == 2
    bundle = load_benchmark_bundle(result.output_dir)
    assert (
        bundle.config["resolved_internal"]["strict_target_audit"][
            "support_change_count"
        ]
        == 0
    )

    confounded_scenario = resolve_preset(
        "da-null-confounded-global",
        {
            "design.n_samples": 16,
            "design.n_batches": 2,
            "design.case_fractions": [0.25, 0.75],
            "design.minimum_cell_count": 2,
            "batch.mechanisms": ["global", "depth"],
            "observation.model": "multinomial",
            "observation.depth_mode": "fixed",
            "observation.library_size": 100,
            "observation.depth_min": 50,
            "observation.depth_max": 200,
            "runtime.batch_size": 4,
        },
    )
    confounded = generate_benchmark(
        simulator, confounded_scenario, output_dir=tmp_path / "benchmark_confounded"
    )
    config = yaml.safe_load((confounded.output_dir / "benchmark.yaml").read_text())
    assert config["actual_cell_counts"]["batch_1"] == {"control": 6, "case": 2}
    assert config["actual_cell_counts"]["batch_2"] == {"control": 2, "case": 6}


def test_benchmark_accepts_models_without_default_condition_fields(tmp_path):
    model_dir = tmp_path / "generic_model"
    _tiny_bundle(model_dir, generic_condition_only=True)
    simulator = Simulator.from_directory(model_dir, "cpu")
    result = benchmark(
        simulator,
        preset="da-null-balanced",
        n_samples=8,
        n_batches=2,
        overrides={
            "conditions": {
                "common": {},
                "groups": [
                    {"label": "control", "conditions": {"Cohort": "control"}},
                    {"label": "case", "conditions": {"Cohort": "case"}},
                ],
            },
            "design": {"minimum_cell_count": 2},
            "observation": {
                "model": "multinomial",
                "depth_mode": "fixed",
                "library_size": 100,
                "depth_min": 100,
                "depth_max": 100,
            },
        },
    )
    assert set(result.metadata["Cohort"]) == {"case", "control"}
    assert "Phenotype_Tokens" not in result.metadata
    assert "Body_Site_Coarse" not in result.metadata


def test_public_model_loader_and_condition_listing(tmp_path):
    model_dir = tmp_path / "model"
    _tiny_bundle(model_dir)
    simulator = load_model(path=model_dir, device="cpu")
    conditions = list_conditions(simulator)
    assert conditions["Phenotype_Tokens"] == ["Healthy"]
    assert conditions["Body_Site_Coarse"] == ["stool_feces"]


def test_generate_memory_limit_and_streamed_output(tmp_path):
    model_dir = tmp_path / "model"
    _tiny_bundle(model_dir)
    simulator = Simulator.from_directory(model_dir, "cpu")

    expected = generate(simulator, n_samples=5, batch_size=2, seed=42)
    assert expected.output_dir is None
    assert expected.abundance is not None
    assert expected.metadata is not None
    assert expected.loaded is True

    with pytest.raises(ValueError, match="increase max_in_memory_mb"):
        generate(
            simulator,
            n_samples=5,
            batch_size=2,
            seed=42,
            max_in_memory_mb=1e-6,
        )

    streamed = generate(
        simulator,
        n_samples=5,
        batch_size=2,
        seed=42,
        max_in_memory_mb=1e-6,
        output_dir=tmp_path / "streamed",
    )
    assert streamed.abundance is None
    assert streamed.metadata is None
    assert streamed.loaded is False
    pd.testing.assert_frame_equal(
        pd.read_csv(streamed.files["abundance"]), expected.abundance, check_dtype=False
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(streamed.files["metadata"]), expected.metadata
    )
    assert set(path.name for path in streamed.output_dir.iterdir()) == {
        "abundance.csv",
        "metadata.csv",
        "generation.yaml",
    }
    assert streamed.config["load"] is True
    assert streamed.config["loaded"] is False

    with pytest.raises(ValueError, match="load=False requires output_dir"):
        generate(simulator, n_samples=2, load=False)
    files_only = generate(
        simulator,
        n_samples=5,
        batch_size=2,
        seed=42,
        output_dir=tmp_path / "files_only",
        load=False,
    )
    assert files_only.abundance is None
    assert files_only.metadata is None
    assert files_only.config["load"] is False
    assert files_only.config["loaded"] is False
    pd.testing.assert_frame_equal(
        pd.read_csv(files_only.files["abundance"]),
        expected.abundance,
        check_dtype=False,
    )
    different_chunking = generate(simulator, n_samples=5, batch_size=4, seed=42)
    pd.testing.assert_frame_equal(expected.abundance, different_chunking.abundance)
    pd.testing.assert_frame_equal(expected.metadata, different_chunking.metadata)


def test_benchmark_memory_limit_and_streamed_output_are_equivalent(tmp_path):
    model_dir = tmp_path / "model"
    _tiny_bundle(model_dir)
    simulator = Simulator.from_directory(model_dir, "cpu")
    scenario = resolve_preset(
        "da-strict-common-moderate",
        {
            "design.n_samples": 8,
            "design.n_batches": 2,
            "design.minimum_cell_count": 2,
            "biology.target_taxa": ["g1", "g2"],
            "observation.model": "multinomial",
            "observation.depth_mode": "fixed",
            "observation.library_size": 100,
            "observation.depth_min": 100,
            "observation.depth_max": 100,
            "observation.paired_neutral_counts": True,
            "runtime.batch_size": 3,
        },
    )
    expected = generate_benchmark(simulator, scenario)
    assert expected.output_dir is None
    assert expected.counts is not None
    assert expected.loaded is True

    with pytest.raises(ValueError, match="increase max_in_memory_mb"):
        generate_benchmark(simulator, scenario, max_in_memory_mb=1e-6)

    streamed = generate_benchmark(
        simulator,
        scenario,
        output_dir=tmp_path / "streamed_benchmark",
        max_in_memory_mb=1e-6,
    )
    assert streamed.counts is None
    assert streamed.loaded is False
    loaded = load_benchmark_bundle(streamed.output_dir)
    for name in (
        "counts",
        "relative_abundance",
        "latent_composition",
        "metadata",
        "feature_truth",
        "sample_truth",
        "batch_neutral_reference",
        "batch_neutral_counts",
    ):
        pd.testing.assert_frame_equal(
            getattr(loaded, name),
            getattr(expected, name),
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
    assert not any(path.name.startswith(".") for path in streamed.output_dir.iterdir())
    assert streamed.config["runtime"]["load"] is True
    assert streamed.config["runtime"]["loaded"] is False

    with pytest.raises(ValueError, match="load=False requires output_dir"):
        generate_benchmark(simulator, scenario, load=False)
    files_only = generate_benchmark(
        simulator,
        scenario,
        output_dir=tmp_path / "files_only_benchmark",
        load=False,
    )
    assert files_only.counts is None
    assert files_only.config["runtime"]["load"] is False
    assert files_only.config["runtime"]["loaded"] is False
    files_only_loaded = load_benchmark_bundle(files_only.output_dir)
    pd.testing.assert_frame_equal(
        files_only_loaded.counts, expected.counts, check_dtype=False
    )


def test_streamed_null_dirichlet_multinomial_keeps_rng_order(tmp_path):
    model_dir = tmp_path / "model"
    _tiny_bundle(model_dir)
    simulator = Simulator.from_directory(model_dir, "cpu")
    scenario = resolve_preset(
        "da-null-balanced",
        {
            "design.n_samples": 8,
            "design.n_batches": 2,
            "design.minimum_cell_count": 2,
            "observation.depth_mode": "fixed",
            "observation.library_size": 100,
            "observation.depth_min": 100,
            "observation.depth_max": 100,
            "observation.concentration_mode": "fixed",
            "observation.concentration_median": 500.0,
            "observation.concentration_min": 500.0,
            "observation.concentration_max": 500.0,
            "observation.paired_neutral_counts": True,
            "runtime.batch_size": 3,
        },
    )
    expected = generate_benchmark(simulator, scenario)
    streamed = generate_benchmark(
        simulator,
        scenario,
        output_dir=tmp_path / "streamed_null",
        max_in_memory_mb=1e-6,
    )
    loaded = load_benchmark_bundle(streamed.output_dir)
    pd.testing.assert_frame_equal(loaded.counts, expected.counts, check_dtype=False)
    pd.testing.assert_frame_equal(
        loaded.batch_neutral_counts,
        expected.batch_neutral_counts,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        loaded.latent_composition,
        expected.latent_composition,
        check_dtype=False,
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )


def test_donor_matching_uses_exact_then_body_site_then_global():
    donors = pd.DataFrame(
        {
            "Phenotype_Tokens": ["Healthy"] * 8 + ["Disease"] * 4 + ["Other"],
            "Body_Site_Coarse": ["stool_feces"] * 12 + ["oral"],
            "raw_total_reads": range(13),
        }
    )
    vocabularies = {
        "Phenotype_Tokens": {"type": "multilabel"},
        "Body_Site_Coarse": {"type": "categorical"},
    }
    exact, level = _select_donor_pool(
        donors,
        {"Phenotype_Tokens": ["Healthy"], "Body_Site_Coarse": ["stool_feces"]},
        vocabularies,
    )
    assert level == "exact_conditions"
    assert len(exact) == 8

    body_site, level = _select_donor_pool(
        donors,
        {"Phenotype_Tokens": ["Disease"], "Body_Site_Coarse": ["stool_feces"]},
        vocabularies,
    )
    assert level == "body_site"
    assert len(body_site) == 12

    global_pool, level = _select_donor_pool(
        donors,
        {"Phenotype_Tokens": ["Missing"], "Body_Site_Coarse": ["missing"]},
        vocabularies,
    )
    assert level == "global"
    assert len(global_pool) == len(donors)

    multilabel_donors = pd.DataFrame(
        {
            "Phenotype_Tokens": ["Healthy|IBD"] * 8 + ["Healthy"] * 4,
            "Body_Site_Coarse": ["stool_feces"] * 12,
            "raw_total_reads": range(12),
        }
    )
    fallback, level = _select_donor_pool(
        multilabel_donors,
        {"Phenotype_Tokens": ["Healthy"], "Body_Site_Coarse": ["stool_feces"]},
        vocabularies,
    )
    assert level == "body_site"
    assert len(fallback) == 12


def test_multilabel_condition_status_requires_the_observed_token_set(tmp_path):
    combinations = tmp_path / "condition_combinations.csv"
    pd.DataFrame(
        {
            "Phenotype_Tokens": ["Healthy", "IBD"],
            "Body_Site_Coarse": ["stool_feces", "stool_feces"],
        }
    ).to_csv(combinations, index=False)
    simulator = object.__new__(Simulator)
    simulator.bundle = SimpleNamespace(
        root=tmp_path,
        vocabularies={
            "Phenotype_Tokens": {"type": "multilabel"},
            "Body_Site_Coarse": {"type": "categorical"},
        },
    )
    simulator.diffusion_config = SimpleNamespace(
        primary_fields=("Phenotype_Tokens", "Body_Site_Coarse")
    )
    requested = {
        "Phenotype_Tokens": ["Healthy", "IBD"],
        "Body_Site_Coarse": ["stool_feces"],
    }
    with pytest.warns(UserWarning, match="combinatorial extrapolation"):
        assert simulator._condition_status(requested) == "unobserved_combination"

    pd.DataFrame(
        {
            "Phenotype_Tokens": ["IBD|Healthy"],
            "Body_Site_Coarse": ["stool_feces"],
        }
    ).to_csv(combinations, index=False)
    assert simulator._condition_status(requested) == "observed"


def test_custom_biological_condition_generation(tmp_path):
    model_dir = tmp_path / "model"
    _tiny_bundle(model_dir, custom_condition=True)
    simulator = Simulator.from_directory(model_dir, "cpu")
    result = simulator.generate(n_samples=2, seed=42, conditions={"Cohort": "case"})
    assert result.metadata["Cohort"].eq("case").all()
    with pytest.warns(UserWarning, match="combinatorial extrapolation"):
        unseen = simulator.generate(
            n_samples=1, seed=42, conditions={"Cohort": "control"}
        )
    assert unseen.metadata.loc[0, "condition_status"] == "unobserved_combination"
    with pytest.raises(ValueError, match="did you mean 'Cohort'"):
        simulator.generate(n_samples=1, conditions={"cohort": "case"})
    with pytest.raises(ValueError, match="did you mean 'case'"):
        simulator.generate(n_samples=1, conditions={"Cohort": "Case"})


def test_generate_from_condition_table_accepts_data_frames(tmp_path):
    model_dir = tmp_path / "model"
    _tiny_bundle(model_dir, custom_condition=True)
    simulator = Simulator.from_directory(model_dir, "cpu")
    condition_table = pd.DataFrame(
        {
            "Phenotype_Tokens": ["Healthy", "Healthy"],
            "Body_Site_Coarse": ["stool_feces", "stool_feces"],
            "Cohort": ["case", "control"],
            "n_samples": [2, 3],
        }
    )
    with pytest.warns(UserWarning, match="combinatorial extrapolation"):
        result = generate_from_condition_table(simulator, condition_table, seed=42)
    assert result.loaded is True
    assert len(result.abundance) == 5
    assert result.metadata["Cohort"].tolist() == [
        "case",
        "case",
        "control",
        "control",
        "control",
    ]

    repeated = condition_table.copy()
    repeated["phenotype"] = "Healthy"
    with pytest.raises(ValueError, match="repeats equivalent fields"):
        generate_from_condition_table(simulator, repeated)


def test_high_level_benchmark_accepts_nested_native_overrides(tmp_path):
    model_dir = tmp_path / "model"
    _tiny_bundle(model_dir, custom_condition=True)
    simulator = Simulator.from_directory(model_dir, "cpu")
    with pytest.warns(UserWarning, match="combinatorial extrapolation"):
        result = benchmark(
            simulator,
            preset="da-null-balanced",
            n_samples=8,
            n_batches=2,
            overrides={
                "conditions": {"common": {"Cohort": "control"}},
                "design": {"minimum_cell_count": 2},
                "observation": {
                    "model": "multinomial",
                    "depth_mode": "fixed",
                    "library_size": 100,
                    "depth_min": 100,
                    "depth_max": 100,
                },
                "runtime": {"batch_size": 4},
            },
        )
    assert result.loaded is True
    assert result.counts.iloc[:, 1:].sum(axis=1).eq(100).all()
    assert result.metadata["condition_status"].eq("unobserved_combination").all()


def test_missing_combination_table_is_not_reported_as_observed(tmp_path):
    model_dir = tmp_path / "model"
    _tiny_bundle(model_dir)
    (model_dir / "condition_combinations.csv").unlink()
    result = Simulator.from_directory(model_dir, "cpu").generate(n_samples=1, seed=42)
    assert result.metadata.loc[0, "condition_status"] == "unknown_not_auditable"


def _generate_command(model_dir, output_dir, *extra):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mbbench.cli",
            "generate",
            "--model",
            "custom",
            "--model-dir",
            str(model_dir),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--output-dir",
            str(output_dir),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_generate_rejects_condition_alias_conflicts_and_invalid_counts(tmp_path):
    model_dir = tmp_path / "model"
    _tiny_bundle(model_dir)

    direct = _generate_command(
        model_dir,
        tmp_path / "direct",
        "--condition",
        "Phenotype_Tokens=Healthy",
    )
    assert direct.returncode != 0
    assert "Use --phenotype and --body-site" in direct.stderr

    conflicts = tmp_path / "conflicts.csv"
    conflicts.write_text(
        "phenotype,Phenotype_Tokens,body_site,n_samples\nHealthy,Healthy,stool_feces,2\n",
        encoding="utf-8",
    )
    repeated = _generate_command(
        model_dir,
        tmp_path / "repeated",
        "--condition-table",
        str(conflicts),
    )
    assert repeated.returncode != 0
    assert "repeats equivalent condition fields" in repeated.stderr

    invalid_counts = tmp_path / "invalid_counts.csv"
    invalid_counts.write_text(
        "phenotype,body_site,n_samples\nHealthy,stool_feces,2.5\n",
        encoding="utf-8",
    )
    invalid = _generate_command(
        model_dir,
        tmp_path / "invalid",
        "--condition-table",
        str(invalid_counts),
    )
    assert invalid.returncode != 0
    assert "positive integer" in invalid.stderr


def test_generate_writes_resolved_yaml(tmp_path):
    model_dir = tmp_path / "model"
    _tiny_bundle(model_dir)
    output_dir = tmp_path / "generated"
    result = _generate_command(model_dir, output_dir, "--n-samples", "3")
    assert result.returncode == 0, result.stderr

    config = yaml.safe_load(
        (output_dir / "generation.yaml").read_text(encoding="utf-8")
    )
    assert config["model"]["type"] == "ae"
    assert config["model"]["path"] == str(model_dir.resolve())
    assert config["n_samples"] == 3
    assert config["condition_table"] == [
        {
            "n_samples": 3,
            "conditions": {
                "Phenotype_Tokens": "Healthy",
                "Body_Site_Coarse": "stool_feces",
            },
        }
    ]
    assert config["seed"] == 42
    assert config["batch_size"] == 2
