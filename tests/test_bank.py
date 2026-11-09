from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import yaml

import mbbench
from mbbench.cli import _run_generate, build_parser, validate_model_arguments


def _tiny_bank(root: Path, *, field: str = "Phenotype_Tokens") -> Path:
    root.mkdir()
    taxa = ["Taxon_A", "Taxon_B", "Taxon_C", "Taxon_D"]
    values = ["Healthy"] * 80 + ["Disease"] * 80
    rng = np.random.default_rng(17)
    np.save(root / "abundance_logits.npy", rng.normal(size=(160, 4)).astype(np.float32))
    np.save(root / "support_logits.npy", rng.normal(size=(160, 4)).astype(np.float32))
    pd.DataFrame(
        {
            "sample_id": [f"bank_{index:03d}" for index in range(160)],
            field: values,
            "Body_Site_Coarse": "stool_feces",
        }
    ).to_csv(root / "metadata.csv.gz", index=False)
    pd.DataFrame({"taxon": taxa}).to_csv(root / "taxa.csv", index=False)
    pd.DataFrame(
        {
            "raw_total_reads": [8000, 10000, 12000, 15000],
            "Body_Site_Coarse": "stool_feces",
        }
    ).to_csv(root / "donors.csv", index=False)
    (root / "bank.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "bank_type": "batch-neutral-logit-bank",
                "condition_fields": [field, "Body_Site_Coarse"],
                "support_threshold": 0.5,
                "supports_body_system": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root


def test_download_bank_installs_nested_zip_and_reuses_cache(tmp_path, capsys):
    source = _tiny_bank(tmp_path / "core")
    archive = tmp_path / "core_bank.zip"
    with ZipFile(archive, "w") as zipped:
        for file in source.iterdir():
            zipped.write(file, f"core/{file.name}")
    target = tmp_path / "installed"
    with mbbench.download_bank(url=archive.as_uri(), install_dir=target, progress=True) as bank:
        result = mbbench.generate(bank=bank, n_samples=2, seed=42)
        assert bank.root == target.resolve()
        assert np.allclose(result.abundance.iloc[:, 1:].sum(axis=1), 1)
    assert "100%" in capsys.readouterr().err

    archive.unlink()
    with mbbench.download_bank(url=archive.as_uri(), install_dir=target, progress=True) as bank:
        assert bank.abundance_logits.shape == (160, 4)
    assert not capsys.readouterr().err


def test_download_bank_discards_invalid_bank(tmp_path):
    source = _tiny_bank(tmp_path / "core")
    np.save(source / "support_logits.npy", np.zeros((1, 4), dtype=np.float32))
    archive = tmp_path / "core_bank.zip"
    with ZipFile(archive, "w") as zipped:
        for file in source.iterdir():
            zipped.write(file, file.name)
    target = tmp_path / "banks" / "installed"
    with pytest.raises(ValueError, match="matching matrices"):
        mbbench.download_bank(url=archive.as_uri(), install_dir=target)
    assert not list(target.parent.iterdir())


def test_load_bank_and_generate_without_reusing_rows(tmp_path: Path) -> None:
    bank = mbbench.load_bank(_tiny_bank(tmp_path / "bank"))
    result = mbbench.generate(
        bank=bank,
        phenotype="Healthy",
        body_site="stool_feces",
        n_samples=11,
        batch_size=3,
        seed=42,
    )

    assert isinstance(bank, mbbench.BatchNeutralBank)
    assert result.loaded
    assert result.abundance.shape == (11, 5)
    assert result.metadata["source_bank_sample_id"].is_unique
    assert np.allclose(result.abundance.iloc[:, 1:].sum(axis=1), 1.0)
    assert result.config["generation_source"] == "bank"
    assert result.config["bank"]["type"] == "batch-neutral-logit-bank"

    repeated = mbbench.generate_from_condition_table(
        bank=bank,
        condition_table=pd.DataFrame(
            {
                "phenotype": ["Healthy", "Healthy"],
                "body_site": ["stool_feces", "stool_feces"],
                "n_samples": [20, 20],
            }
        ),
        batch_size=7,
        seed=42,
    )
    assert repeated.metadata["source_bank_sample_id"].is_unique

    different_chunking = mbbench.generate(
        bank=bank,
        phenotype="Healthy",
        body_site="stool_feces",
        n_samples=11,
        batch_size=8,
        seed=42,
    )
    pd.testing.assert_frame_equal(result.abundance, different_chunking.abundance)
    pd.testing.assert_frame_equal(result.metadata, different_chunking.metadata)


def test_bank_condition_requests_are_stable_under_split_and_reorder(tmp_path: Path) -> None:
    bank = mbbench.load_bank(_tiny_bank(tmp_path / "bank"))
    combined = mbbench.generate_from_condition_table(
        bank=bank,
        condition_table=[
            {"phenotype": "Healthy", "body_site": "stool_feces", "n_samples": 12}
        ],
        seed=42,
    )
    split = mbbench.generate_from_condition_table(
        bank=bank,
        condition_table=[
            {"phenotype": "Healthy", "body_site": "stool_feces", "n_samples": 5},
            {"phenotype": "Healthy", "body_site": "stool_feces", "n_samples": 7},
        ],
        seed=42,
    )
    pd.testing.assert_frame_equal(combined.abundance, split.abundance)
    pd.testing.assert_frame_equal(combined.metadata, split.metadata)

    first = mbbench.generate_from_condition_table(
        bank=bank,
        condition_table=[
            {"phenotype": "Healthy", "body_site": "stool_feces", "n_samples": 6},
            {"phenotype": "Disease", "body_site": "stool_feces", "n_samples": 7},
        ],
        seed=43,
    )
    reversed_rows = mbbench.generate_from_condition_table(
        bank=bank,
        condition_table=[
            {"phenotype": "Disease", "body_site": "stool_feces", "n_samples": 7},
            {"phenotype": "Healthy", "body_site": "stool_feces", "n_samples": 6},
        ],
        seed=43,
    )
    for phenotype in ("Healthy", "Disease"):
        left = first.metadata["Phenotype_Tokens"].eq(phenotype)
        right = reversed_rows.metadata["Phenotype_Tokens"].eq(phenotype)
        assert first.metadata.loc[left, "source_bank_sample_id"].tolist() == (
            reversed_rows.metadata.loc[right, "source_bank_sample_id"].tolist()
        )
        np.testing.assert_allclose(
            first.abundance.loc[left].iloc[:, 1:].to_numpy(),
            reversed_rows.abundance.loc[right].iloc[:, 1:].to_numpy(),
        )


def test_bank_observed_generation_requires_library_size(tmp_path: Path) -> None:
    bank = mbbench.load_bank(_tiny_bank(tmp_path / "bank"))
    with pytest.raises(ValueError, match="requires a positive integer library_size"):
        mbbench.generate(bank=bank, n_samples=2, output_mode="observed")
    result = mbbench.generate(
        bank=bank,
        n_samples=2,
        output_mode="observed",
        library_size=5000,
    )
    assert result.counts.iloc[:, 1:].sum(axis=1).eq(5000).all()
    with pytest.raises(ValueError, match="use_body_system does not apply"):
        mbbench.generate(bank=bank, n_samples=2, use_body_system=False)


def test_bank_rejects_requests_above_matching_capacity(tmp_path: Path) -> None:
    bank = mbbench.load_bank(_tiny_bank(tmp_path / "bank"))
    with pytest.raises(ValueError, match="80 unused matching rows"):
        mbbench.generate(bank=bank, n_samples=81)


def test_custom_condition_fields_do_not_silently_accept_standard_aliases(
    tmp_path: Path,
) -> None:
    bank = mbbench.load_bank(_tiny_bank(tmp_path / "bank", field="Cohort"))
    result = mbbench.generate(
        bank=bank,
        body_site="stool_feces",
        conditions={"Cohort": "Healthy"},
        n_samples=3,
    )
    assert result.metadata["Cohort"].eq("Healthy").all()
    with pytest.raises(ValueError, match="unknown fields.*phenotype"):
        mbbench.generate_from_condition_table(
            bank=bank,
            condition_table=[
                {
                    "phenotype": "Healthy",
                    "body_site": "stool_feces",
                    "Cohort": "Healthy",
                    "n_samples": 3,
                }
            ],
        )


def test_bank_runs_fixed_depth_benchmark(tmp_path: Path) -> None:
    bank = mbbench.load_bank(_tiny_bank(tmp_path / "bank"))
    result = mbbench.benchmark(
        bank=bank,
        preset="da-null-balanced",
        n_samples=64,
        overrides={"observation.depth_mode": "fixed"},
    )
    assert result.counts.shape[0] == 64
    assert result.metadata["source_bank_sample_id"].is_unique
    assert result.config["generation_source"] == "bank"
    assert result.config["bank"]["path"] == str(bank.root)
    with pytest.raises(ValueError, match="Model selection"):
        mbbench.benchmark(
            bank=bank,
            preset="da-null-balanced",
            overrides={
                "model.guidance_scale": 2.0,
                "observation.depth_mode": "fixed",
            },
        )
    with pytest.raises(ValueError, match="repeat paths in overrides"):
        mbbench.benchmark(
            bank=bank,
            preset="da-null-balanced",
            n_samples=64,
            overrides={
                "design.n_samples": 72,
                "observation.depth_mode": "fixed",
            },
        )


def test_bank_benchmark_is_independent_of_chunk_size(tmp_path: Path) -> None:
    bank = mbbench.load_bank(_tiny_bank(tmp_path / "bank"))
    expected = mbbench.benchmark(
        bank=bank,
        preset="da-null-balanced",
        n_samples=64,
        batch_size=3,
        overrides={"observation.depth_mode": "fixed"},
    )
    streamed = mbbench.benchmark(
        bank=bank,
        preset="da-null-balanced",
        n_samples=64,
        batch_size=11,
        overrides={"observation.depth_mode": "fixed"},
        output_dir=tmp_path / "streamed",
        load=False,
    )
    loaded = mbbench.load_benchmark_bundle(streamed.output_dir)
    for name in (
        "counts",
        "relative_abundance",
        "latent_composition",
        "metadata",
        "feature_truth",
        "sample_truth",
        "batch_neutral_reference",
    ):
        pd.testing.assert_frame_equal(
            getattr(expected, name),
            getattr(loaded, name),
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )


def test_bank_context_manager_closes_memory_maps(tmp_path: Path) -> None:
    with mbbench.load_bank(_tiny_bank(tmp_path / "bank")) as bank:
        abundance_map = bank.abundance_logits._mmap
        support_map = bank.support_logits._mmap
        assert not abundance_map.closed
        assert not support_map.closed
    assert abundance_map.closed
    assert support_map.closed


def test_cli_bank_source_and_resolved_yaml(tmp_path: Path) -> None:
    bank_dir = _tiny_bank(tmp_path / "bank")
    output_dir = tmp_path / "generated"
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate",
            "--bank",
            str(bank_dir),
            "--n-samples",
            "5",
            "--output-dir",
            str(output_dir),
        ]
    )
    validate_model_arguments(args)
    _run_generate(args)
    resolved = yaml.safe_load(
        (output_dir / "generation.yaml").read_text(encoding="utf-8")
    )
    assert resolved["generation_source"] == "bank"
    assert resolved["bank"]["path"] == str(bank_dir.resolve())

    conflict = parser.parse_args(
        [
            "benchmark",
            "--bank",
            str(bank_dir),
            "--preset",
            "da-null-balanced",
            "--model",
            "vae",
        ]
    )
    with pytest.raises(SystemExit, match="--bank cannot be combined"):
        validate_model_arguments(conflict)
