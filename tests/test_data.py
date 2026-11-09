from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mbbench.data import MicrobiomeDataset, PreparedData, _random_split, prepare_data
from mbbench.common import new_output_dir


def test_overwrite_replaces_the_complete_output_directory(tmp_path):
    output = new_output_dir(tmp_path / "run")
    (output / "stale.txt").write_text("old", encoding="utf-8")
    replaced = new_output_dir(output, overwrite=True)
    assert replaced == output
    assert list(replaced.iterdir()) == []
    with pytest.raises(TypeError, match="overwrite must be True or False"):
        new_output_dir(output, overwrite="False")
    file_path = tmp_path / "result"
    file_path.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="existing file"):
        new_output_dir(file_path)


def test_prepare_data_normalizes_and_retains_supplied_split(tmp_path):
    samples = [f"s{i}" for i in range(6)]
    abundance = pd.DataFrame(
        {"Sample": samples, "g1": [1, 2, 3, 4, 5, 6], "g2": [3, 2, 1, 2, 1, 4]}
    )
    metadata = pd.DataFrame(
        {
            "Sample": samples,
            "Phenotype_Tokens": ["Healthy"] * 3 + ["IBD"] * 3,
            "Body_Site_Coarse": ["stool_feces"] * 6,
            "my_split": ["train", "train", "validation", "train", "validation", "test"],
        }
    )
    abundance.to_csv(tmp_path / "abundance.csv", index=False)
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    pd.DataFrame(
        {"body_site": ["stool_feces"], "body_system": ["digestive_gut"]}
    ).to_csv(tmp_path / "body_system.csv", index=False)
    output = prepare_data(
        abundance_path=tmp_path / "abundance.csv",
        metadata_path=tmp_path / "metadata.csv",
        output_dir=tmp_path / "prepared",
        split_column="my_split",
        body_system_map=tmp_path / "body_system.csv",
        technical_fields=(),
    )
    prepared = PreparedData.load(output)
    assert prepared.abundance[prepared.taxa].sum(axis=1).round(12).eq(1.0).all()
    assert set(prepared.metadata["split"]) == {"train", "validation", "test"}
    assert prepared.metadata["Body_System_Model"].eq("digestive_gut").all()
    assert "Body_System_Model" in prepared.vocabularies


def test_prepare_data_rejects_relative_abundance(tmp_path):
    abundance = pd.DataFrame({"Sample": ["s1"], "g1": [0.75], "g2": [0.25]})
    metadata = pd.DataFrame(
        {
            "Sample": ["s1"],
            "Phenotype_Tokens": ["Healthy"],
            "Body_Site_Coarse": ["stool_feces"],
        }
    )
    abundance.to_csv(tmp_path / "abundance.csv", index=False)
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    with pytest.raises(ValueError, match="genus counts"):
        prepare_data(
            abundance_path=tmp_path / "abundance.csv",
            metadata_path=tmp_path / "metadata.csv",
            output_dir=tmp_path / "prepared",
        )


def test_prepare_data_accepts_missing_phenotype_and_custom_technical_fields(tmp_path):
    samples = [f"s{i}" for i in range(6)]
    abundance = pd.DataFrame(
        {"Sample": samples, "g1": [1, 2, 3, 4, 5, 6], "g2": [2, 3, 4, 5, 6, 7]}
    )
    metadata = pd.DataFrame(
        {
            "Sample": samples,
            "Phenotype_Tokens": ["Healthy", None, "Healthy", "IBD", "IBD", "IBD"],
            "Body_Site_Coarse": ["stool_feces"] * 6,
            "Body_System_Model": ["user_value"] * 6,
            "Platform": ["A", "A", "B", "B", "A", "B"],
            "split": ["train", "train", "validation", "train", "validation", "test"],
        }
    )
    abundance.to_csv(tmp_path / "abundance.csv", index=False)
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    output = prepare_data(
        abundance_path=tmp_path / "abundance.csv",
        metadata_path=tmp_path / "metadata.csv",
        output_dir=tmp_path / "prepared",
        split_column="split",
        technical_fields=("Platform",),
    )
    prepared = PreparedData.load(output)
    assert prepared.technical_fields == ("Platform",)
    assert prepared.vocabularies["Platform"]["values"] == [
        "__NULL__",
        "__UNK__",
        "A",
        "B",
    ]
    assert pd.isna(prepared.metadata.loc["s1", "Phenotype_Tokens"])
    assert "Body_System_Model" not in prepared.metadata


def test_prepare_data_accepts_custom_biological_fields(tmp_path):
    samples = [f"s{i}" for i in range(6)]
    pd.DataFrame(
        {"Sample": samples, "g1": [1, 2, 3, 4, 5, 6], "g2": [2, 3, 4, 5, 6, 7]}
    ).to_csv(tmp_path / "abundance.csv", index=False)
    pd.DataFrame(
        {
            "Sample": samples,
            "Disease_State": ["control", "control", "case", "case", "case", "control"],
            "Treatment_Tokens": [
                "none",
                "none",
                "drug_a|diet",
                "drug_a",
                "diet",
                "none",
            ],
            "split": ["train", "train", "train", "validation", "validation", "test"],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)
    output = prepare_data(
        abundance_path=tmp_path / "abundance.csv",
        metadata_path=tmp_path / "metadata.csv",
        output_dir=tmp_path / "prepared",
        split_column="split",
        biological_fields=("Disease_State", "Treatment_Tokens"),
        multilabel_fields=("Treatment_Tokens",),
        technical_fields=(),
    )
    prepared = PreparedData.load(output)
    assert prepared.biological_fields == ("Disease_State", "Treatment_Tokens")
    assert prepared.multilabel_fields == ("Treatment_Tokens",)
    assert prepared.vocabularies["Disease_State"]["type"] == "categorical"
    assert prepared.vocabularies["Treatment_Tokens"]["tokens"] == [
        "diet",
        "drug_a",
        "none",
    ]


def test_vocabularies_are_built_from_train_only(tmp_path):
    samples = [f"s{i}" for i in range(6)]
    pd.DataFrame(
        {"Sample": samples, "g1": [1, 2, 3, 4, 5, 6], "g2": [2, 3, 4, 5, 6, 7]}
    ).to_csv(tmp_path / "abundance.csv", index=False)
    pd.DataFrame(
        {
            "Sample": samples,
            "Phenotype_Tokens": ["Healthy"] * 6,
            "Body_Site_Coarse": ["stool_feces"] * 6,
            "Cohort": [
                "train_a",
                "train_a",
                "validation_only",
                "train_a",
                "validation_only",
                "test_only",
            ],
            "split": ["train", "train", "validation", "train", "validation", "test"],
        }
    ).to_csv(tmp_path / "metadata.csv", index=False)
    output = prepare_data(
        abundance_path=tmp_path / "abundance.csv",
        metadata_path=tmp_path / "metadata.csv",
        output_dir=tmp_path / "prepared",
        split_column="split",
        technical_fields=("Cohort",),
    )
    prepared = PreparedData.load(output)
    assert prepared.vocabularies["Cohort"]["values"] == [
        "__NULL__",
        "__UNK__",
        "train_a",
    ]
    validation = MicrobiomeDataset(prepared, "validation")
    assert validation.categorical["Cohort"].tolist() == [1, 1]


def test_split_json_rejects_duplicate_and_extra_sample_ids(tmp_path):
    package_root = Path(__file__).parents[1]
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"train": ["s01", "s02"], "test": ["s01"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="assigns sample 's01' more than once"):
        prepare_data(
            abundance_path=package_root / "examples" / "abundance_small.csv",
            metadata_path=package_root / "examples" / "metadata_small.csv",
            output_dir=tmp_path / "prepared",
            split_file=split_path,
        )


def test_automatic_split_canonicalizes_multilabel_token_order():
    metadata = pd.DataFrame(
        {
            "Phenotype_Tokens": (["A|B", "B|A", "A;B", "B;A"] * 5),
            "Body_Site_Coarse": ["site"] * 20,
        },
        index=[f"s{index}" for index in range(20)],
    )
    split = _random_split(
        metadata,
        ("Phenotype_Tokens", "Body_Site_Coarse"),
        ("Phenotype_Tokens",),
        42,
    )
    assert split.value_counts().to_dict() == {"train": 16, "validation": 2, "test": 2}


def test_prepare_validation_failure_preserves_existing_output(tmp_path):
    package_root = Path(__file__).parents[1]
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    taxonomy = tmp_path / "taxonomy.csv"
    pd.DataFrame({"taxon": ["g1"]}).to_csv(taxonomy, index=False)
    with pytest.raises(ValueError, match="taxon and family"):
        prepare_data(
            abundance_path=package_root / "examples" / "abundance_small.csv",
            metadata_path=package_root / "examples" / "metadata_small.csv",
            output_dir=output,
            split_column="split",
            taxonomy_path=taxonomy,
            overwrite=True,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"
