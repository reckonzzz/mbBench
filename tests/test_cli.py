from __future__ import annotations

import subprocess
import sys

import pytest
import yaml

from mbbench.cli import build_parser


def test_help_is_lightweight_and_version_is_available():
    import mbbench

    assert {
        "benchmark",
        "download_model",
        "evaluate_batch",
        "evaluate_da",
        "evaluate_da_many",
        "generate_from_condition_table",
        "list_conditions",
        "list_parameters",
        "list_presets",
        "load_model",
        "package_model",
        "show_preset",
        "summarize_batch",
        "summarize_da",
        "write_config_template",
    }.issubset(mbbench.__all__)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mbbench.cli; "
            "print(int('torch' in sys.modules), int('pandas' in sys.modules), int('sklearn' in sys.modules))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "0 0 0"

    version = subprocess.run(
        [sys.executable, "-m", "mbbench.cli", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert version.stdout.strip() == "mbbench 0.1.0"
    parser = build_parser()
    conditions = parser.parse_args(["list-conditions", "--model", "ae"])
    assert conditions.command == "list-conditions"
    with pytest.raises(SystemExit):
        parser.parse_args(["list"])


def test_lazy_public_functions_are_not_shadowed_by_submodules():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib, mbbench; "
            "[importlib.import_module(name) for name in "
            "('mbbench._benchmark', 'mbbench._package_model', "
            "'mbbench._train_diffusion', 'mbbench._train_tokenizer')]; "
            "print(*(callable(getattr(mbbench, name)) for name in "
            "('benchmark', 'package_model', 'train_diffusion', 'train_tokenizer')))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "True True True True"


def test_explicit_field_lists_and_mutual_exclusion():
    parser = build_parser()
    args = parser.parse_args(
        [
            "prepare",
            "--abundance",
            "counts.csv",
            "--metadata",
            "metadata.csv",
            "--output-dir",
            "prepared",
            "--biological-fields",
            "Phenotype",
            "Body_Site",
            "--no-technical-fields",
        ]
    )
    assert args.biological_fields == ["Phenotype", "Body_Site"]
    assert args.technical_fields == []

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "generate",
                "--condition-table",
                "conditions.csv",
                "--condition",
                "Cohort=case",
                "--output-dir",
                "generated",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["benchmark", "--preset", "da-null-balanced", "--config", "custom.yaml"]
        )


def test_benchmark_discovery_is_lightweight_and_custom_yaml_rejects_overrides(tmp_path):
    discovery = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from mbbench.cli import main; "
            "sys.argv=['mbbench','benchmark','--list-presets']; main(); "
            "print('runtime', int('torch' in sys.modules), int('pandas' in sys.modules))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert discovery.stdout.strip().endswith("runtime 0 0")
    template = tmp_path / "benchmark.yaml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mbbench.cli",
            "benchmark",
            "--write-config-template",
            str(template),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    mixed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mbbench.cli",
            "benchmark",
            "--config",
            str(template),
            "--n-samples",
            "100",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert mixed.returncode != 0
    assert "cannot be combined" in mixed.stderr


def test_benchmark_parameter_listing_uses_cli_metadata():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mbbench.cli",
            "benchmark",
            "--preset",
            "da-strict-common-moderate",
            "--list-parameters",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--n-samples\tdesign.n_samples\t240\tpositive integer" in result.stdout
    assert "--use-body-system/--no-use-body-system" in result.stdout
    assert "--paired-neutral-counts/--no-paired-neutral-counts" in result.stdout

    from mbbench import list_parameters

    rows = list_parameters("da-strict-common-moderate")
    sample_count = next(row for row in rows if row["yaml_path"] == "design.n_samples")
    assert sample_count["default"] == 240
    assert sample_count["option"] == "--n-samples"


def test_benchmark_cli_uses_generic_group_conditions():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mbbench.cli",
            "benchmark",
            "--preset",
            "da-null-balanced",
            "--common-condition",
            "Cohort=study_a",
            "--group",
            "control,Phenotype_Tokens=Healthy",
            "--group",
            "case,Phenotype_Tokens=IBD",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = yaml.safe_load(result.stdout)
    assert resolved["conditions"]["common"] == {
        "Body_Site_Coarse": "stool_feces",
        "Cohort": "study_a",
    }
    assert resolved["conditions"]["groups"][1] == {
        "label": "case",
        "conditions": {"Phenotype_Tokens": "IBD"},
    }


def test_dataset_commands_and_generation_memory_options_are_exposed():
    parser = build_parser()
    download = parser.parse_args(
        [
            "download-dataset",
            "da-strict-common-moderate",
            "--seed",
            "42",
            "--dataset-version",
            "latest",
        ]
    )
    assert download.preset == "da-strict-common-moderate"
    assert download.seed == 42

    generate = parser.parse_args(
        ["generate", "--output-dir", "generated", "--max-in-memory-mb", "1024"]
    )
    assert generate.max_in_memory_mb == 1024

    benchmark = parser.parse_args(
        [
            "benchmark",
            "--preset",
            "da-null-balanced",
            "--output-dir",
            "benchmark",
            "--max-in-memory-mb",
            "1024",
        ]
    )
    assert benchmark.max_in_memory_mb == 1024

    batch_eval = parser.parse_args(
        [
            "evaluate-batch",
            "--benchmark",
            "benchmark",
            "--input",
            "combat=corrected.csv",
            "--output-dir",
            "evaluation",
        ]
    )
    assert batch_eval.inputs == ["combat=corrected.csv"]
    assert batch_eval.calculate_scores is False

    scored_batch_eval = parser.parse_args(
        [
            "evaluate-batch",
            "--benchmark",
            "benchmark",
            "--input",
            "combat=corrected.csv",
            "--calculate-scores",
            "--output-dir",
            "evaluation",
        ]
    )
    assert scored_batch_eval.calculate_scores is True

    da_eval = parser.parse_args(
        [
            "evaluate-da",
            "--manifest",
            "manifest.csv",
            "--output-dir",
            "evaluation",
        ]
    )
    assert da_eval.manifest == "manifest.csv"
