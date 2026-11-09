from __future__ import annotations

import pytest
import yaml

from mbbench.benchmark_config import (
    PRESET_NAMES,
    design_cell_counts,
    load_custom_config,
    resolve_preset,
    show_preset,
    write_config_template,
)


def test_every_preset_is_complete_and_records_sample_count():
    scenarios = [resolve_preset(name) for name in PRESET_NAMES]
    assert all(scenario.design.n_samples == 240 for scenario in scenarios)
    assert all(scenario.design.n_batches == 4 for scenario in scenarios)
    common = resolve_preset("da-strict-common-moderate", {})
    broad = resolve_preset("da-strict-broad-strong", {})
    assert (common.biology.prevalence_min, common.biology.prevalence_max) == (
        0.60,
        0.95,
    )
    assert (broad.biology.prevalence_min, broad.biology.prevalence_max) == (0.30, 0.90)


def test_preset_overrides_are_explicit_and_normalized():
    scenario = resolve_preset(
        "da-strict-common-moderate",
        {"design.n_samples": 120, "biology.target_taxa": ["g1", "g2"]},
    )
    assert scenario.design.n_samples == 120
    assert scenario.biology.target_selection == "specified"
    assert scenario.biology.n_targets == 2
    assert scenario.biology.prevalence_min is None
    assert scenario.cli_overrides == ("design.n_samples", "biology.target_taxa")
    assert design_cell_counts(scenario)["batch_1"] == {"control": 15, "case": 15}

    nested = resolve_preset(
        "da-strict-common-moderate",
        {"design": {"n_samples": 120}, "biology": {"target_taxa": ["g1", "g2"]}},
    )
    assert nested.design.n_samples == scenario.design.n_samples
    assert nested.biology.target_taxa == scenario.biology.target_taxa

    with pytest.raises(ValueError, match="below minimum_cell_count"):
        resolve_preset(
            "da-null-confounded-global",
            {"design.case_fractions": [0.05, 0.4, 0.6, 0.95]},
        )


def test_custom_template_is_a_complete_standalone_config(tmp_path):
    path = write_config_template(tmp_path / "benchmark.yaml")
    scenario = load_custom_config(path)
    assert scenario.preset == "custom"
    assert scenario.design.n_samples == 240
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["runtime"]["seed"] == 42
    native = load_custom_config(payload)
    assert native.as_dict() == scenario.as_dict()

    resolved_null = resolve_preset("da-null-balanced").as_dict()
    assert load_custom_config(resolved_null).as_dict() == resolved_null


def test_conditions_are_generic_and_legacy_configs_are_rejected():
    current = show_preset("da-strict-common-moderate")
    assert current["conditions"] == {
        "common": {"Body_Site_Coarse": "stool_feces"},
        "groups": [
            {
                "label": "control",
                "conditions": {"Phenotype_Tokens": "Healthy"},
            },
            {
                "label": "case",
                "conditions": {"Phenotype_Tokens": "Healthy"},
            },
        ],
    }
    legacy = dict(current)
    legacy["conditions"] = {
        "body_site": "stool_feces",
        "additional": {},
        "groups": [
            {"label": "control", "phenotype": "Healthy", "conditions": {}},
            {"label": "case", "phenotype": "Healthy", "conditions": {}},
        ],
    }
    with pytest.raises(ValueError, match="Unknown conditions fields"):
        load_custom_config(legacy)

    generic = dict(current)
    generic["conditions"] = {
        "common": {},
        "groups": [
            {"label": "control", "conditions": {"Cohort": "control"}},
            {"label": "case", "conditions": {"Cohort": "case"}},
        ],
    }
    scenario = load_custom_config(generic)
    assert scenario.common_conditions == {}
    assert scenario.groups[0].conditions == {"Cohort": "control"}


def test_group_weight_is_rejected(tmp_path):
    path = write_config_template(tmp_path / "benchmark.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["conditions"]["groups"][0]["weight"] = 0.5
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"Unknown conditions.groups\[0\] fields: weight"
    ):
        load_custom_config(path)


def test_duplicate_and_no_op_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="provided more than once"):
        resolve_preset(
            "da-null-balanced",
            {"design": {"n_samples": 120}, "design.n_samples": 160},
        )
    with pytest.raises(ValueError, match="conflicts with nested override"):
        resolve_preset(
            "da-null-balanced",
            {"batch": {}, "batch.mechanisms": []},
        )

    duplicate_yaml = tmp_path / "duplicate.yaml"
    duplicate_yaml.write_text(
        "schema_version: mbbench_benchmark_v1\n"
        "schema_version: mbbench_benchmark_v1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate YAML"):
        load_custom_config(duplicate_yaml)

    with pytest.raises(ValueError, match="require the targeted_abundance"):
        resolve_preset("da-null-balanced", {"batch.target_fraction": 0.2})
    with pytest.raises(ValueError, match="exactly one"):
        resolve_preset(
            "batch-sparse-balanced",
            {
                "batch.target_fraction": 0.2,
                "batch.target_taxa": ["g1", "g2"],
            },
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_preset("da-null-balanced", {"biology.effect_strength": 1.0})
    with pytest.raises(ValueError, match="target_selection is derived"):
        resolve_preset(
            "da-strict-common-moderate",
            {"biology.target_selection": "specified"},
        )
    with pytest.raises(ValueError, match="requires direct biological targets"):
        resolve_preset(
            "batch-sparse-balanced",
            {"batch.allow_target_overlap": True},
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"design.n_samples": 120.5}, "must be an integer"),
        ({"design.n_samples": True}, "not Boolean"),
        ({"design.case_fractions": [True, 0.5]}, "not Boolean"),
        ({"biology.prevalence_min": True}, "not Boolean"),
        ({"batch.target_fraction": True}, "not Boolean"),
    ],
)
def test_numeric_configuration_is_strict(override, message):
    preset = (
        "da-null-confounded-global"
        if "design.case_fractions" in override
        else (
            "batch-sparse-balanced"
            if "batch.target_fraction" in override
            else "da-strict-common-moderate"
        )
    )
    with pytest.raises(ValueError, match=message):
        resolve_preset(preset, override)


def test_observation_mode_specific_settings_are_rejected():
    with pytest.raises(ValueError, match="applies only when depth_mode is fixed"):
        resolve_preset("da-null-balanced", {"observation.library_size": 10000})
    with pytest.raises(ValueError, match="Multinomial observation does not use"):
        resolve_preset(
            "da-null-balanced",
            {
                "observation.model": "multinomial",
                "observation.concentration_median": 500,
            },
        )
    with pytest.raises(ValueError, match="dispersion batch mechanism requires"):
        resolve_preset(
            "batch-global-balanced",
            {"observation.model": "multinomial"},
        )
    fixed = resolve_preset(
        "da-null-balanced", {"observation.concentration_mode": "fixed"}
    )
    assert fixed.observation.concentration_log_sd is None
    with pytest.raises(ValueError, match="Fixed concentration does not use"):
        resolve_preset(
            "da-null-balanced",
            {
                "observation.concentration_mode": "fixed",
                "observation.concentration_log_sd": 0.4,
            },
        )


def test_support_mode_specific_settings_are_resolved_or_rejected():
    deterministic = resolve_preset(
        "da-null-balanced", {"support.mode": "deterministic"}
    )
    assert deterministic.support.threshold == 0.5
    assert deterministic.support.temperature is None
    bernoulli = resolve_preset("da-null-balanced", {"support.mode": "bernoulli"})
    assert bernoulli.support.threshold is None
    assert bernoulli.support.temperature is None
    with pytest.raises(ValueError, match="does not use support.temperature"):
        resolve_preset(
            "da-null-balanced",
            {"support.mode": "deterministic", "support.temperature": 0.6},
        )


def test_batch_mechanisms_can_be_cleared_from_a_preset():
    scenario = resolve_preset("batch-global-balanced", {"batch.mechanisms": []})
    assert scenario.batch.mechanisms == ()
    assert scenario.batch.global_strength == 0
    assert scenario.batch.depth_strength == 0
    assert scenario.batch.dispersion_strength == 0
