from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.resources import files
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import yaml

from .yaml_io import load_yaml


SCHEMA_VERSION = "mbbench_benchmark_v1"
PRESET_NAMES = (
    "da-null-balanced",
    "da-null-confounded-global",
    "da-strict-common-moderate",
    "da-strict-broad-strong",
    "da-strict-common-strong",
    "da-strict-confounded-global",
    "batch-null-balanced",
    "batch-global-balanced",
    "batch-global-confounded",
    "batch-sparse-balanced",
    "batch-sparse-confounded",
)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping.")
    return {str(key): item for key, item in value.items()}


def _reject_unknown(values: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {', '.join(unknown)}.")


def _choice(value: object, choices: set[str], name: str) -> str:
    result = str(value).strip()
    if result not in choices:
        raise ValueError(f"{name} must be one of: {', '.join(sorted(choices))}.")
    return result


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not Boolean.")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = f" and at least {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be finite{qualifier}.")
    return result


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not Boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer.") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer.")
    result = int(numeric)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _fraction_sequence(value: object, name: str) -> tuple[float, ...] | None:
    if value is None or value == "":
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a YAML list of finite numbers.")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))


def _optional_taxa(value: object, name: str) -> tuple[str, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError(
            f"{name} must be null, a comma-separated string, or a YAML list."
        )
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique nonempty taxon names.")
    return tuple(values)


@dataclass(frozen=True)
class ModelSettings:
    type: str
    directory: str | None
    guidance_scale: float
    use_body_system: bool | None


@dataclass(frozen=True)
class GroupSettings:
    label: str
    conditions: dict[str, object]


@dataclass(frozen=True)
class DesignSettings:
    n_samples: int
    n_batches: int
    mode: str
    case_fractions: tuple[float, ...] | None
    group_fractions: tuple[tuple[float, ...], ...] | None
    minimum_cell_count: int


@dataclass(frozen=True)
class BiologySettings:
    mode: str
    target_selection: str
    n_targets: int | None
    target_taxa: tuple[str, ...] | None
    prevalence_min: float | None
    prevalence_max: float | None
    effect_strength: float | None
    effect_directions: str | None


@dataclass(frozen=True)
class BatchSettings:
    mechanisms: tuple[str, ...]
    global_strength: float
    target_fraction: float | None
    target_taxa: tuple[str, ...] | None
    target_strength: float
    prevalence_fraction: float | None
    prevalence_taxa: tuple[str, ...] | None
    prevalence_strength: float
    depth_strength: float
    dispersion_strength: float
    allow_target_overlap: bool


@dataclass(frozen=True)
class SupportSettings:
    mode: str
    threshold: float | None
    temperature: float | None


@dataclass(frozen=True)
class ObservationSettings:
    model: str
    depth_mode: str
    library_size: int | None
    depth_min: int
    depth_max: int
    concentration_mode: str | None
    concentration_median: float | None
    concentration_log_sd: float | None
    concentration_min: float | None
    concentration_max: float | None
    paired_neutral_counts: bool


@dataclass(frozen=True)
class RuntimeSettings:
    seed: int
    batch_size: int
    device: str


@dataclass(frozen=True)
class ResolvedBenchmarkScenario:
    schema_version: str
    preset: str
    model: ModelSettings
    common_conditions: dict[str, object]
    groups: tuple[GroupSettings, ...]
    design: DesignSettings
    biology: BiologySettings
    batch: BatchSettings
    support: SupportSettings
    observation: ObservationSettings
    runtime: RuntimeSettings
    cli_overrides: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["conditions"] = {
            "common": payload.pop("common_conditions"),
            "groups": payload.pop("groups"),
        }
        payload["cli_overrides"] = list(self.cli_overrides)

        def convert(value: object) -> object:
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {str(key): convert(item) for key, item in value.items()}
            return value

        return convert(payload)  # type: ignore[return-value]


def _largest_remainder_counts(total: int, fractions: list[float]) -> tuple[int, ...]:
    desired = [total * fraction for fraction in fractions]
    counts = [math.floor(value) for value in desired]
    order = sorted(
        range(len(fractions)), key=lambda index: -(desired[index] - counts[index])
    )
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return tuple(counts)


def design_cell_count_matrix(
    scenario: ResolvedBenchmarkScenario,
) -> tuple[tuple[int, ...], ...]:
    design = scenario.design
    n_groups = len(scenario.groups)
    batch_sizes = [design.n_samples // design.n_batches] * design.n_batches
    for index in range(design.n_samples % design.n_batches):
        batch_sizes[index] += 1
    rows: list[tuple[int, ...]] = []
    for batch_index, batch_size in enumerate(batch_sizes):
        if design.mode == "balanced":
            fractions = [1.0 / n_groups] * n_groups
        elif n_groups == 2:
            case_fraction = float(design.case_fractions[batch_index])
            fractions = [1.0 - case_fraction, case_fraction]
        else:
            fractions = list(design.group_fractions[batch_index])
        counts = _largest_remainder_counts(batch_size, fractions)
        if min(counts) < design.minimum_cell_count:
            raise ValueError(
                f"The batch-by-group design contains a cell with {min(counts)} samples, "
                f"below minimum_cell_count={design.minimum_cell_count}."
            )
        rows.append(counts)
    return tuple(rows)


def design_cell_counts(
    scenario: ResolvedBenchmarkScenario,
) -> dict[str, dict[str, int]]:
    matrix = design_cell_count_matrix(scenario)
    return {
        f"batch_{batch_index + 1}": {
            group.label: matrix[batch_index][group_index]
            for group_index, group in enumerate(scenario.groups)
        }
        for batch_index in range(scenario.design.n_batches)
    }


def scenario_truth_applicability(
    scenario: ResolvedBenchmarkScenario,
) -> dict[str, object]:
    has_biological_targets = scenario.biology.mode not in {"null", "natural"}
    strict_null = scenario.biology.mode in {"null", "strict"}
    has_batch_effects = bool(scenario.batch.mechanisms)
    return {
        "da_power": has_biological_targets,
        "da_fdr": strict_null,
        "batch_recovery": has_batch_effects,
        "biological_preservation": scenario.biology.mode != "null",
        "da_truth_mode": (
            "strict_relative_abundance_targets"
            if scenario.biology.mode == "strict"
            else (
                "global_null"
                if scenario.biology.mode == "null"
                else "direct_targets_only"
            )
        ),
        "direct_biological_targets": has_biological_targets,
        "model_native_biology": scenario.biology.mode == "natural",
        "batch_abundance_offsets": bool(
            {"global", "targeted_abundance"} & set(scenario.batch.mechanisms)
        ),
        "batch_support_offsets": "prevalence" in scenario.batch.mechanisms,
        "batch_depth_offsets": "depth" in scenario.batch.mechanisms,
        "batch_dispersion_offsets": "dispersion" in scenario.batch.mechanisms,
    }


def _preset_path(name: str):
    if name not in PRESET_NAMES:
        raise ValueError(
            f"Unknown preset {name!r}. Use --list-presets to see available presets."
        )
    return files("mbbench.defaults").joinpath("benchmark", f"{name}.yaml")


def load_preset(name: str) -> dict[str, Any]:
    raw = yaml.safe_load(_preset_path(name).read_text(encoding="utf-8"))
    return _mapping(raw, f"preset {name}")


def show_preset(name: str) -> dict[str, Any]:
    """Return one complete benchmark preset."""

    return deepcopy(load_preset(name))


def list_presets() -> tuple[tuple[str, str], ...]:
    """Return ``(preset, description)`` pairs for all built-in benchmarks."""

    rows = []
    for name in PRESET_NAMES:
        raw = load_preset(name)
        rows.append((name, str(raw.get("description", "")).strip()))
    return tuple(rows)


def write_config_template(path: str | Path) -> Path:
    """Copy the complete custom benchmark YAML template to a new path."""

    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Template target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source = files("mbbench.defaults").joinpath("benchmark_custom.yaml")
    with source.open("rb") as source_handle, target.open("wb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
    return target


def _set_path(payload: dict[str, Any], path: str, value: object) -> None:
    parts = path.split(".")
    current = payload
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            raise ValueError(f"Cannot override {path}; {part} is not a mapping.")
        current = nested
    current[parts[-1]] = value


def _flatten_overrides(
    values: Mapping[str, object],
    prefix: str = "",
    flattened: dict[str, object] | None = None,
) -> dict[str, object]:
    if flattened is None:
        flattened = {}
    for raw_key, value in values.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping) and "." not in key:
            if value:
                _flatten_overrides(value, path, flattened)
            elif path in flattened:
                raise ValueError(f"Override {path!r} was provided more than once.")
            else:
                flattened[path] = {}
        else:
            if path in flattened:
                raise ValueError(f"Override {path!r} was provided more than once.")
            flattened[path] = value
    if not prefix:
        for path in flattened:
            parts = path.split(".")
            for length in range(1, len(parts)):
                parent = ".".join(parts[:length])
                if parent in flattened:
                    raise ValueError(
                        f"Override {parent!r} conflicts with nested override {path!r}."
                    )
    return flattened


def resolve_preset(
    name: str, overrides: Mapping[str, object] | None = None
) -> ResolvedBenchmarkScenario:
    raw = deepcopy(load_preset(name))
    normalized = _flatten_overrides(overrides or {})
    biology_taxa_override = normalized.get("biology.target_taxa")
    if biology_taxa_override is not None and biology_taxa_override != "":
        for path in (
            "biology.n_targets",
            "biology.prevalence_min",
            "biology.prevalence_max",
        ):
            if path not in normalized:
                _set_path(raw, path, None)
    for taxa_path, fraction_path in (
        ("batch.target_taxa", "batch.target_fraction"),
        ("batch.prevalence_taxa", "batch.prevalence_fraction"),
    ):
        taxa_override = normalized.get(taxa_path)
        if (
            taxa_override is not None
            and taxa_override != ""
            and fraction_path not in normalized
        ):
            _set_path(raw, fraction_path, None)
        if normalized.get(fraction_path) is not None and taxa_path not in normalized:
            _set_path(raw, taxa_path, None)
    if "batch.mechanisms" in normalized:
        requested = set(normalized["batch.mechanisms"] or [])
        mechanism_fields = {
            "global": ("global_strength",),
            "targeted_abundance": (
                "target_fraction",
                "target_taxa",
                "target_strength",
            ),
            "prevalence": (
                "prevalence_fraction",
                "prevalence_taxa",
                "prevalence_strength",
            ),
            "depth": ("depth_strength",),
            "dispersion": ("dispersion_strength",),
        }
        for mechanism, fields_for_mechanism in mechanism_fields.items():
            if mechanism not in requested:
                for field in fields_for_mechanism:
                    path = f"batch.{field}"
                    if path not in normalized:
                        _set_path(
                            raw,
                            path,
                            None if "fraction" in field or "taxa" in field else 0.0,
                        )
    if normalized.get("observation.model") == "multinomial":
        for field in (
            "concentration_mode",
            "concentration_median",
            "concentration_log_sd",
            "concentration_min",
            "concentration_max",
        ):
            path = f"observation.{field}"
            if path not in normalized:
                _set_path(raw, path, None)
    if normalized.get("observation.concentration_mode") == "fixed":
        if "observation.concentration_log_sd" not in normalized:
            _set_path(raw, "observation.concentration_log_sd", None)
    if "support.mode" in normalized:
        mode = normalized["support.mode"]
        inactive_support_fields = {
            "deterministic": ("temperature",),
            "bernoulli": ("threshold", "temperature"),
        }.get(mode, ())
        for field in inactive_support_fields:
            path = f"support.{field}"
            if path not in normalized:
                _set_path(raw, path, None)
    if (
        normalized.get("observation.depth_mode") == "model-donor"
        and "observation.library_size" not in normalized
    ):
        _set_path(raw, "observation.library_size", None)
    for path, value in normalized.items():
        _set_path(raw, path, value)
    return parse_scenario(raw, cli_overrides=tuple(normalized))


def load_custom_config(
    config: str | Path | Mapping[str, object]
) -> ResolvedBenchmarkScenario:
    if isinstance(config, Mapping):
        raw = deepcopy(dict(config))
    else:
        source = Path(config).expanduser().resolve()
        raw = load_yaml(source)
    return parse_scenario(_mapping(raw, "custom benchmark config"))


def _parse_model(raw: object) -> ModelSettings:
    model_raw = _mapping(raw, "model")
    _reject_unknown(
        model_raw, {"type", "directory", "guidance_scale", "use_body_system"}, "model"
    )
    model_type = _choice(
        model_raw.get("type", "ae"), {"ae", "vae", "custom"}, "model.type"
    )
    directory_raw = model_raw.get("directory")
    directory = None if directory_raw in {None, ""} else str(directory_raw)
    if model_type == "custom" and directory is None:
        raise ValueError("model.directory is required when model.type is custom.")
    if model_type != "custom" and directory is not None:
        raise ValueError("model.directory is only valid when model.type is custom.")
    body_system_raw = model_raw.get("use_body_system", "auto")
    if body_system_raw in {None, "auto"}:
        use_body_system = None
    elif type(body_system_raw) is bool:
        use_body_system = body_system_raw
    else:
        raise ValueError("model.use_body_system must be true, false, or auto.")
    return ModelSettings(
        type=model_type,
        directory=directory,
        guidance_scale=_finite(
            model_raw.get("guidance_scale", 1.25), "model.guidance_scale", minimum=0
        ),
        use_body_system=use_body_system,
    )


def _parse_conditions(
    raw: object,
) -> tuple[dict[str, object], tuple[GroupSettings, ...]]:
    conditions_raw = _mapping(raw, "conditions")
    _reject_unknown(conditions_raw, {"common", "groups"}, "conditions")
    common = _mapping(conditions_raw.get("common", {}), "conditions.common")
    groups_raw = conditions_raw.get("groups")
    if not isinstance(groups_raw, list) or len(groups_raw) < 2:
        raise ValueError("conditions.groups must contain at least two groups.")
    groups: list[GroupSettings] = []
    for index, item in enumerate(groups_raw):
        group_raw = _mapping(item, f"conditions.groups[{index}]")
        _reject_unknown(
            group_raw,
            {"label", "conditions"},
            f"conditions.groups[{index}]",
        )
        label = str(group_raw.get("label", "")).strip()
        if not label:
            raise ValueError("Every group requires a nonempty label.")
        group_conditions = _mapping(
            group_raw.get("conditions", {}), f"conditions.groups[{index}].conditions"
        )
        overlap = sorted(set(common) & set(group_conditions))
        if overlap:
            raise ValueError(
                "Condition fields are repeated in common and group settings: "
                + ", ".join(overlap)
                + "."
            )
        groups.append(GroupSettings(label, group_conditions))
    labels = [group.label for group in groups]
    if len(labels) != len(set(labels)):
        raise ValueError("Group labels must be unique.")
    return common, tuple(groups)


def _parse_design(raw: object, groups: tuple[GroupSettings, ...]) -> DesignSettings:
    design_raw = _mapping(raw, "design")
    _reject_unknown(
        design_raw,
        {
            "n_samples",
            "n_batches",
            "mode",
            "case_fractions",
            "group_fractions",
            "minimum_cell_count",
        },
        "design",
    )
    n_samples = _integer(
        design_raw.get("n_samples", 240), "design.n_samples", minimum=1
    )
    n_batches = _integer(design_raw.get("n_batches", 4), "design.n_batches", minimum=1)
    minimum_cell_count = _integer(
        design_raw.get("minimum_cell_count", 8),
        "design.minimum_cell_count",
        minimum=1,
    )
    if n_samples < n_batches * len(groups) * minimum_cell_count:
        raise ValueError(
            f"design.n_samples={n_samples} cannot provide {minimum_cell_count} samples in each "
            f"of {n_batches * len(groups)} batch-by-group cells."
        )
    design_mode = _choice(
        design_raw.get("mode", "balanced"), {"balanced", "confounded"}, "design.mode"
    )
    case_raw = design_raw.get("case_fractions")
    case_fractions = _fraction_sequence(case_raw, "design.case_fractions")
    group_raw = design_raw.get("group_fractions")
    if group_raw is None or group_raw == "":
        group_fractions = None
    elif not isinstance(group_raw, (list, tuple)):
        raise ValueError("design.group_fractions must be a YAML list of rows.")
    else:
        group_fractions = tuple(
            _fraction_sequence(row, f"design.group_fractions[{index}]") or ()
            for index, row in enumerate(group_raw)
        )
    if design_mode == "balanced" and (
        case_fractions is not None or group_fractions is not None
    ):
        raise ValueError(
            "Balanced design must not provide case_fractions or group_fractions."
        )
    if design_mode == "confounded":
        if len(groups) == 2:
            if case_fractions is None or group_fractions is not None:
                raise ValueError(
                    "Binary confounded design requires case_fractions only."
                )
            if len(case_fractions) != n_batches or any(
                not 0 < value < 1 for value in case_fractions
            ):
                raise ValueError(
                    "case_fractions must contain one value in (0, 1) per batch."
                )
        else:
            if group_fractions is None or case_fractions is not None:
                raise ValueError(
                    "Multiclass confounded design requires group_fractions only."
                )
            if len(group_fractions) != n_batches:
                raise ValueError("group_fractions must contain one row per batch.")
            for row in group_fractions:
                if (
                    len(row) != len(groups)
                    or any(value <= 0 for value in row)
                    or not math.isclose(sum(row), 1, abs_tol=1e-8)
                ):
                    raise ValueError(
                        "Every group_fractions row must be positive and sum to one."
                    )
    return DesignSettings(
        n_samples=n_samples,
        n_batches=n_batches,
        mode=design_mode,
        case_fractions=case_fractions,
        group_fractions=group_fractions,
        minimum_cell_count=minimum_cell_count,
    )


def _parse_biology(raw: object, n_groups: int) -> BiologySettings:
    biology_raw = _mapping(raw, "biology")
    _reject_unknown(
        biology_raw,
        {
            "mode",
            "n_targets",
            "target_taxa",
            "prevalence_min",
            "prevalence_max",
            "effect_strength",
            "effect_directions",
            "target_selection",
        },
        "biology",
    )
    biology_mode = _choice(
        biology_raw.get("mode", "null"),
        {"null", "strict", "abundance", "prevalence", "mixed", "natural"},
        "biology.mode",
    )
    if biology_mode != "natural" and n_groups != 2:
        raise ValueError(
            "Artificial biological modes currently require exactly two groups."
        )
    target_taxa = _optional_taxa(biology_raw.get("target_taxa"), "biology.target_taxa")
    requested_selection = biology_raw.get("target_selection")
    if biology_mode in {"null", "natural"}:
        inactive = sorted(
            field
            for field in (
                "n_targets",
                "target_taxa",
                "prevalence_min",
                "prevalence_max",
                "effect_strength",
                "effect_directions",
            )
            if biology_raw.get(field) is not None
        )
        if inactive:
            raise ValueError(
                f"biology.mode={biology_mode!r} cannot be combined with: "
                + ", ".join(inactive)
                + "."
            )
        if requested_selection not in {None, "none"}:
            raise ValueError(
                f"biology.target_selection must be 'none' when biology.mode={biology_mode!r}."
            )
        return BiologySettings(biology_mode, "none", None, None, None, None, None, None)
    else:
        if target_taxa and biology_raw.get("n_targets") is not None:
            raise ValueError(
                "Specify biology.target_taxa or biology.n_targets, not both."
            )
        if target_taxa and any(
            biology_raw.get(field) is not None
            for field in ("prevalence_min", "prevalence_max")
        ):
            raise ValueError(
                "biology prevalence bounds apply only to automatic target selection."
            )
        n_targets = (
            len(target_taxa)
            if target_taxa
            else _integer(
                biology_raw.get("n_targets", 12), "biology.n_targets", minimum=1
            )
        )
        if n_targets <= 0 or (biology_mode == "strict" and n_targets % 2):
            raise ValueError(
                "biology.n_targets must be positive and even in strict mode."
            )
        if target_taxa:
            target_selection = "specified"
            prevalence_min = prevalence_max = None
        else:
            target_selection = "automatic"
            prevalence_min = _finite(
                biology_raw.get("prevalence_min", 0.30),
                "biology.prevalence_min",
            )
            prevalence_max = _finite(
                biology_raw.get("prevalence_max", 0.90),
                "biology.prevalence_max",
            )
            if not 0 < prevalence_min < prevalence_max < 1:
                raise ValueError(
                    "Target prevalence bounds must satisfy 0 < min < max < 1."
                )
        if requested_selection is not None and requested_selection != target_selection:
            raise ValueError(
                "biology.target_selection is derived from target_taxa and must be "
                f"{target_selection!r} for this configuration."
            )
        effect_strength = _finite(
            biology_raw.get(
                "effect_strength", math.log(2) if biology_mode == "strict" else 1.0
            ),
            "biology.effect_strength",
            minimum=0,
        )
        if effect_strength <= 0:
            raise ValueError(
                "biology.effect_strength must be positive when targets are enabled."
            )
        directions = _choice(
            biology_raw.get("effect_directions", "balanced"),
            {"balanced", "random-balanced", "increase"},
            "biology.effect_directions",
        )
        if biology_mode == "strict" and directions == "increase":
            raise ValueError(
                "Strict targets require balanced positive and negative directions."
            )
        return BiologySettings(
            biology_mode,
            target_selection,
            n_targets,
            target_taxa,
            prevalence_min,
            prevalence_max,
            effect_strength,
            directions,
        )


def _parse_batch(raw: object, n_batches: int) -> BatchSettings:
    batch_raw = _mapping(raw, "batch")
    _reject_unknown(
        batch_raw,
        {
            "mechanisms",
            "global_strength",
            "target_fraction",
            "target_taxa",
            "target_strength",
            "prevalence_fraction",
            "prevalence_taxa",
            "prevalence_strength",
            "depth_strength",
            "dispersion_strength",
            "allow_target_overlap",
        },
        "batch",
    )
    raw_mechanisms = batch_raw.get("mechanisms", [])
    if not isinstance(raw_mechanisms, list) or any(
        not isinstance(value, str) for value in raw_mechanisms
    ):
        raise ValueError("batch.mechanisms must be a YAML list of names.")
    mechanisms = tuple(str(value).strip() for value in raw_mechanisms)
    allowed_mechanisms = {
        "global",
        "targeted_abundance",
        "prevalence",
        "depth",
        "dispersion",
    }
    unknown_mechanisms = sorted(set(mechanisms) - allowed_mechanisms)
    if unknown_mechanisms:
        raise ValueError(
            "Unknown batch mechanisms: " + ", ".join(unknown_mechanisms) + "."
        )
    if len(mechanisms) != len(set(mechanisms)):
        raise ValueError("batch.mechanisms cannot contain duplicates.")
    if "global" in mechanisms and "targeted_abundance" in mechanisms:
        raise ValueError(
            "batch.mechanisms cannot combine global and targeted_abundance."
        )
    if mechanisms and n_batches < 2:
        raise ValueError("Batch effects require at least two batches.")

    batch_target_taxa = _optional_taxa(
        batch_raw.get("target_taxa"), "batch.target_taxa"
    )
    prevalence_taxa = _optional_taxa(
        batch_raw.get("prevalence_taxa"), "batch.prevalence_taxa"
    )
    target_fraction_raw = batch_raw.get("target_fraction")
    prevalence_fraction_raw = batch_raw.get("prevalence_fraction")
    target_fraction = (
        None
        if target_fraction_raw is None
        else _finite(target_fraction_raw, "batch.target_fraction")
    )
    prevalence_fraction = (
        None
        if prevalence_fraction_raw is None
        else _finite(prevalence_fraction_raw, "batch.prevalence_fraction")
    )
    for value, name in (
        (target_fraction, "target_fraction"),
        (prevalence_fraction, "prevalence_fraction"),
    ):
        if value is not None and not 0 < value <= 1:
            raise ValueError(f"batch.{name} must lie in (0, 1].")
    if "targeted_abundance" in mechanisms and (target_fraction is None) == (
        batch_target_taxa is None
    ):
        raise ValueError(
            "targeted_abundance requires exactly one of batch.target_fraction or batch.target_taxa."
        )
    if "prevalence" in mechanisms and (prevalence_fraction is None) == (
        prevalence_taxa is None
    ):
        raise ValueError(
            "prevalence requires exactly one of batch.prevalence_fraction or batch.prevalence_taxa."
        )
    if "targeted_abundance" not in mechanisms and (
        target_fraction is not None or batch_target_taxa is not None
    ):
        raise ValueError(
            "batch.target_fraction and batch.target_taxa require the targeted_abundance mechanism."
        )
    if "prevalence" not in mechanisms and (
        prevalence_fraction is not None or prevalence_taxa is not None
    ):
        raise ValueError(
            "batch.prevalence_fraction and batch.prevalence_taxa require the prevalence mechanism."
        )

    strength_defaults = {
        "global_strength": 0.35,
        "target_strength": 0.35,
        "prevalence_strength": 0.80,
        "depth_strength": 0.20,
        "dispersion_strength": 0.20,
    }
    strength_mechanisms = {
        "global_strength": "global",
        "target_strength": "targeted_abundance",
        "prevalence_strength": "prevalence",
        "depth_strength": "depth",
        "dispersion_strength": "dispersion",
    }
    strengths: dict[str, float] = {}
    for field, mechanism in strength_mechanisms.items():
        active = mechanism in mechanisms
        if field in batch_raw:
            value = _finite(batch_raw[field], f"batch.{field}", minimum=0)
            if not active and value != 0:
                raise ValueError(f"batch.{field} requires the {mechanism} mechanism.")
        else:
            value = strength_defaults[field] if active else 0.0
        if active and value <= 0:
            raise ValueError(
                f"batch.{field} must be positive when {mechanism} is enabled."
            )
        strengths[field] = value

    allow_overlap = batch_raw.get("allow_target_overlap", False)
    if type(allow_overlap) is not bool:
        raise ValueError("batch.allow_target_overlap must be true or false.")
    if allow_overlap and not {
        "targeted_abundance",
        "prevalence",
    }.intersection(mechanisms):
        raise ValueError(
            "batch.allow_target_overlap only applies to targeted batch mechanisms."
        )
    return BatchSettings(
        mechanisms=mechanisms,
        global_strength=strengths["global_strength"],
        target_fraction=target_fraction,
        target_taxa=batch_target_taxa,
        target_strength=strengths["target_strength"],
        prevalence_fraction=prevalence_fraction,
        prevalence_taxa=prevalence_taxa,
        prevalence_strength=strengths["prevalence_strength"],
        depth_strength=strengths["depth_strength"],
        dispersion_strength=strengths["dispersion_strength"],
        allow_target_overlap=allow_overlap,
    )


def _parse_support(raw: object) -> SupportSettings:
    support_raw = _mapping(raw, "support")
    _reject_unknown(support_raw, {"mode", "threshold", "temperature"}, "support")
    mode = _choice(
        support_raw.get("mode", "temperature-bernoulli"),
        {"deterministic", "bernoulli", "temperature-bernoulli"},
        "support.mode",
    )
    if mode == "bernoulli":
        inactive = [
            field
            for field in ("threshold", "temperature")
            if support_raw.get(field) is not None
        ]
        if inactive:
            raise ValueError(
                "support.mode='bernoulli' does not use: "
                + ", ".join(f"support.{field}" for field in inactive)
                + "."
            )
        return SupportSettings(mode, None, None)
    threshold = _finite(support_raw.get("threshold", 0.5), "support.threshold")
    if not 0 < threshold < 1:
        raise ValueError("support.threshold must lie in (0, 1).")
    if mode == "deterministic":
        if support_raw.get("temperature") is not None:
            raise ValueError(
                "support.mode='deterministic' does not use support.temperature."
            )
        return SupportSettings(mode, threshold, None)
    temperature = _finite(
        support_raw.get("temperature", 0.6), "support.temperature", minimum=0
    )
    if temperature <= 0:
        raise ValueError(
            "support.temperature must be positive in temperature-bernoulli mode."
        )
    return SupportSettings(mode, threshold, temperature)


def _parse_observation(raw: object) -> ObservationSettings:
    observation_raw = _mapping(raw, "observation")
    _reject_unknown(
        observation_raw,
        {
            "model",
            "depth_mode",
            "library_size",
            "depth_min",
            "depth_max",
            "concentration_mode",
            "concentration_median",
            "concentration_log_sd",
            "concentration_min",
            "concentration_max",
            "paired_neutral_counts",
        },
        "observation",
    )
    paired = observation_raw.get("paired_neutral_counts", False)
    if type(paired) is not bool:
        raise ValueError("observation.paired_neutral_counts must be true or false.")
    observation_model = _choice(
        observation_raw.get("model", "dirichlet-multinomial"),
        {"multinomial", "dirichlet-multinomial"},
        "observation.model",
    )
    depth_mode = _choice(
        observation_raw.get("depth_mode", "model-donor"),
        {"fixed", "model-donor"},
        "observation.depth_mode",
    )
    library_raw = observation_raw.get("library_size")
    if depth_mode == "model-donor":
        if library_raw is not None:
            raise ValueError(
                "observation.library_size applies only when depth_mode is fixed."
            )
        library_size = None
    else:
        library_size = _integer(
            10000 if library_raw is None else library_raw,
            "observation.library_size",
            minimum=1,
        )

    concentration_fields = (
        "concentration_mode",
        "concentration_median",
        "concentration_log_sd",
        "concentration_min",
        "concentration_max",
    )
    if observation_model == "multinomial":
        active_concentration_fields = [
            field
            for field in concentration_fields
            if observation_raw.get(field) is not None
        ]
        if active_concentration_fields:
            raise ValueError(
                "Multinomial observation does not use: "
                + ", ".join(
                    f"observation.{field}" for field in active_concentration_fields
                )
                + "."
            )
        concentration_mode = None
        concentration_median = None
        concentration_log_sd = None
        concentration_min = None
        concentration_max = None
    else:
        concentration_mode = _choice(
            observation_raw.get("concentration_mode", "lognormal"),
            {"fixed", "lognormal"},
            "observation.concentration_mode",
        )
        concentration_median = _finite(
            observation_raw.get("concentration_median", 1000),
            "observation.concentration_median",
            minimum=0,
        )
        if concentration_mode == "fixed":
            if observation_raw.get("concentration_log_sd") is not None:
                raise ValueError(
                    "Fixed concentration does not use observation.concentration_log_sd."
                )
            concentration_log_sd = None
        else:
            concentration_log_sd = _finite(
                observation_raw.get("concentration_log_sd", 0.45),
                "observation.concentration_log_sd",
                minimum=0,
            )
        concentration_min = _finite(
            observation_raw.get("concentration_min", 250),
            "observation.concentration_min",
            minimum=0,
        )
        concentration_max = _finite(
            observation_raw.get("concentration_max", 4000),
            "observation.concentration_max",
            minimum=0,
        )
    observation = ObservationSettings(
        model=observation_model,
        depth_mode=depth_mode,
        library_size=library_size,
        depth_min=_integer(
            observation_raw.get("depth_min", 1000),
            "observation.depth_min",
            minimum=1,
        ),
        depth_max=_integer(
            observation_raw.get("depth_max", 150000),
            "observation.depth_max",
            minimum=1,
        ),
        concentration_mode=concentration_mode,
        concentration_median=concentration_median,
        concentration_log_sd=concentration_log_sd,
        concentration_min=concentration_min,
        concentration_max=concentration_max,
        paired_neutral_counts=paired,
    )
    if observation.depth_min > observation.depth_max:
        raise ValueError("observation.depth_min cannot exceed depth_max.")
    if observation.model == "dirichlet-multinomial":
        assert observation.concentration_min is not None
        assert observation.concentration_median is not None
        assert observation.concentration_max is not None
        if (
            not 0
            < observation.concentration_min
            <= observation.concentration_median
            <= observation.concentration_max
        ):
            raise ValueError(
                "Observation concentration bounds must contain the positive median."
            )
        if observation.concentration_mode == "lognormal" and (
            observation.concentration_log_sd is None
            or observation.concentration_log_sd <= 0
        ):
            raise ValueError(
                "Lognormal concentration requires positive concentration_log_sd."
            )
    return observation


def _parse_runtime(raw: object) -> RuntimeSettings:
    runtime_raw = _mapping(raw, "runtime")
    _reject_unknown(runtime_raw, {"seed", "batch_size", "device"}, "runtime")
    runtime = RuntimeSettings(
        seed=_integer(runtime_raw.get("seed", 42), "runtime.seed", minimum=0),
        batch_size=_integer(
            runtime_raw.get("batch_size", 64), "runtime.batch_size", minimum=1
        ),
        device=str(runtime_raw.get("device", "auto")),
    )
    return runtime


def parse_scenario(
    raw: Mapping[str, Any],
    *,
    cli_overrides: tuple[str, ...] = (),
) -> ResolvedBenchmarkScenario:
    root = dict(raw)
    _reject_unknown(
        root,
        {
            "schema_version",
            "preset",
            "description",
            "model",
            "conditions",
            "design",
            "biology",
            "batch",
            "support",
            "observation",
            "runtime",
            "cli_overrides",
            "resolved_internal",
            "actual_cell_counts",
            "truth_applicability",
        },
        "top-level",
    )
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")

    model = _parse_model(root.get("model"))
    common, groups = _parse_conditions(root.get("conditions"))
    design = _parse_design(root.get("design"), groups)
    biology = _parse_biology(root.get("biology"), len(groups))
    batch = _parse_batch(root.get("batch"), design.n_batches)
    support = _parse_support(root.get("support"))
    observation = _parse_observation(root.get("observation"))
    runtime = _parse_runtime(root.get("runtime"))
    if "dispersion" in batch.mechanisms and observation.model == "multinomial":
        raise ValueError(
            "The dispersion batch mechanism requires dirichlet-multinomial observation."
        )
    if batch.allow_target_overlap and biology.mode in {"null", "natural"}:
        raise ValueError(
            "batch.allow_target_overlap requires direct biological targets."
        )
    scenario = ResolvedBenchmarkScenario(
        schema_version=SCHEMA_VERSION,
        preset=str(root.get("preset", "custom")),
        model=model,
        common_conditions=common,
        groups=groups,
        design=design,
        biology=biology,
        batch=batch,
        support=support,
        observation=observation,
        runtime=runtime,
        cli_overrides=cli_overrides
        or tuple(str(value) for value in root.get("cli_overrides", [])),
    )
    design_cell_counts(scenario)
    return scenario
