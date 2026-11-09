from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .bank import BatchNeutralBank
from .benchmark_bundle import BenchmarkStreamWriter, write_benchmark_bundle
from .benchmark_config import (
    ResolvedBenchmarkScenario,
    design_cell_count_matrix,
    load_custom_config,
    resolve_preset,
    scenario_truth_applicability,
)
from .benchmark_effects import (
    global_offsets,
    redistribute_target_mass,
    scalar_offsets,
    sparse_offsets,
)
from .generation_core import CompositionStage, decode_composition, sample_counts
from .common import require_bool, staged_output_dir, write_yaml
from .defaults import DEFAULT_MAX_IN_MEMORY_MB
from .simulator import Simulator, _select_donor_pool


@dataclass(frozen=True)
class BenchmarkResult:
    """Generated benchmark tables, truth, configuration, paths, and memory estimate.

    Large streamed runs retain ``feature_truth``, ``config``, ``files``, and
    ``estimated_memory_mb`` while sample-level tables are ``None``. ``loaded``
    reports whether full tables are in memory; ``output_dir`` is ``None`` when
    nothing was written.
    """

    output_dir: Path | None
    counts: pd.DataFrame | None
    relative_abundance: pd.DataFrame | None
    latent_composition: pd.DataFrame | None
    metadata: pd.DataFrame | None
    feature_truth: pd.DataFrame
    sample_truth: pd.DataFrame | None
    batch_neutral_reference: pd.DataFrame | None
    batch_neutral_counts: pd.DataFrame | None
    config: dict[str, object]
    files: dict[str, Path]
    estimated_memory_mb: float

    @property
    def loaded(self) -> bool:
        return self.counts is not None


def _seed(base: int, *parts: int) -> int:
    return int(
        np.random.SeedSequence([int(base), *map(int, parts)]).generate_state(1)[0]
    )


def _support_threshold(scenario: ResolvedBenchmarkScenario) -> float:
    return 0.5 if scenario.support.threshold is None else scenario.support.threshold


def _support_temperature(scenario: ResolvedBenchmarkScenario) -> float:
    return 1.0 if scenario.support.temperature is None else scenario.support.temperature


def _design(
    scenario: ResolvedBenchmarkScenario,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_counts = np.asarray(design_cell_count_matrix(scenario), dtype=int)
    group_indices: list[int] = []
    batch_indices: list[int] = []
    for batch_index, counts in enumerate(cell_counts):
        for group_index, count in enumerate(counts):
            group_indices.extend([group_index] * int(count))
            batch_indices.extend([batch_index] * int(count))
    group_array = np.asarray(group_indices, dtype=int)
    batch_array = np.asarray(batch_indices, dtype=int)
    order = np.random.default_rng(_seed(scenario.runtime.seed, 1)).permutation(
        len(group_array)
    )
    return group_array[order], batch_array[order], cell_counts


def _conditions(
    scenario: ResolvedBenchmarkScenario, group_index: int
) -> dict[str, object]:
    values = dict(scenario.common_conditions)
    values.update(scenario.groups[group_index].conditions)
    return values


def _generation_arguments(conditions: Mapping[str, object]) -> dict[str, object]:
    return {
        "phenotype": str(conditions.get("Phenotype_Tokens", "Healthy")),
        "body_site": str(conditions.get("Body_Site_Coarse", "stool_feces")),
        "conditions": dict(conditions),
    }


def _generate_logits(
    simulator: Simulator | BatchNeutralBank,
    scenario: ResolvedBenchmarkScenario,
    group_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(group_indices)
    n_taxa = len(simulator.taxa)
    abundance = np.empty((n_samples, n_taxa), dtype=np.float64)
    support = np.empty_like(abundance)
    statuses = np.empty(n_samples, dtype=object)
    source_ids = np.empty(n_samples, dtype=object)
    _generate_logits_into(
        simulator,
        scenario,
        group_indices,
        abundance,
        support,
        statuses,
        source_ids,
    )
    return abundance, support, statuses, source_ids


def _generate_logits_into(
    simulator: Simulator | BatchNeutralBank,
    scenario: ResolvedBenchmarkScenario,
    group_indices: np.ndarray,
    abundance: np.ndarray,
    support: np.ndarray,
    statuses: np.ndarray,
    source_ids: np.ndarray,
) -> None:
    if isinstance(simulator, BatchNeutralBank):
        grouped: dict[tuple[tuple[str, str], ...], list[int]] = {}
        for group_index in range(len(scenario.groups)):
            key = tuple(
                sorted(
                    (str(field), str(value))
                    for field, value in _conditions(scenario, group_index).items()
                )
            )
            grouped.setdefault(key, []).append(group_index)
        for selection_index, (key, group_ids) in enumerate(grouped.items()):
            positions = np.concatenate(
                [
                    np.flatnonzero(group_indices == group_index)
                    for group_index in group_ids
                ]
            )
            selected = simulator.select_indices(
                dict(key),
                n_samples=len(positions),
                seed=_seed(scenario.runtime.seed, 10, selection_index),
            )
            for start in range(0, len(positions), scenario.runtime.batch_size):
                chunk = positions[start : start + scenario.runtime.batch_size]
                source_rows = selected[start : start + scenario.runtime.batch_size]
                abundance[chunk] = np.asarray(
                    simulator.abundance_logits[source_rows], dtype=np.float64
                )
                support[chunk] = np.asarray(
                    simulator.support_logits[source_rows], dtype=np.float64
                )
                statuses[chunk] = "bank-supported"
                source_ids[chunk] = (
                    simulator.metadata.iloc[source_rows]["sample_id"]
                    .astype(str)
                    .to_numpy()
                )
        return
    for group_index, _group in enumerate(scenario.groups):
        positions = np.flatnonzero(group_indices == group_index)
        conditions = _conditions(scenario, group_index)
        for start in range(0, len(positions), scenario.runtime.batch_size):
            chunk = positions[start : start + scenario.runtime.batch_size]
            generated = simulator._generate_indexed(
                **_generation_arguments(conditions),
                sample_indices=chunk,
                output_mode="relative",
                guidance_scale=scenario.model.guidance_scale,
                support_temperature=_support_temperature(scenario),
                seed=_seed(scenario.runtime.seed, 10, group_index),
                library_size=None,
                use_body_system=scenario.model.use_body_system,
            )
            abundance[chunk] = generated.abundance_logits
            support[chunk] = generated.support_logits
            statuses[chunk] = (
                generated.metadata["condition_status"].astype(str).to_numpy()
            )
            source_ids[chunk] = None


def _reference_logits(
    simulator: Simulator | BatchNeutralBank,
    scenario: ResolvedBenchmarkScenario,
) -> tuple[np.ndarray, np.ndarray]:
    abundance: list[np.ndarray] = []
    support: list[np.ndarray] = []
    reference_size = min(64, max(16, scenario.design.n_samples // len(scenario.groups)))
    seen_conditions: set[tuple[tuple[str, str], ...]] = set()
    for group_index, _group in enumerate(scenario.groups):
        conditions = _conditions(scenario, group_index)
        key = tuple(
            sorted((str(field), str(value)) for field, value in conditions.items())
        )
        if key in seen_conditions:
            continue
        seen_conditions.add(key)
        if isinstance(simulator, BatchNeutralBank):
            reference_abundance, reference_support, _ = simulator.sample_logits(
                dict(key),
                n_samples=reference_size,
                seed=_seed(scenario.runtime.seed, 20, group_index),
            )
        else:
            generated = simulator._generate_indexed(
                **_generation_arguments(conditions),
                sample_indices=np.arange(reference_size, dtype=np.int64),
                output_mode="relative",
                guidance_scale=scenario.model.guidance_scale,
                support_temperature=_support_temperature(scenario),
                seed=_seed(scenario.runtime.seed, 20, group_index),
                library_size=None,
                use_body_system=scenario.model.use_body_system,
            )
            reference_abundance = generated.abundance_logits
            reference_support = generated.support_logits
        abundance.append(np.asarray(reference_abundance, dtype=np.float64))
        support.append(np.asarray(reference_support, dtype=np.float64))
    return np.vstack(abundance), np.vstack(support)


def _taxon_indices(
    taxa: list[str], selected: tuple[str, ...] | None, name: str
) -> np.ndarray | None:
    if selected is None:
        return None
    lookup = {taxon: index for index, taxon in enumerate(taxa)}
    missing = [taxon for taxon in selected if taxon not in lookup]
    if missing:
        raise ValueError(f"Unknown {name} taxa: {', '.join(missing)}.")
    return np.asarray([lookup[taxon] for taxon in selected], dtype=int)


def _biological_coefficients(
    scenario: ResolvedBenchmarkScenario,
    taxa: list[str],
    reference_stage: CompositionStage,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_taxa = len(taxa)
    abundance = np.zeros(n_taxa, dtype=np.float64)
    support = np.zeros(n_taxa, dtype=np.float64)
    biology = scenario.biology
    if biology.mode in {"null", "natural"}:
        return abundance, support, np.array([], dtype=int)
    selected = _taxon_indices(taxa, biology.target_taxa, "biological target")
    selection_rng = np.random.default_rng(
        np.random.SeedSequence([scenario.runtime.seed, 4103])
    )
    if selected is None:
        prevalence = reference_stage.support_mask.mean(axis=0, dtype=np.float64)
        eligible = np.flatnonzero(
            (prevalence >= biology.prevalence_min)
            & (prevalence <= biology.prevalence_max)
        )
        if len(eligible) < biology.n_targets:
            raise ValueError(
                f"Only {len(eligible)} taxa satisfy the requested prevalence range; "
                f"{biology.n_targets} targets were requested. Broaden the range or specify taxa."
            )
        selected = np.sort(
            selection_rng.choice(eligible, size=biology.n_targets, replace=False)
        )
    coefficients = np.full(
        len(selected), float(biology.effect_strength), dtype=np.float64
    )
    if biology.effect_directions in {"balanced", "random-balanced"}:
        if biology.target_selection == "automatic":
            shuffled = selection_rng.permutation(selected)
            positive = set(shuffled[: len(selected) // 2].tolist())
            coefficients = np.asarray(
                [
                    (
                        float(biology.effect_strength)
                        if index in positive
                        else -float(biology.effect_strength)
                    )
                    for index in selected
                ]
            )
        else:
            coefficients[len(selected) // 2 :] *= -1
            if biology.effect_directions == "random-balanced":
                selection_rng.shuffle(coefficients)
    if biology.mode in {"strict", "abundance", "mixed"}:
        abundance[selected] = coefficients
    if biology.mode in {"prevalence", "mixed"}:
        support[selected] = coefficients
    return abundance, support, selected


def _observation_parameters(
    simulator: Simulator | BatchNeutralBank,
    scenario: ResolvedBenchmarkScenario,
    group_indices: np.ndarray,
    batch_indices: np.ndarray,
    batch_depth_offsets: np.ndarray,
    dispersion_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    observation = scenario.observation
    rng = np.random.default_rng(scenario.runtime.seed + 104_729)
    if observation.depth_mode == "fixed":
        assert observation.library_size is not None
        depths = np.rint(
            observation.library_size * np.exp(batch_depth_offsets[batch_indices])
        ).astype(np.int64)
        if np.any(depths < observation.depth_min) or np.any(
            depths > observation.depth_max
        ):
            raise ValueError(
                "Fixed library size and batch depth effects cannot satisfy depth_min/depth_max."
            )
        donor_match_levels = None
    else:
        if isinstance(simulator, BatchNeutralBank):
            raise ValueError(
                "model-donor depth mode requires a model bundle; use fixed depth with a bank."
            )
        donor_path = simulator.bundle.root / "donors.csv"
        if not donor_path.is_file():
            raise ValueError(
                "model-donor depth mode requires donors.csv in the selected model bundle."
            )
        donors = pd.read_csv(donor_path)
        if "raw_total_reads" not in donors:
            raise ValueError("donors.csv must contain raw_total_reads.")
        pools: dict[int, np.ndarray] = {}
        group_match_levels: dict[int, str] = {}
        for group_index in range(len(scenario.groups)):
            conditions = _conditions(scenario, group_index)
            prepared = simulator._prepare_conditions(
                str(conditions.get("Phenotype_Tokens", "Healthy")),
                str(conditions.get("Body_Site_Coarse", "stool_feces")),
                conditions,
                scenario.model.use_body_system,
            )
            donor_rows, match_level = _select_donor_pool(
                donors, prepared, simulator.bundle.vocabularies
            )
            pool = pd.to_numeric(
                donor_rows["raw_total_reads"], errors="coerce"
            ).dropna()
            pool = pool[(pool > 0) & np.isfinite(pool)]
            if pool.empty:
                raise ValueError(
                    f"No valid model donor depths are available after {match_level} matching."
                )
            pools[group_index] = pool.to_numpy(dtype=np.float64)
            group_match_levels[group_index] = match_level
        donor_match_levels = np.asarray(
            [group_match_levels[int(group)] for group in group_indices], dtype=object
        )
        depths = np.empty(len(batch_indices), dtype=np.int64)
        for batch_index in range(scenario.design.n_batches):
            multiplier = float(np.exp(batch_depth_offsets[batch_index]))
            for row in np.flatnonzero(batch_indices == batch_index):
                values = pools[int(group_indices[row])]
                for _ in range(10_000):
                    shifted = max(
                        1, int(np.rint(float(rng.choice(values)) * multiplier))
                    )
                    if observation.depth_min <= shifted <= observation.depth_max:
                        depths[row] = shifted
                        break
                else:
                    raise ValueError(
                        f"No valid donor depth could be sampled for batch_{batch_index + 1}; "
                        "widen the depth bounds or reduce the batch depth strength."
                    )
    if observation.model == "multinomial":
        concentrations = np.full(len(batch_indices), np.nan, dtype=np.float64)
    elif observation.concentration_mode == "fixed":
        assert observation.concentration_median is not None
        assert observation.concentration_min is not None
        assert observation.concentration_max is not None
        concentrations = observation.concentration_median * np.exp(
            dispersion_offsets[batch_indices]
        )
        if np.any(concentrations < observation.concentration_min) or np.any(
            concentrations > observation.concentration_max
        ):
            raise ValueError(
                "Fixed concentration and batch dispersion effects cannot satisfy concentration bounds."
            )
    else:
        assert observation.concentration_median is not None
        assert observation.concentration_log_sd is not None
        assert observation.concentration_min is not None
        assert observation.concentration_max is not None
        concentrations = np.empty(len(batch_indices), dtype=np.float64)
        for batch_index in range(scenario.design.n_batches):
            multiplier = float(np.exp(dispersion_offsets[batch_index]))
            for row in np.flatnonzero(batch_indices == batch_index):
                for _ in range(10_000):
                    shifted = (
                        float(
                            rng.lognormal(
                                mean=np.log(observation.concentration_median),
                                sigma=observation.concentration_log_sd,
                            )
                        )
                        * multiplier
                    )
                    if (
                        observation.concentration_min
                        <= shifted
                        <= observation.concentration_max
                    ):
                        concentrations[row] = shifted
                        break
                else:
                    raise ValueError(
                        f"No valid concentration could be sampled for batch_{batch_index + 1}; "
                        "widen the bounds or reduce log_sd/batch dispersion strength."
                    )
    return depths, concentrations, donor_match_levels


def _abundance_frame(
    sample_ids: list[str], taxa: list[str], values: np.ndarray
) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=taxa)
    frame.insert(0, "sample_id", sample_ids)
    return frame


def _maximum(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _batch_effects(
    scenario: ResolvedBenchmarkScenario,
    taxa: list[str],
    batch_indices: np.ndarray,
    reference_abundance: np.ndarray,
    reference_support: np.ndarray,
    biological_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_taxa = len(taxa)
    batch_abundance = np.zeros((scenario.design.n_batches, n_taxa), dtype=np.float64)
    batch_support = np.zeros_like(batch_abundance)
    batch_abundance_targets = np.array([], dtype=int)
    batch_support_targets = np.array([], dtype=int)
    mean_reference_support = (
        1.0 / (1.0 + np.exp(-np.clip(reference_support, -40, 40)))
    ).mean(axis=0)
    eligible = (mean_reference_support >= 0.05) & (mean_reference_support <= 0.95)
    if not scenario.batch.allow_target_overlap:
        eligible[biological_targets] = False
    batch_rng = np.random.default_rng(scenario.runtime.seed + 73_013)
    mechanisms = set(scenario.batch.mechanisms)
    if "global" in mechanisms:
        batch_abundance = global_offsets(
            reference_abundance,
            batch_indices,
            scenario.design.n_batches,
            scenario.batch.global_strength,
            batch_rng,
        )
        batch_abundance_targets = np.arange(n_taxa)
    elif "targeted_abundance" in mechanisms:
        specified_abundance = _taxon_indices(
            taxa, scenario.batch.target_taxa, "batch abundance target"
        )
        batch_abundance, batch_abundance_targets = sparse_offsets(
            eligible,
            batch_indices,
            scenario.design.n_batches,
            scenario.batch.target_fraction or 1.0,
            scenario.batch.target_strength,
            batch_rng,
            selected=specified_abundance,
            center_features=True,
        )
    if "prevalence" in mechanisms:
        specified_support = _taxon_indices(
            taxa, scenario.batch.prevalence_taxa, "batch prevalence target"
        )
        batch_support, batch_support_targets = sparse_offsets(
            eligible,
            batch_indices,
            scenario.design.n_batches,
            scenario.batch.prevalence_fraction or 1.0,
            scenario.batch.prevalence_strength,
            batch_rng,
            selected=specified_support,
            center_features=False,
        )
    depth_offsets = np.zeros(scenario.design.n_batches, dtype=np.float64)
    dispersion_offsets = np.zeros_like(depth_offsets)
    if "depth" in mechanisms:
        depth_offsets = scalar_offsets(
            batch_indices,
            scenario.design.n_batches,
            scenario.batch.depth_strength,
            batch_rng,
        )
    if "dispersion" in mechanisms:
        dispersion_offsets = scalar_offsets(
            batch_indices,
            scenario.design.n_batches,
            scenario.batch.dispersion_strength,
            batch_rng,
        )
    return (
        batch_abundance,
        batch_support,
        batch_abundance_targets,
        batch_support_targets,
        depth_offsets,
        dispersion_offsets,
    )


def _decode_effect_stages(
    scenario: ResolvedBenchmarkScenario,
    abundance_logits: np.ndarray,
    support_logits: np.ndarray,
    group_scores: np.ndarray,
    biological_abundance: np.ndarray,
    biological_support: np.ndarray,
    batch_indices: np.ndarray,
    batch_abundance: np.ndarray,
    batch_support: np.ndarray,
    uniforms: np.ndarray | None = None,
) -> tuple[CompositionStage, CompositionStage, dict[str, object] | None]:
    biological_abundance_offsets = (
        np.zeros_like(abundance_logits)
        if scenario.biology.mode == "strict"
        else np.outer(group_scores, biological_abundance)
    )
    biological_support_offsets = np.outer(group_scores, biological_support)
    if uniforms is None and scenario.support.mode != "deterministic":
        uniforms = np.random.default_rng(scenario.runtime.seed).random(
            support_logits.shape
        )
    decode_options = {
        "mode": scenario.support.mode,
        "threshold": _support_threshold(scenario),
        "temperature": _support_temperature(scenario),
        "uniforms": None if scenario.support.mode == "deterministic" else uniforms,
    }
    neutral_before_strict = decode_composition(
        abundance_logits + biological_abundance_offsets,
        support_logits + biological_support_offsets,
        **decode_options,
    )
    final_before_strict = decode_composition(
        abundance_logits
        + biological_abundance_offsets
        + batch_abundance[batch_indices],
        support_logits + biological_support_offsets + batch_support[batch_indices],
        **decode_options,
    )
    if scenario.biology.mode != "strict":
        return neutral_before_strict, final_before_strict, None

    neutral = redistribute_target_mass(
        neutral_before_strict, group_scores, biological_abundance
    )
    final = redistribute_target_mass(
        final_before_strict, group_scores, biological_abundance
    )
    target = biological_abundance != 0
    non_target_changes = np.concatenate(
        [
            (neutral.composition - neutral_before_strict.composition)[
                :, ~target
            ].ravel(),
            (final.composition - final_before_strict.composition)[:, ~target].ravel(),
        ]
    )
    target_mass_changes = np.concatenate(
        [
            neutral.composition[:, target].sum(axis=1)
            - neutral_before_strict.composition[:, target].sum(axis=1),
            final.composition[:, target].sum(axis=1)
            - final_before_strict.composition[:, target].sum(axis=1),
        ]
    )
    support_changes = int(
        np.count_nonzero(neutral.support_mask != neutral_before_strict.support_mask)
        + np.count_nonzero(final.support_mask != final_before_strict.support_mask)
    )
    audit = {
        "maximum_non_target_change": _maximum(non_target_changes),
        "maximum_target_mass_change": _maximum(target_mass_changes),
        "support_change_count": support_changes,
    }
    return neutral, final, audit


def _sample_observations(
    simulator: Simulator | BatchNeutralBank,
    scenario: ResolvedBenchmarkScenario,
    group_indices: np.ndarray,
    batch_indices: np.ndarray,
    depth_offsets: np.ndarray,
    dispersion_offsets: np.ndarray,
    support_shape: tuple[int, ...],
    neutral_stage: CompositionStage,
    final_stage: CompositionStage,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray,
    np.ndarray | None,
]:
    depths, concentrations, donor_match_levels = _observation_parameters(
        simulator,
        scenario,
        group_indices,
        batch_indices,
        depth_offsets,
        dispersion_offsets,
    )
    strict_seed = int(
        np.random.SeedSequence([scenario.runtime.seed, 4201]).generate_state(1)[0]
    )
    count_rng = np.random.default_rng(
        strict_seed if scenario.biology.mode == "strict" else scenario.runtime.seed
    )
    if scenario.biology.mode != "strict" and scenario.support.mode != "deterministic":
        count_rng.random(support_shape)
    counts = sample_counts(
        final_stage.composition,
        depths,
        model=scenario.observation.model,
        concentrations=concentrations,
        rng=count_rng,
    )
    neutral_counts = None
    if scenario.observation.paired_neutral_counts:
        neutral_rng = np.random.default_rng(
            strict_seed if scenario.biology.mode == "strict" else scenario.runtime.seed
        )
        if (
            scenario.biology.mode != "strict"
            and scenario.support.mode != "deterministic"
        ):
            neutral_rng.random(support_shape)
        neutral_counts = sample_counts(
            neutral_stage.composition,
            depths,
            model=scenario.observation.model,
            concentrations=concentrations,
            rng=neutral_rng,
        )
    return depths, concentrations, donor_match_levels, counts, neutral_counts


def estimate_benchmark_memory_mb(n_samples: int, n_taxa: int) -> float:
    """Estimate peak memory for the in-memory benchmark path."""

    matrix_bytes = n_samples * n_taxa * 192
    row_bytes = n_samples * 4096
    return (matrix_bytes + row_bytes) / 2**20


def _effect_setup(
    simulator: Simulator | BatchNeutralBank,
    scenario: ResolvedBenchmarkScenario,
    taxa: list[str],
    group_indices: np.ndarray,
    batch_indices: np.ndarray,
):
    reference_abundance, reference_support = _reference_logits(simulator, scenario)
    reference_uniforms = np.random.default_rng(
        int(np.random.SeedSequence([scenario.runtime.seed, 4101]).generate_state(1)[0])
    ).random(reference_support.shape)
    reference_stage = decode_composition(
        reference_abundance,
        reference_support,
        mode=scenario.support.mode,
        threshold=_support_threshold(scenario),
        temperature=_support_temperature(scenario),
        uniforms=(
            None if scenario.support.mode == "deterministic" else reference_uniforms
        ),
    )
    biological_abundance, biological_support, biological_targets = (
        _biological_coefficients(scenario, taxa, reference_stage)
    )
    group_scores = (group_indices == 1).astype(np.float64)
    (
        batch_abundance,
        batch_support,
        batch_abundance_targets,
        batch_support_targets,
        depth_offsets,
        dispersion_offsets,
    ) = _batch_effects(
        scenario,
        taxa,
        batch_indices,
        reference_abundance,
        reference_support,
        biological_targets,
    )
    return (
        group_scores,
        biological_abundance,
        biological_support,
        biological_targets,
        batch_abundance,
        batch_support,
        batch_abundance_targets,
        batch_support_targets,
        depth_offsets,
        dispersion_offsets,
    )


def _feature_truth(
    taxa: list[str],
    biological_abundance: np.ndarray,
    biological_support: np.ndarray,
    batch_abundance: np.ndarray,
    batch_support: np.ndarray,
    batch_abundance_targets: np.ndarray,
    batch_support_targets: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "taxon": taxa,
            "is_direct_biological_target": (biological_abundance != 0)
            | (biological_support != 0),
            "biological_abundance_coefficient": biological_abundance,
            "biological_support_coefficient": biological_support,
            "is_batch_abundance_target": np.isin(
                np.arange(len(taxa)), batch_abundance_targets
            ),
            "is_batch_support_target": np.isin(
                np.arange(len(taxa)), batch_support_targets
            ),
            "maximum_absolute_batch_abundance_offset": np.max(
                np.abs(batch_abundance), axis=0
            ),
            "maximum_absolute_batch_support_offset": np.max(
                np.abs(batch_support), axis=0
            ),
        }
    )


def _benchmark_config(
    scenario: ResolvedBenchmarkScenario,
    simulator: Simulator | BatchNeutralBank,
    taxa: list[str],
    cell_counts: np.ndarray,
    biological_targets: np.ndarray,
    batch_abundance_targets: np.ndarray,
    batch_support_targets: np.ndarray,
    strict_audit: dict[str, object] | None,
    max_in_memory_mb: float,
    load: bool,
    loaded: bool,
) -> dict[str, object]:
    config = scenario.as_dict()
    if isinstance(simulator, BatchNeutralBank):
        config.pop("model", None)
        config["generation_source"] = "bank"
        config["bank"] = {
            "path": str(simulator.root),
            "type": str(simulator.manifest["bank_type"]),
            "schema_version": int(simulator.manifest.get("schema_version", 1)),
        }
    else:
        config["generation_source"] = "model"
        config["model"] = {
            "path": str(simulator.bundle.root),
            "type": simulator.bundle.model_type,
            "condition_fields": list(simulator.diffusion_config.primary_fields),
            "supports_body_system": simulator.bundle.supports_body_system,
            "guidance_scale": scenario.model.guidance_scale,
        }
    config["actual_cell_counts"] = {
        f"batch_{batch_index + 1}": {
            scenario.groups[group_index].label: int(
                cell_counts[batch_index, group_index]
            )
            for group_index in range(len(scenario.groups))
        }
        for batch_index in range(scenario.design.n_batches)
    }
    config["truth_applicability"] = scenario_truth_applicability(scenario)
    config["resolved_internal"] = {
        "biological_target_taxa": [taxa[index] for index in biological_targets],
        "batch_abundance_target_taxa": [
            taxa[index] for index in batch_abundance_targets
        ],
        "batch_support_target_taxa": [taxa[index] for index in batch_support_targets],
        "strict_target_audit": strict_audit,
    }
    config["runtime"]["max_in_memory_mb"] = max_in_memory_mb
    config["runtime"]["load"] = load
    config["runtime"]["loaded"] = loaded
    return config


def _sample_tables(
    scenario: ResolvedBenchmarkScenario,
    taxa: list[str],
    indices: np.ndarray,
    group_indices: np.ndarray,
    batch_indices: np.ndarray,
    group_scores: np.ndarray,
    condition_status: np.ndarray,
    source_bank_sample_ids: np.ndarray,
    depths: np.ndarray,
    concentrations: np.ndarray,
    donor_match_levels: np.ndarray | None,
    counts: np.ndarray,
    neutral_counts: np.ndarray | None,
    neutral_stage: CompositionStage,
    final_stage: CompositionStage,
    depth_offsets: np.ndarray,
    dispersion_offsets: np.ndarray,
) -> dict[str, pd.DataFrame]:
    sample_ids = [
        f"mbbench_benchmark_{scenario.runtime.seed}_{index + 1:06d}"
        for index in indices
    ]
    group_labels = np.asarray([group.label for group in scenario.groups], dtype=object)[
        group_indices
    ]
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group": group_labels,
            "batch": [f"batch_{index + 1}" for index in batch_indices],
        }
    )
    condition_rows = [
        _conditions(scenario, int(group_index)) for group_index in group_indices
    ]
    condition_fields = sorted({field for row in condition_rows for field in row})
    for field in condition_fields:
        metadata[field] = [
            (
                "|".join(str(item) for item in value)
                if isinstance(value, (list, tuple, set))
                else value
            )
            for row in condition_rows
            for value in [row.get(field)]
        ]
    metadata["condition_status"] = condition_status
    if donor_match_levels is not None:
        metadata["donor_match_level"] = donor_match_levels
    if any(value is not None for value in source_bank_sample_ids):
        metadata["source_bank_sample_id"] = source_bank_sample_ids
    metadata["raw_total_reads"] = depths
    sample_truth = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group_index": group_indices,
            "batch_index": batch_indices,
            "biological_score": group_scores,
            "batch_depth_log_multiplier": depth_offsets[batch_indices],
            "batch_dispersion_log_multiplier": dispersion_offsets[batch_indices],
            "dirichlet_multinomial_concentration": concentrations,
        }
    )
    relative = counts / counts.sum(axis=1, keepdims=True)
    tables = {
        "counts": _abundance_frame(sample_ids, taxa, counts),
        "relative_abundance": _abundance_frame(sample_ids, taxa, relative),
        "latent_composition": _abundance_frame(
            sample_ids, taxa, final_stage.composition
        ),
        "metadata": metadata,
        "sample_truth": sample_truth,
        "batch_neutral_reference": _abundance_frame(
            sample_ids, taxa, neutral_stage.composition
        ),
    }
    if neutral_counts is not None:
        tables["batch_neutral_counts"] = _abundance_frame(
            sample_ids, taxa, neutral_counts
        )
    return tables


def _output_files(output: Path, paired_neutral_counts: bool) -> dict[str, Path]:
    files = {
        "counts": output / "counts.csv.gz",
        "relative_abundance": output / "relative_abundance.csv.gz",
        "latent_composition": output / "latent_composition.csv.gz",
        "metadata": output / "metadata.csv.gz",
        "feature_truth": output / "feature_truth.csv.gz",
        "sample_truth": output / "sample_truth.csv.gz",
        "batch_neutral_reference": output / "batch_neutral_biological_reference.csv.gz",
        "config": output / "benchmark.yaml",
    }
    if paired_neutral_counts:
        files["batch_neutral_counts"] = output / "batch_neutral_counts.csv.gz"
    return files


def _generate_benchmark_in_memory(
    simulator: Simulator | BatchNeutralBank,
    scenario: ResolvedBenchmarkScenario,
    *,
    output_dir: str | Path | None,
    overwrite: bool,
    max_in_memory_mb: float,
    estimated_memory_mb: float,
    load: bool,
) -> BenchmarkResult:
    taxa = simulator.taxa
    group_indices, batch_indices, cell_counts = _design(scenario)
    (
        abundance_logits,
        support_logits,
        condition_status,
        source_bank_sample_ids,
    ) = _generate_logits(simulator, scenario, group_indices)
    (
        group_scores,
        biological_abundance,
        biological_support,
        biological_targets,
        batch_abundance,
        batch_support,
        batch_abundance_targets,
        batch_support_targets,
        depth_offsets,
        dispersion_offsets,
    ) = _effect_setup(simulator, scenario, taxa, group_indices, batch_indices)
    neutral_stage, final_stage, strict_audit = _decode_effect_stages(
        scenario,
        abundance_logits,
        support_logits,
        group_scores,
        biological_abundance,
        biological_support,
        batch_indices,
        batch_abundance,
        batch_support,
    )
    depths, concentrations, donor_match_levels, counts, neutral_counts = (
        _sample_observations(
            simulator,
            scenario,
            group_indices,
            batch_indices,
            depth_offsets,
            dispersion_offsets,
            support_logits.shape,
            neutral_stage,
            final_stage,
        )
    )
    tables = _sample_tables(
        scenario,
        taxa,
        np.arange(len(counts)),
        group_indices,
        batch_indices,
        group_scores,
        condition_status,
        source_bank_sample_ids,
        depths,
        concentrations,
        donor_match_levels,
        counts,
        neutral_counts,
        neutral_stage,
        final_stage,
        depth_offsets,
        dispersion_offsets,
    )
    feature_truth = _feature_truth(
        taxa,
        biological_abundance,
        biological_support,
        batch_abundance,
        batch_support,
        batch_abundance_targets,
        batch_support_targets,
    )
    tables["feature_truth"] = feature_truth
    config = _benchmark_config(
        scenario,
        simulator,
        taxa,
        cell_counts,
        biological_targets,
        batch_abundance_targets,
        batch_support_targets,
        strict_audit,
        max_in_memory_mb,
        load,
        True,
    )
    output = None
    files: dict[str, Path] = {}
    if output_dir is not None:
        output = write_benchmark_bundle(
            output_dir, tables=tables, config=config, overwrite=overwrite
        )
        files = _output_files(output, scenario.observation.paired_neutral_counts)
    return BenchmarkResult(
        output_dir=output,
        counts=tables["counts"],
        relative_abundance=tables["relative_abundance"],
        latent_composition=tables["latent_composition"],
        metadata=tables["metadata"],
        feature_truth=feature_truth,
        sample_truth=tables["sample_truth"],
        batch_neutral_reference=tables["batch_neutral_reference"],
        batch_neutral_counts=tables.get("batch_neutral_counts"),
        config=config,
        files=files,
        estimated_memory_mb=estimated_memory_mb,
    )


def _advance_support_draws(
    rng: np.random.Generator, n_samples: int, n_taxa: int, batch_size: int
) -> None:
    for start in range(0, n_samples, batch_size):
        rng.random((min(batch_size, n_samples - start), n_taxa))


def _close_memmap(array: np.memmap | None) -> None:
    if array is None:
        return
    array.flush()
    array._mmap.close()


def _generate_benchmark_streamed(
    simulator: Simulator | BatchNeutralBank,
    scenario: ResolvedBenchmarkScenario,
    *,
    output_dir: str | Path,
    overwrite: bool,
    max_in_memory_mb: float,
    estimated_memory_mb: float,
    load: bool,
) -> BenchmarkResult:
    taxa = simulator.taxa
    n_taxa = len(taxa)
    group_indices, batch_indices, cell_counts = _design(scenario)
    n_samples = len(group_indices)
    output = Path(output_dir).expanduser().resolve()
    abundance_logits = None
    support_logits = None
    feature_truth = None
    config = None
    with staged_output_dir(output, overwrite=overwrite) as staging:
        abundance_path = staging / ".abundance_logits.dat"
        support_path = staging / ".support_logits.dat"
        try:
            abundance_logits = np.memmap(
                abundance_path, mode="w+", dtype=np.float32, shape=(n_samples, n_taxa)
            )
            support_logits = np.memmap(
                support_path, mode="w+", dtype=np.float32, shape=(n_samples, n_taxa)
            )
            condition_status = np.empty(n_samples, dtype=object)
            source_bank_sample_ids = np.empty(n_samples, dtype=object)
            _generate_logits_into(
                simulator,
                scenario,
                group_indices,
                abundance_logits,
                support_logits,
                condition_status,
                source_bank_sample_ids,
            )
            (
                group_scores,
                biological_abundance,
                biological_support,
                biological_targets,
                batch_abundance,
                batch_support,
                batch_abundance_targets,
                batch_support_targets,
                depth_offsets,
                dispersion_offsets,
            ) = _effect_setup(simulator, scenario, taxa, group_indices, batch_indices)
            depths, concentrations, donor_match_levels = _observation_parameters(
                simulator,
                scenario,
                group_indices,
                batch_indices,
                depth_offsets,
                dispersion_offsets,
            )
            strict_seed = int(
                np.random.SeedSequence([scenario.runtime.seed, 4201]).generate_state(1)[
                    0
                ]
            )
            count_seed = (
                strict_seed
                if scenario.biology.mode == "strict"
                else scenario.runtime.seed
            )
            count_rng = np.random.default_rng(count_seed)
            neutral_rng = (
                np.random.default_rng(count_seed)
                if scenario.observation.paired_neutral_counts
                else None
            )
            if (
                scenario.biology.mode != "strict"
                and scenario.support.mode != "deterministic"
            ):
                _advance_support_draws(
                    count_rng, n_samples, n_taxa, scenario.runtime.batch_size
                )
                if neutral_rng is not None:
                    _advance_support_draws(
                        neutral_rng, n_samples, n_taxa, scenario.runtime.batch_size
                    )
            support_rng = np.random.default_rng(scenario.runtime.seed)
            strict_audit = (
                {
                    "maximum_non_target_change": 0.0,
                    "maximum_target_mass_change": 0.0,
                    "support_change_count": 0,
                }
                if scenario.biology.mode == "strict"
                else None
            )
            with BenchmarkStreamWriter(
                staging,
                paired_neutral_counts=scenario.observation.paired_neutral_counts,
            ) as writer:
                for start in range(0, n_samples, scenario.runtime.batch_size):
                    stop = min(start + scenario.runtime.batch_size, n_samples)
                    rows = np.arange(start, stop)
                    uniforms = (
                        None
                        if scenario.support.mode == "deterministic"
                        else support_rng.random((stop - start, n_taxa))
                    )
                    neutral_stage, final_stage, chunk_audit = _decode_effect_stages(
                        scenario,
                        np.asarray(abundance_logits[start:stop]),
                        np.asarray(support_logits[start:stop]),
                        group_scores[start:stop],
                        biological_abundance,
                        biological_support,
                        batch_indices[start:stop],
                        batch_abundance,
                        batch_support,
                        uniforms=uniforms,
                    )
                    if strict_audit is not None and chunk_audit is not None:
                        strict_audit["maximum_non_target_change"] = max(
                            strict_audit["maximum_non_target_change"],
                            chunk_audit["maximum_non_target_change"],
                        )
                        strict_audit["maximum_target_mass_change"] = max(
                            strict_audit["maximum_target_mass_change"],
                            chunk_audit["maximum_target_mass_change"],
                        )
                        strict_audit["support_change_count"] += chunk_audit[
                            "support_change_count"
                        ]
                    counts = sample_counts(
                        final_stage.composition,
                        depths[start:stop],
                        model=scenario.observation.model,
                        concentrations=concentrations[start:stop],
                        rng=count_rng,
                    )
                    neutral_counts = (
                        None
                        if neutral_rng is None
                        else sample_counts(
                            neutral_stage.composition,
                            depths[start:stop],
                            model=scenario.observation.model,
                            concentrations=concentrations[start:stop],
                            rng=neutral_rng,
                        )
                    )
                    writer.write(
                        _sample_tables(
                            scenario,
                            taxa,
                            rows,
                            group_indices[start:stop],
                            batch_indices[start:stop],
                            group_scores[start:stop],
                            condition_status[start:stop],
                            source_bank_sample_ids[start:stop],
                            depths[start:stop],
                            concentrations[start:stop],
                            (
                                None
                                if donor_match_levels is None
                                else donor_match_levels[start:stop]
                            ),
                            counts,
                            neutral_counts,
                            neutral_stage,
                            final_stage,
                            depth_offsets,
                            dispersion_offsets,
                        )
                    )
                if writer.rows != n_samples:
                    raise ValueError(
                        f"Streamed benchmark wrote {writer.rows} rows; expected {n_samples}."
                    )
            feature_truth = _feature_truth(
                taxa,
                biological_abundance,
                biological_support,
                batch_abundance,
                batch_support,
                batch_abundance_targets,
                batch_support_targets,
            )
            config = _benchmark_config(
                scenario,
                simulator,
                taxa,
                cell_counts,
                biological_targets,
                batch_abundance_targets,
                batch_support_targets,
                strict_audit,
                max_in_memory_mb,
                load,
                False,
            )
            feature_truth.to_csv(staging / "feature_truth.csv.gz", index=False)
            write_yaml(staging / "benchmark.yaml", config)
        finally:
            _close_memmap(abundance_logits)
            _close_memmap(support_logits)
            if abundance_path.is_file():
                abundance_path.unlink()
            if support_path.is_file():
                support_path.unlink()
    return BenchmarkResult(
        output_dir=output,
        counts=None,
        relative_abundance=None,
        latent_composition=None,
        metadata=None,
        feature_truth=feature_truth,
        sample_truth=None,
        batch_neutral_reference=None,
        batch_neutral_counts=None,
        config=config,
        files=_output_files(output, scenario.observation.paired_neutral_counts),
        estimated_memory_mb=estimated_memory_mb,
    )


def generate_benchmark(
    simulator: Simulator | BatchNeutralBank,
    scenario: ResolvedBenchmarkScenario,
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    max_in_memory_mb: float = DEFAULT_MAX_IN_MEMORY_MB,
    load: bool = True,
) -> BenchmarkResult:
    """Generate samples and truth tables for a resolved benchmark scenario."""

    overwrite = require_bool(overwrite, "overwrite")
    if isinstance(simulator, Simulator):
        scenario = replace(
            scenario,
            model=replace(
                scenario.model,
                type=simulator.bundle.model_type,
                directory=str(simulator.bundle.root),
                use_body_system=simulator.bundle.supports_body_system,
            ),
        )
    else:
        model = scenario.model
        if (
            model.type != "ae"
            or model.directory is not None
            or model.guidance_scale != 1.25
            or model.use_body_system is not None
        ):
            raise ValueError(
                "Model selection, guidance_scale, and use_body_system do not apply "
                "when generating a benchmark from a bank."
            )
    if not np.isfinite(max_in_memory_mb) or max_in_memory_mb <= 0:
        raise ValueError("max_in_memory_mb must be finite and positive.")
    if type(load) is not bool:
        raise ValueError("load must be true or false.")
    if not load and output_dir is None:
        raise ValueError("load=False requires output_dir for streamed generation.")
    estimated = estimate_benchmark_memory_mb(
        scenario.design.n_samples, len(simulator.taxa)
    )
    if not load or estimated > max_in_memory_mb:
        if output_dir is None:
            raise ValueError(
                f"The requested benchmark is estimated to require {estimated:.1f} MiB, "
                f"which exceeds max_in_memory_mb={max_in_memory_mb:g}. Provide output_dir "
                "to use streamed generation, or increase max_in_memory_mb if sufficient "
                "memory is available."
            )
        return _generate_benchmark_streamed(
            simulator,
            scenario,
            output_dir=output_dir,
            overwrite=overwrite,
            max_in_memory_mb=max_in_memory_mb,
            estimated_memory_mb=estimated,
            load=load,
        )
    return _generate_benchmark_in_memory(
        simulator,
        scenario,
        output_dir=output_dir,
        overwrite=overwrite,
        max_in_memory_mb=max_in_memory_mb,
        estimated_memory_mb=estimated,
        load=load,
    )


def benchmark(
    model: Simulator | None = None,
    *,
    bank: BatchNeutralBank | None = None,
    preset: str | None = None,
    config: str | Path | Mapping[str, object] | None = None,
    n_samples: int | None = None,
    n_batches: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    overrides: Mapping[str, object] | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    max_in_memory_mb: float = DEFAULT_MAX_IN_MEMORY_MB,
    load: bool = True,
) -> BenchmarkResult:
    """Generate a benchmark from one preset or one complete configuration.

    Provide exactly one of ``model`` and ``bank``, and exactly one of ``preset``
    and ``config``. ``config`` may be a YAML path or nested mapping and cannot be
    combined with overrides. Preset settings may be changed with direct arguments
    or ``overrides``. Large requests require ``output_dir`` and are streamed.
    """

    if (model is None) == (bank is None):
        raise ValueError("Provide exactly one of model or bank.")
    if (preset is None) == (config is None):
        raise ValueError("Provide exactly one of preset or config.")
    direct = {
        "design.n_samples": n_samples,
        "design.n_batches": n_batches,
        "runtime.seed": seed,
        "runtime.batch_size": batch_size,
    }
    supplied_direct = {
        path: value for path, value in direct.items() if value is not None
    }
    if config is not None:
        if overrides or supplied_direct:
            raise ValueError(
                "A complete config cannot be combined with preset overrides."
            )
        scenario = load_custom_config(config)
    else:
        combined = dict(overrides or {})
        repeated = sorted(set(combined) & set(supplied_direct))
        if repeated:
            raise ValueError(
                "Direct benchmark arguments repeat paths in overrides: "
                + ", ".join(repeated)
                + "."
            )
        combined.update(supplied_direct)
        scenario = resolve_preset(str(preset), combined)
    if model is not None:
        scenario = replace(
            scenario,
            model=replace(
                scenario.model,
                type=model.bundle.model_type,
                directory=str(model.bundle.root),
                use_body_system=model.bundle.supports_body_system,
            ),
        )
    source = model if model is not None else bank
    assert source is not None
    return generate_benchmark(
        source,
        scenario,
        output_dir=output_dir,
        overwrite=overwrite,
        max_in_memory_mb=max_in_memory_mb,
        load=load,
    )
