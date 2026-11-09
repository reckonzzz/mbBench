from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TextIO

import numpy as np
import pandas as pd

from .common import require_bool, staged_output_dir, write_yaml
from .defaults import DEFAULT_MAX_IN_MEMORY_MB
from .bank import BatchNeutralBank
from .simulator import Simulator


@dataclass(frozen=True)
class GenerationOutput:
    """Generated tables, resolved configuration, output paths, and memory estimate.

    ``abundance``, ``counts``, and ``metadata`` are ``None`` when a large request
    is streamed to ``files`` instead of retained in memory. ``config`` records
    resolved conditions and runtime settings; ``estimated_memory_mb`` records the
    preflight estimate. ``loaded`` reports whether tables are present.
    """

    output_dir: Path | None
    abundance: pd.DataFrame | None
    counts: pd.DataFrame | None
    metadata: pd.DataFrame | None
    config: dict[str, object]
    files: dict[str, Path]
    estimated_memory_mb: float

    @property
    def loaded(self) -> bool:
        return self.abundance is not None


def estimate_generation_memory_mb(
    n_samples: int, n_taxa: int, *, output_mode: str, batch_size: int
) -> float:
    """Estimate peak memory for collected generation output."""

    output_bytes = n_samples * n_taxa * (16 if output_mode == "observed" else 8)
    frame_overhead = output_bytes
    working_bytes = min(n_samples, batch_size) * n_taxa * 80
    metadata_bytes = n_samples * 2048
    return (output_bytes + frame_overhead + working_bytes + metadata_bytes) / 2**20


def _memory_error(estimated: float, maximum: float) -> ValueError:
    return ValueError(
        f"The requested generation is estimated to require {estimated:.1f} MiB, "
        f"which exceeds max_in_memory_mb={maximum:g}. Provide output_dir to use "
        "streamed generation, or increase max_in_memory_mb if sufficient memory is available."
    )


def _append_frame(handle: TextIO, frame: pd.DataFrame, first: bool) -> None:
    frame.to_csv(handle, index=False, header=first)


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    return bool(pd.isna(value))


def _sample_count(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(
            "Every condition-table row requires a positive integer n_samples."
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Every condition-table row requires a positive integer n_samples."
        ) from error
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError(
            "Every condition-table row requires a positive integer n_samples."
        )
    return int(numeric)


def _condition_seed(seed: int, conditions: Mapping[str, str]) -> int:
    entropy = [int(seed)]
    for field, value in conditions.items():
        encoded = f"{field}\x1f{value}\x1e".encode("utf-8")
        entropy.extend(encoded)
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def _normalize_condition_table(
    simulator: Simulator,
    condition_table: pd.DataFrame | Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(condition_table, pd.DataFrame):
        duplicates = sorted(
            {
                str(field)
                for field in condition_table.columns
                if list(condition_table.columns).count(field) > 1
            }
        )
        if duplicates:
            raise ValueError(
                f"Condition table contains duplicate columns: {duplicates}."
            )
        records = condition_table.to_dict(orient="records")
    else:
        records = list(condition_table)

    condition_fields = set(simulator.diffusion_config.primary_fields)
    aliases = (
        ("phenotype", "Phenotype_Tokens", "Healthy"),
        ("body_site", "Body_Site_Coarse", "stool_feces"),
    )
    supported_aliases = {
        alias for alias, canonical, _ in aliases if canonical in condition_fields
    }
    allowed = {"n_samples", "conditions", *supported_aliases, *condition_fields}
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Condition-table row {index + 1} must be a mapping or data-frame row."
            )
        row = {str(field): value for field, value in raw.items() if not _missing(value)}
        unknown = sorted(set(row) - allowed)
        if unknown:
            raise ValueError(
                f"Condition-table row {index + 1} contains unknown fields: {unknown}. "
                f"Accepted fields: {sorted(allowed)}."
            )
        nested = row.pop("conditions", {})
        if not isinstance(nested, Mapping):
            raise ValueError(
                f"Condition-table row {index + 1} conditions must be a mapping."
            )
        conditions = {str(field): value for field, value in nested.items()}
        direct_conditions = {
            field: row.pop(field) for field in list(row) if field in condition_fields
        }
        repeated = sorted(set(conditions) & set(direct_conditions))
        if repeated:
            raise ValueError(
                f"Condition-table row {index + 1} repeats condition fields: {repeated}."
            )
        conditions.update(direct_conditions)

        resolved: dict[str, object] = {
            "phenotype": "Healthy",
            "body_site": "stool_feces",
        }
        for alias, canonical, default in aliases:
            if canonical not in condition_fields:
                continue
            if alias in row and canonical in conditions:
                raise ValueError(
                    f"Condition-table row {index + 1} repeats equivalent fields "
                    f"{alias!r} and {canonical!r}."
                )
            resolved[alias] = row.pop(alias, conditions.pop(canonical, default))
        resolved["n_samples"] = _sample_count(row.pop("n_samples", 100))
        if row:
            raise ValueError(
                f"Condition-table row {index + 1} contains unknown fields: {sorted(row)}."
            )
        resolved["phenotype"] = str(resolved["phenotype"])
        resolved["body_site"] = str(resolved["body_site"])
        resolved["conditions"] = conditions
        normalized.append(resolved)
    if not normalized:
        raise ValueError("condition_table must contain at least one row.")
    return normalized


def _generate_from_condition_table(
    simulator: Simulator | BatchNeutralBank,
    condition_table: pd.DataFrame | Sequence[Mapping[str, object]],
    *,
    output_mode: str = "relative",
    guidance_scale: float = 1.25,
    support_temperature: float = 0.60,
    seed: int = 42,
    library_size: int | None = None,
    use_body_system: bool | None = None,
    batch_size: int = 64,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    max_in_memory_mb: float = DEFAULT_MAX_IN_MEMORY_MB,
    load: bool = True,
    config: Mapping[str, object] | None = None,
) -> GenerationOutput:
    """Generate samples for the rows in a condition table."""

    overwrite = require_bool(overwrite, "overwrite")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, (int, np.integer))
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    if not np.isfinite(max_in_memory_mb) or max_in_memory_mb <= 0:
        raise ValueError("max_in_memory_mb must be finite and positive.")
    if type(load) is not bool:
        raise ValueError("load must be true or false.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not load and output_dir is None:
        raise ValueError("load=False requires output_dir for streamed generation.")
    if isinstance(simulator, BatchNeutralBank):
        if guidance_scale != 1.25:
            raise ValueError(
                "guidance_scale does not apply when generating from a bank."
            )
        if use_body_system is not None:
            raise ValueError(
                "use_body_system does not apply when generating from a bank."
            )
    normalized = _normalize_condition_table(simulator, condition_table)
    bank_requests: list[tuple[dict[str, str], tuple[tuple[str, str], ...]]] = []
    bank_selections: dict[tuple[tuple[str, str], ...], np.ndarray] = {}
    bank_offsets: dict[tuple[tuple[str, str], ...], int] = {}
    bank_seeds: dict[tuple[tuple[str, str], ...], int] = {}
    if isinstance(simulator, BatchNeutralBank):
        totals: dict[tuple[tuple[str, str], ...], int] = {}
        conditions_by_key: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
        for request in normalized:
            resolved = simulator._resolved_conditions(
                str(request["phenotype"]),
                str(request["body_site"]),
                dict(request["conditions"]),
            )
            key = tuple((field, resolved[field]) for field in simulator.condition_fields)
            bank_requests.append((resolved, key))
            conditions_by_key[key] = resolved
            totals[key] = totals.get(key, 0) + int(request["n_samples"])
        for key, total in totals.items():
            condition_seed = _condition_seed(int(seed), conditions_by_key[key])
            bank_seeds[key] = condition_seed
            bank_offsets[key] = 0
            bank_selections[key] = simulator.select_indices(
                conditions_by_key[key], n_samples=total, seed=condition_seed
            )
    total_samples = sum(int(request["n_samples"]) for request in normalized)
    estimated = estimate_generation_memory_mb(
        total_samples,
        len(simulator.taxa),
        output_mode=output_mode,
        batch_size=batch_size,
    )
    collect = load and estimated <= max_in_memory_mb
    if not collect and output_dir is None:
        raise _memory_error(estimated, max_in_memory_mb)

    output = None if output_dir is None else Path(output_dir).expanduser().resolve()
    abundance_parts: list[pd.DataFrame] = []
    count_parts: list[pd.DataFrame] = []
    metadata_parts: list[pd.DataFrame] = []
    resolved_rows: list[dict[str, object]] = []
    files: dict[str, Path] = {}
    context = (
        nullcontext(None)
        if output is None
        else staged_output_dir(output, overwrite=overwrite)
    )
    with context as staging:
        handles: dict[str, TextIO] = {}
        sample_cursor = 0
        first = True
        try:
            if staging is not None:
                files = {
                    "abundance": output / "abundance.csv",
                    "metadata": output / "metadata.csv",
                    "config": output / "generation.yaml",
                }
                handles["abundance"] = (staging / "abundance.csv").open(
                    "w", encoding="utf-8", newline=""
                )
                handles["metadata"] = (staging / "metadata.csv").open(
                    "w", encoding="utf-8", newline=""
                )
                if output_mode == "observed":
                    files["counts"] = output / "counts.csv"
                    handles["counts"] = (staging / "counts.csv").open(
                        "w", encoding="utf-8", newline=""
                    )
            for request_index, request in enumerate(normalized):
                resolved_conditions: dict[str, str] | None = None
                n_samples = int(request["n_samples"])
                bank_indices = None
                bank_random_start = 0
                bank_seed = int(seed)
                if isinstance(simulator, BatchNeutralBank):
                    resolved_conditions, key = bank_requests[request_index]
                    bank_random_start = bank_offsets[key]
                    bank_seed = bank_seeds[key]
                    bank_indices = bank_selections[key][
                        bank_random_start : bank_random_start + n_samples
                    ]
                    bank_offsets[key] += n_samples
                for start in range(0, n_samples, batch_size):
                    chunk_indices = np.arange(
                        sample_cursor + start,
                        sample_cursor + min(start + batch_size, n_samples),
                        dtype=np.int64,
                    )
                    if bank_indices is None:
                        generated = simulator._generate_indexed(
                            phenotype=str(request["phenotype"]),
                            body_site=str(request["body_site"]),
                            sample_indices=chunk_indices,
                            output_mode=output_mode,
                            guidance_scale=guidance_scale,
                            support_temperature=support_temperature,
                            seed=seed,
                            library_size=library_size,
                            use_body_system=use_body_system,
                            conditions=dict(request["conditions"]),
                        )
                    else:
                        generated = simulator.generate_from_indices(
                            bank_indices[start : start + batch_size],
                            resolved_conditions or {},
                            output_mode=output_mode,
                            support_temperature=support_temperature,
                            seed=bank_seed,
                            library_size=library_size,
                            sample_indices=chunk_indices,
                            random_indices=np.arange(
                                bank_random_start + start,
                                bank_random_start + min(start + batch_size, n_samples),
                                dtype=np.int64,
                            ),
                            output_seed=int(seed),
                        )
                    if resolved_conditions is None:
                        resolved_conditions = {
                            field: str(generated.metadata[field].iloc[0])
                            for field in simulator.diffusion_config.primary_fields
                        }
                    if collect:
                        abundance_parts.append(generated.abundance)
                        metadata_parts.append(generated.metadata)
                        if generated.counts is not None:
                            count_parts.append(generated.counts)
                    if staging is not None:
                        _append_frame(handles["abundance"], generated.abundance, first)
                        _append_frame(handles["metadata"], generated.metadata, first)
                        if generated.counts is not None:
                            _append_frame(handles["counts"], generated.counts, first)
                        first = False
                resolved_rows.append(
                    {
                        "n_samples": n_samples,
                        "conditions": resolved_conditions or {},
                    }
                )
                sample_cursor += n_samples
        finally:
            for handle in handles.values():
                handle.close()

        resolved_config: dict[str, object] = {
            **dict(config or {}),
            "condition_table": resolved_rows,
            "n_samples": total_samples,
            "output_mode": output_mode,
            "library_size": library_size,
            "support_temperature": support_temperature,
            "seed": seed,
            "batch_size": batch_size,
            "max_in_memory_mb": max_in_memory_mb,
            "load": load,
            "loaded": collect,
        }
        if isinstance(simulator, Simulator):
            resolved_config["guidance_scale"] = guidance_scale
            resolved_config["use_body_system"] = simulator.bundle.supports_body_system
        if staging is not None:
            write_yaml(staging / "generation.yaml", resolved_config)

    abundance = (
        pd.concat(abundance_parts, ignore_index=True) if abundance_parts else None
    )
    metadata = pd.concat(metadata_parts, ignore_index=True) if metadata_parts else None
    counts = pd.concat(count_parts, ignore_index=True) if count_parts else None
    return GenerationOutput(
        output_dir=output,
        abundance=abundance,
        counts=counts,
        metadata=metadata,
        config=resolved_config,
        files=files,
        estimated_memory_mb=estimated,
    )


def generate_from_condition_table(
    model: Simulator | None = None,
    condition_table: pd.DataFrame | Sequence[Mapping[str, object]] | None = None,
    *,
    bank: BatchNeutralBank | None = None,
    output_mode: str = "relative",
    guidance_scale: float = 1.25,
    support_temperature: float = 0.60,
    seed: int = 42,
    library_size: int | None = None,
    use_body_system: bool | None = None,
    batch_size: int = 64,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    max_in_memory_mb: float = DEFAULT_MAX_IN_MEMORY_MB,
    load: bool = True,
) -> GenerationOutput:
    """Generate samples for every row of a condition table.

    ``condition_table`` is a DataFrame or sequence of mappings. Each row accepts
    ``n_samples`` (default 100), model condition fields, and the ``phenotype`` or
    ``body_site`` aliases when the model contains their standard fields.
    Multilabel values may be sequences or ``|``/``;``-delimited strings. Provide
    exactly one of ``model`` and ``bank``. Small requests return tables in a
    ``GenerationOutput``; large requests require ``output_dir`` and are streamed.
    """

    if (model is None) == (bank is None):
        raise ValueError("Provide exactly one of model or bank.")
    if condition_table is None:
        raise ValueError("condition_table is required.")
    source = model if model is not None else bank
    assert source is not None
    return _generate_from_condition_table(
        source,
        condition_table,
        output_mode=output_mode,
        guidance_scale=guidance_scale,
        support_temperature=support_temperature,
        seed=seed,
        library_size=library_size,
        use_body_system=use_body_system,
        batch_size=batch_size,
        output_dir=output_dir,
        overwrite=overwrite,
        max_in_memory_mb=max_in_memory_mb,
        load=load,
        config=(
            {
                "generation_source": "bank",
                "bank": {
                    "path": str(bank.root),
                    "type": str(bank.manifest["bank_type"]),
                    "schema_version": int(bank.manifest.get("schema_version", 1)),
                },
            }
            if bank is not None
            else {
                "generation_source": "model",
                "model": {
                    "path": str(model.bundle.root),
                    "type": model.bundle.model_type,
                    "condition_fields": list(model.diffusion_config.primary_fields),
                    "supports_body_system": model.bundle.supports_body_system,
                },
            }
        ),
    )


def generate(
    model: Simulator | None = None,
    *,
    bank: BatchNeutralBank | None = None,
    phenotype: str = "Healthy",
    body_site: str = "stool_feces",
    n_samples: int = 100,
    conditions: Mapping[str, object] | None = None,
    output_mode: str = "relative",
    guidance_scale: float = 1.25,
    support_temperature: float = 0.60,
    seed: int = 42,
    library_size: int | None = None,
    use_body_system: bool | None = None,
    batch_size: int = 64,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    max_in_memory_mb: float = DEFAULT_MAX_IN_MEMORY_MB,
    load: bool = True,
) -> GenerationOutput:
    """Generate one conditional microbiome dataset from a model or bank.

    Provide exactly one of ``model`` and ``bank``. ``conditions`` maps any
    additional model condition fields to scalar or multilabel values.
    ``phenotype`` and ``body_site`` apply only when those standard fields occur
    in the selected model. ``output_mode='observed'`` also returns counts. Large
    requests are streamed when ``output_dir`` is supplied and then return
    ``GenerationOutput.loaded == False``.
    """

    if (model is None) == (bank is None):
        raise ValueError("Provide exactly one of model or bank.")
    supplied = dict(conditions or {})
    repeated = sorted(set(supplied) & {"Phenotype_Tokens", "Body_Site_Coarse"})
    if repeated:
        raise ValueError(
            "Use phenotype and body_site instead of repeating standard fields in "
            f"conditions: {', '.join(repeated)}."
        )
    source = model if model is not None else bank
    assert source is not None
    request: dict[str, object] = {
        "n_samples": n_samples,
        "conditions": supplied,
    }
    condition_fields = set(source.diffusion_config.primary_fields)
    if "Phenotype_Tokens" in condition_fields:
        request["phenotype"] = phenotype
    if "Body_Site_Coarse" in condition_fields:
        request["body_site"] = body_site
    return generate_from_condition_table(
        model,
        [request],
        bank=bank,
        output_mode=output_mode,
        guidance_scale=guidance_scale,
        support_temperature=support_temperature,
        seed=seed,
        library_size=library_size,
        use_body_system=use_body_system,
        batch_size=batch_size,
        output_dir=output_dir,
        overwrite=overwrite,
        max_in_memory_mb=max_in_memory_mb,
        load=load,
    )
