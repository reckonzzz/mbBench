from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .common import staged_output_dir, write_yaml


DEFAULT_TECHNICAL_FIELDS = (
    "Instrument",
    "Variable_Region",
    "Sequencing_Type",
    "Library_Layout",
)
DEFAULT_BIOLOGICAL_FIELDS = ("Phenotype_Tokens", "Body_Site_Coarse")
DEFAULT_MULTILABEL_FIELDS = ("Phenotype_Tokens",)


def _read_indexed_csv(path: str | Path, sample_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if sample_column not in frame:
        raise ValueError(f"{path} is missing sample column {sample_column!r}.")
    if (
        frame[sample_column].isna().any()
        or frame[sample_column].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"{path} contains missing or empty sample IDs.")
    frame[sample_column] = frame[sample_column].astype(str)
    if frame[sample_column].duplicated().any():
        raise ValueError(f"{path} contains duplicate sample IDs.")
    return frame.set_index(sample_column, drop=False)


def normalize_abundance(frame: pd.DataFrame, sample_column: str) -> pd.DataFrame:
    """Validate nonnegative abundance and close every sample to one."""

    feature_columns = [column for column in frame.columns if column != sample_column]
    if not feature_columns:
        raise ValueError("Abundance table has no taxon columns.")
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Abundance values must be finite and nonnegative.")
    if not np.allclose(values, np.round(values), rtol=0.0, atol=1e-8):
        raise ValueError(
            "Training abundance input must contain genus counts, not relative abundance."
        )
    totals = values.sum(axis=1)
    if np.any(totals <= 0):
        sample = frame.index[np.flatnonzero(totals <= 0)[0]]
        raise ValueError(f"Sample {sample!r} has zero total abundance.")
    normalized = values / totals[:, None]
    output = pd.DataFrame(normalized, index=frame.index, columns=feature_columns)
    output.insert(0, sample_column, frame[sample_column])
    return output


def _split_tokens(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return sorted(
        {
            token.strip()
            for token in str(value).replace("|", ";").split(";")
            if token.strip()
        }
    )


def build_vocabularies(
    metadata: pd.DataFrame,
    fields: Sequence[str],
    multilabel_fields: Sequence[str] = DEFAULT_MULTILABEL_FIELDS,
) -> dict[str, dict[str, Any]]:
    vocabularies: dict[str, dict[str, Any]] = {}
    multilabel_fields = set(multilabel_fields)
    for field in fields:
        if field not in metadata:
            continue
        if field in multilabel_fields:
            tokens = sorted(
                {token for value in metadata[field] for token in _split_tokens(value)}
            )
            vocabularies[field] = {"type": "multilabel", "tokens": tokens}
        else:
            values = sorted(
                {str(value) for value in metadata[field].dropna() if str(value)}
            )
            if not values:
                continue
            vocabularies[field] = {
                "type": "categorical",
                "values": ["__NULL__", "__UNK__", *values],
            }
    return vocabularies


def _random_split(
    metadata: pd.DataFrame,
    biological_fields: Sequence[str],
    multilabel_fields: Sequence[str],
    seed: int,
) -> pd.Series:
    fields = metadata[list(biological_fields)].copy()
    for field in set(multilabel_fields) & set(fields.columns):
        fields[field] = fields[field].map(
            lambda value: "|".join(_split_tokens(value)) or "<NULL>"
        )
    labels = fields.fillna("<NULL>").astype(str).agg("||".join, axis=1)
    counts = labels.value_counts()
    rare = counts[counts < 10]
    if len(rare):
        raise ValueError(
            "A 80/10/10 stratified split requires at least ten samples per biological-condition combination. "
            "Provide a split column or published split lists for sparse combinations."
        )
    train_ids, held_ids = train_test_split(
        metadata.index.to_numpy(), test_size=0.20, random_state=seed, stratify=labels
    )
    held_labels = labels.loc[held_ids]
    validation_ids, test_ids = train_test_split(
        held_ids, test_size=0.50, random_state=seed + 1, stratify=held_labels
    )
    split = pd.Series(index=metadata.index, dtype="object")
    split.loc[train_ids] = "train"
    split.loc[validation_ids] = "validation"
    split.loc[test_ids] = "test"
    return split


def prepare_data(
    *,
    abundance_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    sample_column: str = "Sample",
    phenotype_column: str = "Phenotype_Tokens",
    body_site_column: str = "Body_Site_Coarse",
    split_column: str | None = None,
    split_file: str | Path | None = None,
    body_system_map: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
    biological_fields: Sequence[str] = DEFAULT_BIOLOGICAL_FIELDS,
    multilabel_fields: Sequence[str] = DEFAULT_MULTILABEL_FIELDS,
    technical_fields: Sequence[str] = DEFAULT_TECHNICAL_FIELDS,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Align abundance and metadata, normalize abundance, and create or retain a sample split."""

    abundance = _read_indexed_csv(abundance_path, sample_column)
    metadata = _read_indexed_csv(metadata_path, sample_column)
    if set(abundance.index) != set(metadata.index):
        missing_abundance = sorted(set(metadata.index) - set(abundance.index))[:5]
        missing_metadata = sorted(set(abundance.index) - set(metadata.index))[:5]
        raise ValueError(
            f"Abundance and metadata sample IDs differ; missing abundance={missing_abundance}, "
            f"missing metadata={missing_metadata}."
        )
    metadata = metadata.loc[abundance.index].copy()
    biological_fields = tuple(dict.fromkeys(str(field) for field in biological_fields))
    multilabel_fields = tuple(dict.fromkeys(str(field) for field in multilabel_fields))
    if not biological_fields:
        raise ValueError(
            "At least one biological field is required for conditional generation."
        )
    invalid_multilabel = sorted(set(multilabel_fields) - set(biological_fields))
    if invalid_multilabel:
        raise ValueError(
            f"Multilabel fields must also be selected biological fields: {invalid_multilabel}"
        )
    if "Project_Group" in biological_fields:
        raise ValueError(
            "Project_Group is modeled separately as z_project and cannot be selected as z_bio."
        )
    if phenotype_column in metadata:
        metadata["Phenotype_Tokens"] = metadata[phenotype_column]
    elif "Phenotype_Tokens" in biological_fields:
        raise ValueError(f"Metadata is missing phenotype column {phenotype_column!r}.")
    if body_site_column in metadata:
        metadata["Body_Site_Coarse"] = metadata[body_site_column]
    elif "Body_Site_Coarse" in biological_fields or body_system_map is not None:
        raise ValueError(f"Metadata is missing Body Site column {body_site_column!r}.")
    if (
        "Phenotype_Tokens" in biological_fields
        and not metadata["Phenotype_Tokens"]
        .map(lambda value: bool(_split_tokens(value)))
        .any()
    ):
        raise ValueError("Metadata contains no usable phenotype tokens.")
    if (
        "Body_Site_Coarse" in biological_fields
        and not metadata["Body_Site_Coarse"].notna().any()
    ):
        raise ValueError("Metadata contains no usable Body Site values.")
    if "Body_System_Model" in metadata:
        metadata = metadata.drop(columns="Body_System_Model")
    mapping = None
    if body_system_map is not None:
        mapping = pd.read_csv(body_system_map)
        if (
            list(mapping.columns) != ["body_site", "body_system"]
            or mapping["body_site"].duplicated().any()
        ):
            raise ValueError(
                "Body-System mapping must have unique body_site and body_system columns."
            )
        mapping["body_site"] = mapping["body_site"].astype(str)
        mapping["body_system"] = mapping["body_system"].astype(str)
        lookup = mapping.set_index("body_site")["body_system"]
        observed_sites = set(metadata["Body_Site_Coarse"].dropna().astype(str))
        missing_sites = sorted(observed_sites - set(lookup.index.astype(str)))
        if missing_sites:
            raise ValueError(
                f"Body-System mapping is missing Body Sites: {missing_sites}"
            )
        metadata["Body_System_Model"] = metadata["Body_Site_Coarse"].map(lookup)
    normalized = normalize_abundance(abundance, sample_column)
    if split_column and split_file:
        raise ValueError("Use either split_column or split_file, not both.")
    if split_file:
        source = Path(split_file)
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Split JSON must map split names to sample-ID lists.")
            assignments: dict[str, str] = {}
            for label, samples in payload.items():
                if not isinstance(samples, list):
                    raise ValueError(
                        f"Split JSON entry {label!r} must contain a sample-ID list."
                    )
                normalized_label = (
                    "validation" if label in {"val", "validation"} else str(label)
                )
                for sample in samples:
                    sample_id = str(sample)
                    if sample_id in assignments:
                        raise ValueError(
                            f"Split JSON assigns sample {sample_id!r} more than once."
                        )
                    assignments[sample_id] = normalized_label
            missing_ids = sorted(set(metadata.index) - set(assignments))
            extra_ids = sorted(set(assignments) - set(metadata.index))
            if missing_ids or extra_ids:
                raise ValueError(
                    "Split JSON sample IDs do not exactly match retained samples; "
                    f"missing={missing_ids[:5]}, extra={extra_ids[:5]}."
                )
            split = pd.Series(
                metadata.index.map(assignments), index=metadata.index, dtype="object"
            )
        else:
            assignments = _read_indexed_csv(source, sample_column)
            if "split" not in assignments:
                raise ValueError("Split CSV must contain a split column.")
            missing_ids = sorted(set(metadata.index) - set(assignments.index))
            extra_ids = sorted(set(assignments.index) - set(metadata.index))
            if missing_ids or extra_ids:
                raise ValueError(
                    "Split CSV sample IDs do not exactly match retained samples; "
                    f"missing={missing_ids[:5]}, extra={extra_ids[:5]}."
                )
            split = assignments["split"].reindex(metadata.index).astype("object")
        if split.isna().any():
            raise ValueError("Split file does not assign every retained sample.")
        split = split.astype(str).str.lower().replace({"val": "validation"})
    elif split_column:
        if split_column not in metadata:
            raise ValueError(f"Metadata is missing split column {split_column!r}.")
        split = (
            metadata[split_column]
            .astype(str)
            .str.lower()
            .replace({"val": "validation"})
        )
    else:
        split = _random_split(metadata, biological_fields, multilabel_fields, seed)
    invalid = sorted(set(split) - {"train", "validation", "test"})
    if invalid:
        raise ValueError(f"Split assignments contain invalid values: {invalid}")
    missing_splits = sorted({"train", "validation", "test"} - set(split))
    if missing_splits:
        raise ValueError(
            f"Split assignments are missing required splits: {missing_splits}"
        )
    metadata["split"] = split
    read_depth_stats: dict[str, float] = {}
    if "raw_total_reads" in metadata:
        train_depth = pd.to_numeric(
            metadata.loc[split == "train", "raw_total_reads"], errors="raise"
        ).to_numpy()
        if not np.isfinite(train_depth).all() or np.any(train_depth <= 0):
            raise ValueError("raw_total_reads must contain finite positive values.")
        transformed = np.log1p(train_depth)
        median = float(np.median(transformed))
        iqr = float(np.quantile(transformed, 0.75) - np.quantile(transformed, 0.25))
        read_depth_stats = {
            "transform": "log1p_train_median_iqr",
            "median": median,
            "iqr": max(iqr, 1e-6),
            "raw_median": float(np.median(train_depth)),
            "train_sample_count": int(len(train_depth)),
        }
    technical_fields = tuple(dict.fromkeys(str(field) for field in technical_fields))
    reserved = {
        "Phenotype_Tokens",
        "Body_Site_Coarse",
        "Body_System_Model",
        "Project_Group",
    }
    invalid_technical = sorted(set(technical_fields) & reserved)
    if invalid_technical:
        raise ValueError(
            f"Technical fields must not reuse biological or Project fields: {invalid_technical}"
        )
    missing_technical = [field for field in technical_fields if field not in metadata]
    if missing_technical:
        raise ValueError(
            f"Metadata is missing requested technical fields: {missing_technical}"
        )
    missing_biological = [field for field in biological_fields if field not in metadata]
    if missing_biological:
        raise ValueError(
            f"Metadata is missing requested biological fields: {missing_biological}"
        )
    overlap = sorted(set(biological_fields) & set(technical_fields))
    if overlap:
        raise ValueError(f"Fields cannot be both biological and technical: {overlap}")
    hierarchy_fields = (
        ["Body_System_Model"]
        if body_system_map is not None and "Body_System_Model" not in biological_fields
        else []
    )
    fields = [*biological_fields, *hierarchy_fields, *technical_fields, "Project_Group"]
    vocabularies = build_vocabularies(
        metadata.loc[split == "train"], fields, multilabel_fields
    )
    missing_vocabularies = [
        field for field in biological_fields if field not in vocabularies
    ]
    if missing_vocabularies:
        raise ValueError(
            f"Selected biological fields contain no usable values: {missing_vocabularies}"
        )

    taxonomy = None
    if taxonomy_path is not None:
        source = Path(taxonomy_path)
        if source.suffix.lower() == ".json":
            family_map = json.loads(source.read_text(encoding="utf-8")).get(
                "family", {}
            )
            taxonomy = pd.DataFrame(
                {
                    "taxon": normalized.columns[1:],
                    "family": [
                        family_map.get(taxon, "unknown")
                        for taxon in normalized.columns[1:]
                    ],
                }
            )
        else:
            taxonomy = pd.read_csv(source)
            if not {"taxon", "family"}.issubset(taxonomy.columns):
                raise ValueError("Taxonomy CSV must contain taxon and family columns.")
            taxonomy = (
                taxonomy.set_index("taxon")
                .reindex(normalized.columns[1:])
                .fillna("unknown")
                .reset_index()
            )
    output = Path(output_dir).expanduser().resolve()
    with staged_output_dir(output, overwrite=overwrite) as staging:
        normalized.to_csv(staging / "abundance.csv", index=False)
        metadata.reset_index(drop=True).to_csv(staging / "metadata.csv", index=False)
        pd.DataFrame(
            {"taxon": [column for column in normalized.columns if column != sample_column]}
        ).to_csv(staging / "taxa.csv", index=False)
        if taxonomy is not None:
            taxonomy.to_csv(staging / "taxonomy.csv", index=False)
        (staging / "vocabularies.json").write_text(
            json.dumps(vocabularies, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if mapping is not None:
            mapping.to_csv(staging / "body_site_to_system.csv", index=False)
        write_yaml(
            staging / "data.yaml",
            {
                "sample_column": sample_column,
                "phenotype_field": "Phenotype_Tokens",
                "body_site_field": "Body_Site_Coarse",
                "split_seed": seed if split_column is None and split_file is None else None,
                "split_source": (
                    str(split_file) if split_file else split_column or "generated"
                ),
                "n_samples": len(metadata),
                "n_taxa": len(normalized.columns) - 1,
                "read_depth_stats": read_depth_stats,
                "taxonomy_source": str(taxonomy_path) if taxonomy_path else None,
                "biological_fields": list(biological_fields),
                "multilabel_fields": list(multilabel_fields),
                "technical_fields": list(technical_fields),
            },
        )
    return output


@dataclass(frozen=True)
class PreparedData:
    root: Path
    abundance: pd.DataFrame
    metadata: pd.DataFrame
    taxa: list[str]
    vocabularies: dict[str, dict[str, Any]]
    sample_column: str
    read_depth_stats: dict[str, float]
    biological_fields: tuple[str, ...]
    multilabel_fields: tuple[str, ...]
    technical_fields: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "PreparedData":
        root = Path(path).expanduser().resolve()
        config_path = root / "data.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Prepared data is missing data.yaml: {root}")
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        sample_column = str(config["sample_column"])
        abundance = _read_indexed_csv(root / "abundance.csv", sample_column)
        metadata = _read_indexed_csv(root / "metadata.csv", sample_column).loc[
            abundance.index
        ]
        taxa = pd.read_csv(root / "taxa.csv")["taxon"].astype(str).tolist()
        vocabularies = json.loads(
            (root / "vocabularies.json").read_text(encoding="utf-8")
        )
        return cls(
            root,
            abundance,
            metadata,
            taxa,
            vocabularies,
            sample_column,
            config.get("read_depth_stats", {}),
            tuple(config.get("biological_fields", DEFAULT_BIOLOGICAL_FIELDS)),
            tuple(config.get("multilabel_fields", DEFAULT_MULTILABEL_FIELDS)),
            tuple(config.get("technical_fields", DEFAULT_TECHNICAL_FIELDS)),
        )


class MicrobiomeDataset(Dataset):
    def __init__(
        self,
        data: PreparedData,
        split: str,
        *,
        required_fields: Sequence[str] = (),
        require_nuisance_complete: bool = False,
    ) -> None:
        self.data = data
        metadata = data.metadata[data.metadata["split"] == split].copy()
        selected = pd.Series(True, index=metadata.index)
        for field in required_fields:
            if field == "Phenotype_Tokens":
                selected &= metadata[field].map(
                    lambda value: bool(_split_tokens(value))
                )
            else:
                selected &= metadata[field].notna()
        if require_nuisance_complete:
            nuisance_fields = [
                field
                for field in (*data.technical_fields, "Project_Group")
                if field in data.vocabularies
            ]
            if nuisance_fields:
                selected &= metadata[nuisance_fields].notna().all(axis=1)
        metadata = metadata[selected]
        self.metadata = metadata
        self.sample_ids = metadata.index.tolist()
        self.abundance = torch.tensor(
            data.abundance.loc[self.sample_ids, data.taxa].to_numpy(),
            dtype=torch.float32,
        )
        self.mask = (self.abundance > 0).float()
        self.vocabularies = data.vocabularies
        self.categorical, self.multilabel, self.validity = self._encode_conditions()
        self.adversarial_weights = torch.ones(len(self.metadata), dtype=torch.float32)
        self.reconstruction_weights = torch.ones(
            len(self.metadata), dtype=torch.float32
        )
        nuisance_fields = [
            field
            for field in (*data.technical_fields, "Project_Group")
            if field in self.validity
        ]
        self.nuisance_complete = (
            torch.stack([self.validity[field].bool() for field in nuisance_fields]).all(
                dim=0
            )
            if nuisance_fields
            else torch.ones(len(self.metadata), dtype=torch.bool)
        )

    def _encode_conditions(
        self,
    ) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
    ]:
        categorical: dict[str, torch.Tensor] = {}
        multilabel: dict[str, torch.Tensor] = {}
        validity: dict[str, torch.Tensor] = {}
        for field, vocabulary in self.vocabularies.items():
            if field not in self.metadata:
                continue
            if vocabulary["type"] == "categorical":
                values = vocabulary["values"]
                lookup = {value: index for index, value in enumerate(values)}
                null_index = lookup.get("__NULL__", 0)
                unknown_index = lookup.get("__UNK__", null_index)
                categorical[field] = torch.tensor(
                    [
                        (
                            null_index
                            if pd.isna(value)
                            else lookup.get(str(value), unknown_index)
                        )
                        for value in self.metadata[field]
                    ],
                    dtype=torch.long,
                )
                validity[field] = torch.tensor(
                    self.metadata[field].notna().to_numpy(), dtype=torch.float32
                )
            else:
                tokens = vocabulary["tokens"]
                lookup = {value: index for index, value in enumerate(tokens)}
                encoded = torch.zeros(
                    len(self.metadata), len(tokens), dtype=torch.float32
                )
                for row, value in enumerate(self.metadata[field]):
                    for token in _split_tokens(value):
                        if token in lookup:
                            encoded[row, lookup[token]] = 1.0
                multilabel[field] = encoded
                validity[field] = torch.tensor(
                    [bool(_split_tokens(value)) for value in self.metadata[field]],
                    dtype=torch.float32,
                )
        return categorical, multilabel, validity

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = {
            "sample_id": self.sample_ids[index],
            "abundance": self.abundance[index],
            "mask": self.mask[index],
            "categorical": {
                field: values[index] for field, values in self.categorical.items()
            },
            "multilabel": {
                field: values[index] for field, values in self.multilabel.items()
            },
            "validity": {
                field: values[index] for field, values in self.validity.items()
            },
            "adversarial_weight": self.adversarial_weights[index],
            "nuisance_complete": self.nuisance_complete[index],
            "reconstruction_weight": self.reconstruction_weights[index],
        }
        if "raw_total_reads" in self.metadata:
            raw = np.log1p(float(self.metadata.iloc[index]["raw_total_reads"]))
            median = float(self.data.read_depth_stats.get("median", 0.0))
            iqr = max(float(self.data.read_depth_stats.get("iqr", 1.0)), 1e-6)
            row["read_depth"] = torch.tensor((raw - median) / iqr, dtype=torch.float32)
        return row
