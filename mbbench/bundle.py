from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import load_yaml


REQUIRED_FILES = (
    "model.yaml",
    "tokenizer.pt",
    "diffusion.pt",
    "taxa.csv",
    "vocabularies.json",
)


@dataclass(frozen=True)
class ModelBundle:
    """Model weights and metadata required for generation."""

    root: Path
    config: dict[str, Any]
    vocabularies: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {root}")
        missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
        if missing:
            raise ValueError(
                f"Model directory is incomplete; missing: {', '.join(missing)}"
            )
        config = load_yaml(root / "model.yaml")
        raw_vocabularies = json.loads(
            (root / "vocabularies.json").read_text(encoding="utf-8")
        )
        if not isinstance(raw_vocabularies, dict):
            raise ValueError("vocabularies.json must contain a JSON object.")
        bundle = cls(root=root, config=config, vocabularies=raw_vocabularies)
        bundle.validate()
        return bundle

    def validate(self) -> None:
        model_type = str(self.config.get("model_type", "")).lower()
        if model_type not in {"ae", "vae"}:
            raise ValueError("model.yaml model_type must be 'ae' or 'vae'.")
        fields = self.config.get("condition_fields")
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(field, str) or not field for field in fields)
            or len(fields) != len(set(fields))
        ):
            raise ValueError(
                "model.yaml condition_fields must be a unique non-empty string list."
            )
        modes = self.config.get("supported_output_modes")
        if (
            not isinstance(modes, list)
            or not modes
            or not set(modes).issubset({"relative", "observed"})
            or len(modes) != len(set(modes))
        ):
            raise ValueError(
                "supported_output_modes may contain only relative and observed."
            )
        missing_vocabularies = [
            field for field in fields if field not in self.vocabularies
        ]
        if missing_vocabularies:
            raise ValueError(
                f"Model bundle is missing condition vocabularies: {missing_vocabularies}"
            )
        for field, vocabulary in self.vocabularies.items():
            if not isinstance(vocabulary, dict):
                raise ValueError(f"Vocabulary for {field!r} must be a JSON object.")
            kind = vocabulary.get("type", vocabulary.get("kind", "categorical"))
            if kind not in {"categorical", "multilabel"}:
                raise ValueError(
                    f"Vocabulary for {field!r} has unsupported type {kind!r}."
                )
            values = vocabulary.get("values", vocabulary.get("tokens"))
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(
                    f"Vocabulary for {field!r} must contain unique non-empty strings."
                )
        taxa = pd.read_csv(self.root / "taxa.csv", dtype=str)
        if list(taxa.columns) != ["taxon"]:
            raise ValueError("taxa.csv must contain exactly one column named taxon.")
        if (
            taxa["taxon"].isna().any()
            or taxa["taxon"].eq("").any()
            or taxa["taxon"].duplicated().any()
        ):
            raise ValueError("taxa.csv must contain unique non-empty taxa.")
        if "observed" in modes and not (self.root / "donors.csv").is_file():
            raise ValueError("Observed output is declared, but donors.csv is missing.")
        if "observed" in modes:
            donors = pd.read_csv(self.root / "donors.csv")
            required_donor_fields = {"Project_Group", "raw_total_reads"}
            missing_donor_fields = sorted(required_donor_fields - set(donors.columns))
            if missing_donor_fields:
                raise ValueError(
                    f"donors.csv is missing required columns: {missing_donor_fields}."
                )
            depths = pd.to_numeric(donors["raw_total_reads"], errors="coerce")
            if donors.empty or depths.isna().any() or bool((depths <= 0).any()):
                raise ValueError(
                    "donors.csv raw_total_reads values must be finite and positive."
                )
        combinations = self.root / "condition_combinations.csv"
        if combinations.is_file():
            combination_fields = set(pd.read_csv(combinations, nrows=0).columns)
            missing_combination_fields = sorted(set(fields) - combination_fields)
            if missing_combination_fields:
                raise ValueError(
                    "condition_combinations.csv is missing fields: "
                    + ", ".join(missing_combination_fields)
                    + "."
                )
        if bool(self.config.get("supports_body_system", False)):
            mapping = self.root / "body_site_to_system.csv"
            if not mapping.is_file():
                raise ValueError(
                    "Body-System model bundle is missing body_site_to_system.csv."
                )

    @property
    def model_type(self) -> str:
        return str(self.config["model_type"]).lower()

    @property
    def supports_body_system(self) -> bool:
        return bool(self.config.get("supports_body_system", False))

    def condition_values(self, field: str) -> list[str]:
        payload = self.vocabularies.get(field)
        if not isinstance(payload, dict):
            raise KeyError(f"Model bundle has no vocabulary for {field!r}.")
        values = payload.get("values", payload.get("tokens"))
        if not isinstance(values, list):
            raise ValueError(
                f"Vocabulary for {field!r} does not contain values or tokens."
            )
        excluded = {
            "<UNK>",
            "<NULL>",
            "<MISSING>",
            "__UNK__",
            "__NULL__",
            "__MISSING__",
        }
        return [str(value) for value in values if str(value) not in excluded]
