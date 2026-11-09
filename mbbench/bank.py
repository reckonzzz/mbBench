from __future__ import annotations

import hashlib
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
from urllib.request import urlopen

import numpy as np
import pandas as pd

from ._download import copy_download, extract_zip
from .common import load_yaml
from .generation_core import decode_composition
from .simulator import GenerationResult, _sample_seed


BANK_TYPE = "batch-neutral-logit-bank"
OFFICIAL_BANK_URL = "https://zenodo.org/api/records/24082591/files/core_bank.zip/content"
OFFICIAL_BANK_SHA256 = "74fc7107c857d9c712974a0d368531119ddd673f335290b05a109ccd140d8e8a"
_REQUIRED_FILES = (
    "bank.yaml",
    "abundance_logits.npy",
    "support_logits.npy",
    "metadata.csv.gz",
    "taxa.csv",
)


def _condition_value(value: object, *, multilabel: bool = False) -> str:
    if isinstance(value, (list, tuple, set)):
        values = [str(item).strip() for item in value if str(item).strip()]
    elif multilabel:
        values = [
            item.strip()
            for item in str(value).replace(";", "|").split("|")
            if item.strip()
        ]
    else:
        return str(value)
    if len(values) != len(set(values)):
        raise ValueError("Multilabel conditions cannot contain duplicate values.")
    return "|".join(sorted(values)) if multilabel else "|".join(values)


class BatchNeutralBank:
    """Read batch-neutral abundance and support logits from a bank directory."""

    def __init__(
        self,
        root: Path,
        manifest: dict[str, object],
        abundance_logits: np.ndarray,
        support_logits: np.ndarray,
        metadata: pd.DataFrame,
        taxa: list[str],
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.abundance_logits = abundance_logits
        self.support_logits = support_logits
        self.metadata = metadata
        self.taxa = taxa
        self.condition_fields = tuple(
            str(value) for value in manifest["condition_fields"]
        )
        self.multilabel_fields = frozenset(
            str(value) for value in manifest.get("multilabel_fields", [])
        )
        self.support_threshold = float(manifest.get("support_threshold", 0.5))
        self.supports_body_system = bool(manifest.get("supports_body_system", False))
        self.diffusion_config = SimpleNamespace(primary_fields=self.condition_fields)
        self.bundle = SimpleNamespace(
            root=root,
            supports_body_system=self.supports_body_system,
        )

    @classmethod
    def from_directory(cls, path: str | Path) -> "BatchNeutralBank":
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Bank directory does not exist: {root}")
        missing = [name for name in _REQUIRED_FILES if not (root / name).is_file()]
        if missing:
            raise ValueError(f"Bank is missing required files: {', '.join(missing)}.")
        manifest = load_yaml(root / "bank.yaml")
        if str(manifest.get("bank_type")) != BANK_TYPE:
            raise ValueError(
                f"bank.yaml bank_type must be {BANK_TYPE!r}; "
                f"found {manifest.get('bank_type')!r}."
            )
        fields = manifest.get("condition_fields")
        if (
            not isinstance(fields, list)
            or not fields
            or any(not isinstance(field, str) or not field for field in fields)
        ):
            raise ValueError(
                "bank.yaml condition_fields must be a nonempty string list."
            )
        if len(fields) != len(set(fields)):
            raise ValueError("bank.yaml condition_fields contains duplicates.")
        multilabel_fields = manifest.get("multilabel_fields", [])
        if (
            not isinstance(multilabel_fields, list)
            or any(not isinstance(field, str) for field in multilabel_fields)
            or len(multilabel_fields) != len(set(multilabel_fields))
            or not set(multilabel_fields).issubset(fields)
        ):
            raise ValueError(
                "bank.yaml multilabel_fields must be a unique subset of condition_fields."
            )
        threshold = float(manifest.get("support_threshold", 0.5))
        if not np.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("bank.yaml support_threshold must lie in (0, 1).")

        metadata = pd.read_csv(root / "metadata.csv.gz", dtype=str)
        taxa_frame = pd.read_csv(root / "taxa.csv", dtype=str)
        if list(taxa_frame.columns) != ["taxon"]:
            raise ValueError("taxa.csv must contain exactly one column named taxon.")
        taxa = taxa_frame["taxon"].tolist()
        if not taxa or len(taxa) != len(set(taxa)):
            raise ValueError("taxa.csv must contain unique nonempty taxa.")
        required_metadata = {"sample_id", *fields}
        missing_columns = sorted(required_metadata - set(metadata.columns))
        if missing_columns:
            raise ValueError(
                f"metadata.csv.gz is missing required columns: {missing_columns}."
            )
        if (
            metadata["sample_id"].isna().any()
            or metadata["sample_id"].duplicated().any()
        ):
            raise ValueError("Bank sample_id values must be nonmissing and unique.")
        for field in multilabel_fields:
            metadata[field] = metadata[field].map(
                lambda value: _condition_value(value, multilabel=True)
            )
        with ExitStack() as handles:
            matrices = []
            for name in ("abundance_logits.npy", "support_logits.npy"):
                matrix = np.load(root / name, mmap_mode="r")
                if not isinstance(matrix, np.memmap):
                    matrix.close()
                    raise ValueError(f"{name} must contain a NumPy .npy matrix.")
                handles.callback(matrix._mmap.close)
                matrices.append(matrix)
            abundance, support = matrices
            if abundance.ndim != 2 or abundance.shape != support.shape:
                raise ValueError(
                    "Bank abundance and support logits must be matching matrices."
                )
            if abundance.shape != (len(metadata), len(taxa)):
                raise ValueError(
                    "Bank matrix shape must match metadata rows and taxa.csv rows."
                )
            bank = cls(root, manifest, abundance, support, metadata, taxa)
            handles.pop_all()
            return bank

    def close(self) -> None:
        """Close memory-mapped bank matrices."""

        for matrix in (self.abundance_logits, self.support_logits):
            mmap = getattr(matrix, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "BatchNeutralBank":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def condition_values(self, field: str) -> list[str]:
        if field not in self.condition_fields:
            raise ValueError(f"Bank does not contain condition field {field!r}.")
        if field in self.multilabel_fields:
            return sorted(
                {
                    token
                    for value in self.metadata[field].dropna().astype(str)
                    for token in value.split("|")
                    if token
                }
            )
        return sorted(self.metadata[field].dropna().astype(str).unique().tolist())

    def _resolved_conditions(
        self,
        phenotype: str,
        body_site: str,
        conditions: Mapping[str, object] | None,
    ) -> dict[str, str]:
        supplied = {
            str(field): _condition_value(
                value, multilabel=str(field) in self.multilabel_fields
            )
            for field, value in dict(conditions or {}).items()
        }
        aliases = {
            "Phenotype_Tokens": str(phenotype),
            "Body_Site_Coarse": str(body_site),
        }
        for field, value in aliases.items():
            if field in self.condition_fields:
                supplied.setdefault(field, value)
        unknown = sorted(set(supplied) - set(self.condition_fields))
        if unknown:
            raise ValueError(f"Bank does not contain condition fields: {unknown}.")
        missing = sorted(set(self.condition_fields) - set(supplied))
        if missing:
            raise ValueError(f"Generation requires bank condition fields: {missing}.")
        return {field: supplied[field] for field in self.condition_fields}

    def sample_logits(
        self,
        conditions: Mapping[str, object],
        *,
        n_samples: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = self.select_indices(conditions, n_samples=n_samples, seed=seed)
        abundance = np.asarray(self.abundance_logits[selected], dtype=np.float64)
        support = np.asarray(self.support_logits[selected], dtype=np.float64)
        if not np.isfinite(abundance).all() or not np.isfinite(support).all():
            raise ValueError("Selected bank logits contain nonfinite values.")
        source_ids = self.metadata.iloc[selected]["sample_id"].astype(str).to_numpy()
        return abundance, support, source_ids

    def select_indices(
        self,
        conditions: Mapping[str, object],
        *,
        n_samples: int,
        seed: int,
        exclude: set[int] | None = None,
    ) -> np.ndarray:
        if isinstance(n_samples, bool) or int(n_samples) != n_samples or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        resolved = {
            str(field): _condition_value(
                value, multilabel=str(field) in self.multilabel_fields
            )
            for field, value in conditions.items()
        }
        unknown = sorted(set(resolved) - set(self.condition_fields))
        missing = sorted(set(self.condition_fields) - set(resolved))
        if unknown:
            raise ValueError(f"Bank does not contain condition fields: {unknown}.")
        if missing:
            raise ValueError(f"Generation requires bank condition fields: {missing}.")
        match = np.ones(len(self.metadata), dtype=bool)
        for field in self.condition_fields:
            match &= self.metadata[field].astype(str).eq(resolved[field]).to_numpy()
        candidates = np.flatnonzero(match)
        if exclude:
            candidates = candidates[
                ~np.isin(candidates, np.fromiter(exclude, dtype=int))
            ]
        if len(candidates) < int(n_samples):
            detail = ", ".join(
                f"{field}={resolved[field]}" for field in self.condition_fields
            )
            raise ValueError(
                f"Bank has {len(candidates)} unused matching rows for {detail}; "
                f"{int(n_samples)} were requested."
            )
        return np.random.default_rng(seed).choice(
            candidates, size=int(n_samples), replace=False
        )

    def generate_from_indices(
        self,
        indices: np.ndarray,
        conditions: Mapping[str, object],
        *,
        output_mode: str,
        support_temperature: float,
        seed: int,
        library_size: int | None,
        sample_indices: np.ndarray | None = None,
        random_indices: np.ndarray | None = None,
        output_seed: int | None = None,
    ) -> GenerationResult:
        selected = np.asarray(indices, dtype=int)
        if sample_indices is None:
            sample_indices = np.arange(len(selected), dtype=np.int64)
        sample_indices = np.asarray(sample_indices)
        if (
            sample_indices.ndim != 1
            or len(sample_indices) != len(selected)
            or sample_indices.dtype.kind not in "iu"
            or bool((sample_indices < 0).any())
        ):
            raise ValueError(
                "sample_indices must contain one non-negative integer per selected row."
            )
        if random_indices is None:
            random_indices = sample_indices
        random_indices = np.asarray(random_indices)
        if (
            random_indices.ndim != 1
            or len(random_indices) != len(selected)
            or random_indices.dtype.kind not in "iu"
            or bool((random_indices < 0).any())
        ):
            raise ValueError(
                "random_indices must contain one non-negative integer per selected row."
            )
        abundance_logits = np.asarray(self.abundance_logits[selected], dtype=np.float64)
        support_logits = np.asarray(self.support_logits[selected], dtype=np.float64)
        if (
            not np.isfinite(abundance_logits).all()
            or not np.isfinite(support_logits).all()
        ):
            raise ValueError("Selected bank logits contain nonfinite values.")
        source_ids = self.metadata.iloc[selected]["sample_id"].astype(str).to_numpy()
        uniforms = np.stack(
            [
                np.random.default_rng(
                    _sample_seed(int(seed), 21, int(sample_index))
                ).random(support_logits.shape[1])
                for sample_index in random_indices
            ]
        )
        decoded = decode_composition(
            abundance_logits,
            support_logits,
            mode="temperature-bernoulli",
            threshold=self.support_threshold,
            temperature=support_temperature,
            uniforms=uniforms,
        )
        n_samples = len(selected)
        id_seed = int(seed if output_seed is None else output_seed)
        sample_ids = [
            f"mbbench_bank_{id_seed}_{int(index) + 1:06d}" for index in sample_indices
        ]
        abundance = pd.DataFrame(decoded.composition, columns=self.taxa)
        abundance.insert(0, "sample_id", sample_ids)
        metadata = pd.DataFrame({"sample_id": sample_ids})
        for field, value in conditions.items():
            metadata[field] = _condition_value(
                value, multilabel=field in self.multilabel_fields
            )
        metadata["source_bank_sample_id"] = source_ids
        metadata["condition_status"] = "bank-supported"
        counts = None
        if output_mode == "observed":
            if (
                library_size is None
                or isinstance(library_size, bool)
                or int(library_size) != library_size
                or library_size <= 0
            ):
                raise ValueError(
                    "Bank generation in observed mode requires a positive integer library_size."
                )
            count_values = np.stack(
                [
                    np.random.default_rng(
                        _sample_seed(int(seed), 41, int(sample_index))
                    ).multinomial(int(library_size), probabilities)
                    for sample_index, probabilities in zip(
                        random_indices, decoded.composition, strict=True
                    )
                ]
            )
            counts = pd.DataFrame(count_values, columns=self.taxa)
            counts.insert(0, "sample_id", sample_ids)
        return GenerationResult(
            abundance=abundance,
            counts=counts,
            metadata=metadata,
            abundance_logits=abundance_logits,
            support_logits=support_logits,
        )

    def generate(
        self,
        *,
        phenotype: str = "Healthy",
        body_site: str = "stool_feces",
        n_samples: int = 100,
        output_mode: str = "relative",
        guidance_scale: float = 1.25,
        support_temperature: float = 0.60,
        seed: int = 42,
        library_size: int | None = None,
        use_body_system: bool | None = None,
        conditions: Mapping[str, object] | None = None,
    ) -> GenerationResult:
        if output_mode not in {"relative", "observed"}:
            raise ValueError("output_mode must be relative or observed.")
        if not np.isfinite(guidance_scale):
            raise ValueError("guidance_scale must be finite.")
        if guidance_scale != 1.25:
            raise ValueError(
                "guidance_scale does not apply when generating from a bank."
            )
        if not np.isfinite(support_temperature) or support_temperature <= 0:
            raise ValueError("support_temperature must be finite and positive.")
        if use_body_system is not None:
            raise ValueError(
                "use_body_system does not apply when generating from a bank."
            )
        resolved = self._resolved_conditions(phenotype, body_site, conditions)
        selected = self.select_indices(resolved, n_samples=n_samples, seed=seed)
        return self.generate_from_indices(
            selected,
            resolved,
            output_mode=output_mode,
            support_temperature=support_temperature,
            seed=seed,
            library_size=library_size,
            sample_indices=np.arange(int(n_samples), dtype=np.int64),
        )


def load_bank(path: str | Path) -> BatchNeutralBank:
    """Load a batch-neutral logit bank from a local directory."""

    return BatchNeutralBank.from_directory(path)


def download_bank(
    name: str = "core",
    *,
    install_dir: str | Path | None = None,
    url: str | None = None,
    progress: bool | None = None,
) -> BatchNeutralBank:
    """Download a logit bank ZIP and return a bank ready for generation."""

    if name != "core":
        raise ValueError("name must be 'core'.")
    if url is not None and (not isinstance(url, str) or not url.strip()):
        raise ValueError("url must be a nonempty string.")
    cache = Path.home() / ".cache" / "mbbench" / "banks"
    if install_dir is not None:
        target = Path(install_dir)
    elif url is None:
        target = cache / "pregenerated-v1" / name
    else:
        url_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        target = cache / "url" / url_key
    target = target.expanduser().resolve()
    if target.exists():
        return load_bank(target)
    resolved_url = url if url is not None else OFFICIAL_BANK_URL
    expected_hash = (
        OFFICIAL_BANK_SHA256 if resolved_url == OFFICIAL_BANK_URL else None
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        archive = Path(temporary) / "core_bank.zip"
        with urlopen(resolved_url, timeout=120) as response:
            copy_download(
                response, archive, label=archive.name,
                expected_sha256=expected_hash, progress=progress,
            )
        extracted = Path(temporary) / "extracted"
        extracted.mkdir()
        extract_zip(archive, extracted, label="Bank archive")
        source = extracted
        if not (extracted / "bank.yaml").is_file():
            roots = [item for item in extracted.iterdir() if item.is_dir()]
            if len(roots) == 1:
                source = roots[0]
        with load_bank(source):
            pass
        try:
            source.rename(target)
        except OSError:
            if not target.is_dir():
                raise
    return load_bank(target)
