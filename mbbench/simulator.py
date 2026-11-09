from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

from .bundle import ModelBundle
from .common import select_device, set_seed
from .diffusion import (
    DiffusionConfig,
    GaussianDiffusion,
    LatentDenoiser,
    denormalize_latent,
)
from .tokenizer import TokenizerConfig, build_tokenizer


_DONOR_EXACT_MINIMUM = 8


def _tokens(vocabulary: dict) -> list[str]:
    return [
        str(value) for value in vocabulary.get("values", vocabulary.get("tokens", []))
    ]


def _vocabulary_kind(vocabulary: dict) -> str:
    return str(vocabulary.get("type", vocabulary.get("kind", "categorical")))


def _sample_seed(seed: int, stream: int, sample_index: int) -> int:
    return int(
        np.random.SeedSequence([seed, stream, sample_index]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def _select_donor_pool(
    donors: pd.DataFrame,
    conditions: dict[str, list[str]],
    vocabularies: dict[str, dict],
    *,
    minimum_exact: int = _DONOR_EXACT_MINIMUM,
) -> tuple[pd.DataFrame, str]:
    matched = pd.Series(True, index=donors.index)
    matched_any_field = False
    for field, requested in conditions.items():
        if field not in donors:
            continue
        matched_any_field = True
        if _vocabulary_kind(vocabularies[field]) == "multilabel":
            requested_set = set(requested)
            matched &= (
                donors[field]
                .fillna("")
                .astype(str)
                .str.split(r"[|;]")
                .map(
                    lambda values: requested_set
                    == {value.strip() for value in values if value.strip()}
                )
            )
        else:
            matched &= donors[field].astype(str).eq(requested[0])
    exact = donors[matched] if matched_any_field else donors.iloc[0:0]
    if len(exact) >= minimum_exact:
        return exact, "exact_conditions"

    body_site = conditions.get("Body_Site_Coarse", [])
    if body_site and "Body_Site_Coarse" in donors:
        site_pool = donors[donors["Body_Site_Coarse"].astype(str).eq(body_site[0])]
        if len(site_pool):
            return site_pool, "body_site"
    if not len(donors):
        raise ValueError("The model donor pool is empty.")
    return donors, "global"


@dataclass(frozen=True)
class GenerationResult:
    abundance: pd.DataFrame
    counts: pd.DataFrame | None
    metadata: pd.DataFrame
    abundance_logits: np.ndarray
    support_logits: np.ndarray


class Simulator:
    """Generate relative-abundance or observed-count samples from a model bundle."""

    def __init__(self, bundle: ModelBundle, device: torch.device) -> None:
        self.bundle = bundle
        self.device = device
        tokenizer_payload = torch.load(
            bundle.root / "tokenizer.pt", map_location="cpu", weights_only=False
        )
        diffusion_payload = torch.load(
            bundle.root / "diffusion.pt", map_location="cpu", weights_only=False
        )
        tokenizer_config = tokenizer_payload.get(
            "tokenizer_config", diffusion_payload.get("tokenizer_config")
        )
        model_type = tokenizer_payload.get(
            "model_type", diffusion_payload.get("tokenizer_model_type")
        )
        if str(model_type).startswith("protocol_hurdle_"):
            model_type = str(model_type).removeprefix("protocol_hurdle_")
        self.tokenizer = build_tokenizer(
            TokenizerConfig(**tokenizer_config), str(model_type)
        )
        self.tokenizer.load_state_dict(tokenizer_payload["state_dict"])
        self.tokenizer.to(device).eval()
        if tokenizer_payload.get("calibrated_mask_threshold") is not None:
            self.tokenizer.calibrated_mask_threshold = float(
                tokenizer_payload["calibrated_mask_threshold"]
            )
        raw_diffusion_config = diffusion_payload["model_config"]
        if "bio_latent_dim" in raw_diffusion_config:
            raw_diffusion_config = {
                "latent_dim": raw_diffusion_config["bio_latent_dim"],
                "token_dim": raw_diffusion_config["latent_token_dim"],
                "num_latent_tokens": raw_diffusion_config["num_latent_tokens"],
                "num_layers": raw_diffusion_config["num_layers"],
                "num_heads": raw_diffusion_config["num_heads"],
                "mlp_ratio": raw_diffusion_config["mlp_ratio"],
                "num_steps": raw_diffusion_config["diffusion_steps"],
                "beta_start": raw_diffusion_config["beta_start"],
                "beta_end": raw_diffusion_config["beta_end"],
                "condition_hidden_dim": raw_diffusion_config["condition_hidden_dim"],
                "dropout": raw_diffusion_config["dropout"],
                "condition_dropout": raw_diffusion_config["condition_dropout"],
                "primary_condition_dropout": raw_diffusion_config[
                    "primary_condition_dropout"
                ],
                "secondary_condition_dropout": raw_diffusion_config[
                    "secondary_condition_dropout"
                ],
                "prediction_target": raw_diffusion_config["prediction_target"],
                "primary_fields": tuple(
                    raw_diffusion_config["primary_condition_fields"]
                ),
                "secondary_fields": tuple(
                    raw_diffusion_config.get("secondary_condition_fields", [])
                ),
            }
        else:
            raw_diffusion_config = dict(raw_diffusion_config)
            raw_diffusion_config["primary_fields"] = tuple(
                raw_diffusion_config.get("primary_fields", ())
            )
            raw_diffusion_config["secondary_fields"] = tuple(
                raw_diffusion_config.get("secondary_fields", ())
            )
            raw_diffusion_config["primary_interactions"] = tuple(
                tuple(pair)
                for pair in raw_diffusion_config.get("primary_interactions", ())
            )
        self.diffusion_config = DiffusionConfig(**raw_diffusion_config)
        categorical_sizes, multilabel_dims = {}, {}
        for field in (
            self.diffusion_config.primary_fields
            + self.diffusion_config.secondary_fields
        ):
            vocabulary = bundle.vocabularies[field]
            if _vocabulary_kind(vocabulary) == "multilabel":
                multilabel_dims[field] = len(_tokens(vocabulary))
            else:
                categorical_sizes[field] = len(_tokens(vocabulary))
        self.denoiser = LatentDenoiser(
            self.diffusion_config, categorical_sizes, multilabel_dims
        )
        self.denoiser.load_state_dict(diffusion_payload["state_dict"])
        self.denoiser.to(device).eval()
        self.diffusion = GaussianDiffusion(self.diffusion_config)
        latent = diffusion_payload["latent_stats"]
        self.latent_mean = torch.tensor(latent["mean"], dtype=torch.float32)[None]
        self.latent_std = torch.tensor(latent["std"], dtype=torch.float32)[None]
        self.taxa = pd.read_csv(bundle.root / "taxa.csv")["taxon"].astype(str).tolist()

    @classmethod
    def from_directory(cls, path: str | Path, device: str = "auto") -> "Simulator":
        return cls(ModelBundle.load(path), select_device(device))

    def _prepare_conditions(
        self,
        phenotype: str,
        body_site: str,
        supplied: dict[str, object] | None,
        use_body_system: bool,
    ) -> dict[str, list[str]]:
        supplied = dict(supplied or {})
        allowed = set(self.diffusion_config.primary_fields)
        unknown_fields = sorted(set(supplied) - allowed)
        if unknown_fields:
            details = []
            for field in unknown_fields:
                suggestion = get_close_matches(field, sorted(allowed), n=1)
                details.append(
                    f"{field!r}"
                    + (f" (did you mean {suggestion[0]!r}?)" if suggestion else "")
                )
            raise ValueError(
                "Unknown biological condition fields: " + ", ".join(details) + ". "
                "Accepted fields: " + ", ".join(sorted(allowed)) + "."
            )
        if "Phenotype_Tokens" in self.diffusion_config.primary_fields:
            supplied.setdefault("Phenotype_Tokens", phenotype)
        if "Body_Site_Coarse" in self.diffusion_config.primary_fields:
            supplied.setdefault("Body_Site_Coarse", body_site)
        prepared: dict[str, list[str]] = {}
        for field in self.diffusion_config.primary_fields:
            if field == "Body_System_Model":
                if not use_body_system:
                    raise ValueError("The selected model requires --use-body-system.")
                if field in supplied:
                    raise ValueError(
                        "Body_System_Model is derived from Body_Site_Coarse and cannot be set directly."
                    )
                mapping = pd.read_csv(
                    self.bundle.root / "body_site_to_system.csv"
                ).set_index("body_site")["body_system"]
                site = str(supplied.get("Body_Site_Coarse", body_site))
                if site not in mapping:
                    raise ValueError(
                        f"No Body-System mapping exists for body site {site!r}."
                    )
                raw_values = [str(mapping[site])]
            else:
                value = supplied.get(field)
                missing = value is None or (
                    not isinstance(value, (list, tuple, set)) and bool(pd.isna(value))
                )
                if field not in supplied or missing:
                    raise ValueError(
                        f"Generation requires biological condition {field!r}."
                    )
                vocabulary = self.bundle.vocabularies[field]
                if _vocabulary_kind(vocabulary) == "multilabel":
                    raw_values = (
                        [str(item).strip() for item in value if str(item).strip()]
                        if isinstance(value, (list, tuple, set))
                        else [
                            item.strip()
                            for item in str(value).replace(";", "|").split("|")
                            if item.strip()
                        ]
                    )
                else:
                    raw_values = [str(supplied[field])]
            available = self.bundle.condition_values(field)
            if len(raw_values) != len(set(raw_values)):
                raise ValueError(f"Condition {field!r} contains duplicate values.")
            unknown = sorted(set(raw_values) - set(available))
            if unknown:
                details = []
                for value in unknown:
                    suggestion = get_close_matches(value, available, n=1)
                    details.append(
                        f"{value!r}"
                        + (f" (did you mean {suggestion[0]!r}?)" if suggestion else "")
                    )
                raise ValueError(
                    f"Unknown {field} values: {', '.join(details)}. "
                    f"Available values: {', '.join(available)}"
                )
            prepared[field] = [value for value in available if value in raw_values]
        return prepared

    def _condition_status(self, conditions: dict[str, list[str]]) -> str:
        combinations_path = self.bundle.root / "condition_combinations.csv"
        if not combinations_path.is_file():
            return "unknown_not_auditable"
        combinations = pd.read_csv(combinations_path).astype(str)
        matched = pd.Series(True, index=combinations.index)
        for field in self.diffusion_config.primary_fields:
            if field not in combinations:
                return "unknown_not_auditable"
            if _vocabulary_kind(self.bundle.vocabularies[field]) == "multilabel":
                requested = set(conditions[field])
                matched &= (
                    combinations[field]
                    .fillna("")
                    .astype(str)
                    .map(
                        lambda value: {
                            token.strip()
                            for token in value.replace(";", "|").split("|")
                            if token.strip()
                        }
                        == requested
                    )
                )
            else:
                matched &= combinations[field].astype(str).eq(conditions[field][0])
        observed = bool(matched.any())
        if not observed:
            warnings.warn(
                "This biological-condition combination was not observed during training; generation will continue as combinatorial extrapolation.",
                stacklevel=2,
            )
            return "unobserved_combination"
        return "observed"

    def _conditions(self, conditions: dict[str, list[str]], n: int):
        categorical: dict[str, torch.Tensor] = {}
        multilabel: dict[str, torch.Tensor] = {}
        for field in self.diffusion_config.primary_fields:
            vocabulary = self.bundle.vocabularies[field]
            values = _tokens(vocabulary)
            if _vocabulary_kind(vocabulary) == "multilabel":
                encoded = torch.zeros(n, len(values))
                for value in conditions[field]:
                    encoded[:, values.index(value)] = 1.0
                multilabel[field] = encoded
            else:
                categorical[field] = torch.full(
                    (n,), values.index(conditions[field][0]), dtype=torch.long
                )
        return categorical, multilabel

    def _observed_context(
        self,
        conditions: dict[str, list[str]],
        sample_indices: np.ndarray,
        seed: int,
        library_size: int | None,
    ):
        donor_path = self.bundle.root / "donors.csv"
        if not donor_path.is_file():
            raise ValueError(
                "This model bundle does not contain the train-only donor pool required for observed output."
            )
        donors = pd.read_csv(donor_path)
        pool, match_level = _select_donor_pool(
            donors, conditions, self.bundle.vocabularies
        )
        selected = [
            int(
                np.random.default_rng(
                    _sample_seed(seed, 31, int(sample_index))
                ).integers(len(pool))
            )
            for sample_index in sample_indices
        ]
        sampled = pool.iloc[selected].reset_index(drop=True)
        n = len(sample_indices)
        categorical: dict[str, torch.Tensor] = {}
        for field in [
            *self.tokenizer.config.technical_field_sizes,
            self.tokenizer.config.project_field,
        ]:
            if field not in sampled or field not in self.bundle.vocabularies:
                continue
            values = _tokens(self.bundle.vocabularies[field])
            unknown = next(
                (
                    index
                    for index, value in enumerate(values)
                    if value in {"<UNK>", "__UNK__"}
                ),
                0,
            )
            lookup = {value: index for index, value in enumerate(values)}
            categorical[field] = torch.tensor(
                [lookup.get(str(value), unknown) for value in sampled[field]],
                dtype=torch.long,
                device=self.device,
            )
        if library_size is not None and library_size <= 0:
            raise ValueError("library_size must be positive.")
        raw_depth = (
            np.full(n, library_size, dtype=np.int64)
            if library_size is not None
            else pd.to_numeric(sampled["raw_total_reads"])
            .round()
            .astype(np.int64)
            .to_numpy()
        )
        stats = self.bundle.config.get("read_depth_stats", {})
        median, iqr = float(stats.get("median", 0.0)), max(
            float(stats.get("iqr", 1.0)), 1e-6
        )
        standardized = torch.tensor(
            (np.log1p(raw_depth) - median) / iqr,
            dtype=torch.float32,
            device=self.device,
        )
        tech = self.tokenizer.encode_technical(categorical, n, self.device)
        project = self.tokenizer.project_latent(
            categorical.get(self.tokenizer.config.project_field), n, self.device
        )
        return sampled, tech, project, standardized, raw_depth, match_level

    @torch.no_grad()
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
        conditions: dict[str, object] | None = None,
    ) -> GenerationResult:
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
            raise TypeError("n_samples must be a positive integer.")
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        return self._generate_indexed(
            phenotype=phenotype,
            body_site=body_site,
            sample_indices=np.arange(int(n_samples), dtype=np.int64),
            output_mode=output_mode,
            guidance_scale=guidance_scale,
            support_temperature=support_temperature,
            seed=seed,
            library_size=library_size,
            use_body_system=use_body_system,
            conditions=conditions,
        )

    @torch.no_grad()
    def _generate_indexed(
        self,
        *,
        phenotype: str,
        body_site: str,
        sample_indices: np.ndarray,
        output_mode: str,
        guidance_scale: float,
        support_temperature: float,
        seed: int,
        library_size: int | None,
        use_body_system: bool | None,
        conditions: dict[str, object] | None,
    ) -> GenerationResult:
        sample_indices = np.asarray(sample_indices)
        if (
            sample_indices.ndim != 1
            or sample_indices.dtype.kind not in "iu"
            or len(np.unique(sample_indices)) != len(sample_indices)
            or bool((sample_indices < 0).any())
        ):
            raise ValueError(
                "sample_indices must contain unique non-negative integers."
            )
        if not len(sample_indices):
            raise ValueError("sample_indices cannot be empty.")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer.")
        n_samples = len(sample_indices)
        if output_mode not in {"relative", "observed"}:
            raise ValueError("output_mode must be relative or observed.")
        if output_mode not in self.bundle.config["supported_output_modes"]:
            raise ValueError(f"Selected model does not support {output_mode!r} output.")
        if (
            not np.isfinite(guidance_scale)
            or guidance_scale < 0
            or not np.isfinite(support_temperature)
            or support_temperature <= 0
        ):
            raise ValueError(
                "guidance_scale must be finite and support_temperature must be finite and positive."
            )
        if use_body_system is None:
            use_body_system = self.bundle.supports_body_system
        if use_body_system != self.bundle.supports_body_system:
            expected = "with" if self.bundle.supports_body_system else "without"
            raise ValueError(
                f"Selected model was trained {expected} Body-System conditioning."
            )
        prepared_conditions = self._prepare_conditions(
            phenotype, body_site, conditions, use_body_system
        )
        status = self._condition_status(prepared_conditions)
        set_seed(seed)
        categorical, multilabel = self._conditions(prepared_conditions, n_samples)
        normalized = self.diffusion.sample(
            self.denoiser,
            categorical,
            multilabel,
            n_samples,
            guidance_scale,
            sample_seeds=[
                _sample_seed(int(seed), 11, int(index)) for index in sample_indices
            ],
        )
        bio_latent = denormalize_latent(normalized, self.latent_mean, self.latent_std)
        support_uniforms = torch.tensor(
            np.stack(
                [
                    np.random.default_rng(
                        _sample_seed(int(seed), 21, int(index))
                    ).random(len(self.taxa))
                    for index in sample_indices
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        metadata = pd.DataFrame(
            {
                "sample_id": [
                    f"mbbench_{seed}_{int(index) + 1:06d}" for index in sample_indices
                ]
            }
        )
        for field, values in prepared_conditions.items():
            metadata[field] = "|".join(values)
        metadata["condition_status"] = status
        counts = None
        if output_mode == "relative":
            decoded = self.tokenizer.decode(
                bio_latent,
                support_mode="temperature_bernoulli",
                support_temperature=support_temperature,
                support_uniforms=support_uniforms,
            )
        else:
            donor, tech, project, depth, raw_depth, donor_match_level = (
                self._observed_context(
                    prepared_conditions, sample_indices, int(seed), library_size
                )
            )
            decoded = self.tokenizer.decode(
                bio_latent,
                tech,
                project,
                depth,
                support_mode="temperature_bernoulli",
                support_temperature=support_temperature,
                support_uniforms=support_uniforms,
            )
            abundance_array = decoded["decoded_abundance"].cpu().numpy()
            counts_array = np.stack(
                [
                    np.random.default_rng(
                        _sample_seed(int(seed), 41, int(sample_index))
                    ).multinomial(int(depth_value), probabilities)
                    for sample_index, depth_value, probabilities in zip(
                        sample_indices, raw_depth, abundance_array, strict=True
                    )
                ]
            )
            counts = pd.DataFrame(counts_array, columns=self.taxa)
            counts.insert(0, "sample_id", metadata["sample_id"])
            duplicate = {"sample_id", "Sample", "raw_total_reads", *prepared_conditions}
            donor_columns = [column for column in donor if column not in duplicate]
            metadata = pd.concat(
                [metadata, donor[donor_columns].reset_index(drop=True)], axis=1
            )
            metadata["raw_total_reads"] = raw_depth
            metadata["donor_match_level"] = donor_match_level
        abundance = pd.DataFrame(
            decoded["decoded_abundance"].cpu().numpy(), columns=self.taxa
        )
        abundance.insert(0, "sample_id", metadata["sample_id"])
        return GenerationResult(
            abundance=abundance,
            counts=counts,
            metadata=metadata,
            abundance_logits=decoded["abundance_logits"].cpu().numpy(),
            support_logits=decoded["mask_logits"].cpu().numpy(),
        )
