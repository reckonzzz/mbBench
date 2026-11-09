from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn

from .losses import masked_softmax


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * gradient, None


def gradient_reverse(value: torch.Tensor, coefficient: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(value, coefficient)


def _mlp(
    input_dim: int, hidden_dim: int, output_dim: int, dropout: float
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass(frozen=True)
class TokenizerConfig:
    input_dim: int
    bio_latent_dim: int = 128
    tech_latent_dim: int = 32
    project_latent_dim: int = 16
    encoder_hidden_dim: int = 1024
    decoder_hidden_dim: int = 1024
    condition_embedding_dim: int = 32
    dropout: float = 0.1
    mask_threshold: float = 0.40
    eps: float = 1e-6
    logvar_min: float = -12.0
    logvar_max: float = 12.0
    use_mask_in_encoder: bool = True
    technical_field_sizes: dict[str, int] = field(default_factory=dict)
    project_field: str = "Project_Group"
    project_field_size: int = 0
    bio_categorical_field_sizes: dict[str, int] = field(default_factory=dict)
    bio_multilabel_field_dims: dict[str, int] = field(default_factory=dict)
    grl_lambda: float = 1.0
    technical_center_weights: dict[str, tuple[float, ...]] = field(default_factory=dict)
    project_center_weights: tuple[float, ...] = ()


class HurdleAE(nn.Module):
    """Factorized hurdle autoencoder."""

    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__()
        self.config = config
        encoder_input = config.input_dim * (2 if config.use_mask_in_encoder else 1)
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input, config.encoder_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim),
            nn.SiLU(),
        )
        self.bio_latent_head = nn.Linear(
            config.encoder_hidden_dim, config.bio_latent_dim
        )
        self.technical_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(size, config.condition_embedding_dim)
                for name, size in config.technical_field_sizes.items()
            }
        )
        self.technical_projections = nn.ModuleDict(
            {
                name: nn.Linear(
                    config.condition_embedding_dim, config.tech_latent_dim, bias=False
                )
                for name in config.technical_field_sizes
            }
        )
        self._technical_weight_buffers: dict[str, str] = {}
        for index, (name, size) in enumerate(config.technical_field_sizes.items()):
            weight = torch.tensor(
                config.technical_center_weights.get(name, ()), dtype=torch.float32
            )
            if weight.numel() != size or float(weight.sum()) <= 0:
                weight = torch.ones(size)
            buffer_name = f"technical_center_weight_{index}"
            self.register_buffer(buffer_name, weight / weight.sum())
            self._technical_weight_buffers[name] = buffer_name
        self.project_embedding = (
            nn.Embedding(config.project_field_size, config.project_latent_dim)
            if config.project_field_size > 0
            else None
        )
        project_weights = torch.tensor(
            config.project_center_weights, dtype=torch.float32
        )
        if (
            project_weights.numel() != config.project_field_size
            or float(project_weights.sum()) <= 0
        ):
            project_weights = torch.ones(max(config.project_field_size, 1))
        self.register_buffer(
            "project_center_weight", project_weights / project_weights.sum()
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(config.bio_latent_dim, config.decoder_hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.SiLU(),
        )
        self.mask_head = nn.Linear(config.decoder_hidden_dim, config.input_dim)
        self.abundance_head = nn.Linear(config.decoder_hidden_dim, config.input_dim)
        self.tech_mask_offset = nn.Linear(
            config.tech_latent_dim, config.input_dim, bias=False
        )
        self.tech_abundance_offset = nn.Linear(
            config.tech_latent_dim, config.input_dim, bias=False
        )
        self.project_mask_offset = nn.Linear(
            config.project_latent_dim, config.input_dim, bias=False
        )
        self.project_abundance_offset = nn.Linear(
            config.project_latent_dim, config.input_dim, bias=False
        )
        self.depth_mask_offset = nn.Linear(1, config.input_dim, bias=False)

        probe_hidden = max(32, config.decoder_hidden_dim // 4)
        self.tech_heads = nn.ModuleDict(
            {
                field: _mlp(config.tech_latent_dim, probe_hidden, size, config.dropout)
                for field, size in config.technical_field_sizes.items()
            }
        )
        if config.project_field_size > 0:
            self.tech_heads[config.project_field] = _mlp(
                config.project_latent_dim,
                probe_hidden,
                config.project_field_size,
                config.dropout,
            )
        nuisance_sizes = dict(config.technical_field_sizes)
        if config.project_field_size > 0:
            nuisance_sizes[config.project_field] = config.project_field_size
        self.bio_nuisance_heads = nn.ModuleDict(
            {
                field: _mlp(config.bio_latent_dim, probe_hidden, size, config.dropout)
                for field, size in nuisance_sizes.items()
            }
        )
        self.bio_categorical_heads = nn.ModuleDict(
            {
                field: _mlp(config.bio_latent_dim, probe_hidden, size, config.dropout)
                for field, size in config.bio_categorical_field_sizes.items()
            }
        )
        self.bio_multilabel_heads = nn.ModuleDict(
            {
                field: _mlp(config.bio_latent_dim, probe_hidden, size, config.dropout)
                for field, size in config.bio_multilabel_field_dims.items()
            }
        )

    @property
    def model_type(self) -> str:
        return "ae"

    def _encoder_input(
        self, abundance: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.cat([abundance, mask], dim=-1)
            if self.config.use_mask_in_encoder
            else abundance
        )

    def encode(
        self, abundance: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        latent = self.bio_latent_head(
            self.encoder(self._encoder_input(abundance, mask))
        )
        return {
            "bio_latent": latent,
            "bio_representation_latent": latent,
            "latent": latent,
        }

    def encode_to_latent(
        self, abundance: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return self.encode(abundance, mask)["bio_latent"]

    def encode_to_mu(self, abundance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encode_to_latent(abundance, mask)

    def encode_technical(
        self,
        categorical: dict[str, torch.Tensor],
        batch_size: int,
        device: torch.device,
        validity: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        result = torch.zeros(batch_size, self.config.tech_latent_dim, device=device)
        validity = validity or {}
        for field, embedding in self.technical_embeddings.items():
            ids = categorical.get(field)
            if ids is None:
                continue
            weights = getattr(self, self._technical_weight_buffers[field]).to(
                embedding.weight
            )
            centered = embedding.weight - (embedding.weight * weights[:, None]).sum(
                dim=0, keepdim=True
            )
            contribution = self.technical_projections[field](
                nn.functional.embedding(ids, centered)
            )
            if field in validity:
                contribution = contribution * validity[field].to(contribution)[:, None]
            result = result + contribution
        return result

    def centered_project_embeddings(self) -> torch.Tensor:
        if self.project_embedding is None:
            return torch.zeros(
                1, self.config.project_latent_dim, device=next(self.parameters()).device
            )
        weights = self.project_embedding.weight
        center = self.project_center_weight[: weights.shape[0]].to(weights)
        center = center / center.sum().clamp_min(1e-8)
        return weights - (weights * center[:, None]).sum(dim=0, keepdim=True)

    def project_latent(
        self,
        project_ids: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        validity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if project_ids is None or self.project_embedding is None:
            return torch.zeros(
                batch_size, self.config.project_latent_dim, device=device
            )
        result = nn.functional.embedding(
            project_ids, self.centered_project_embeddings()
        )
        return result if validity is None else result * validity.to(result)[:, None]

    def decode(
        self,
        bio_latent: torch.Tensor,
        tech_latent: torch.Tensor | None = None,
        project_latent: torch.Tensor | None = None,
        read_depth: torch.Tensor | None = None,
        teacher_forced_mask: torch.Tensor | None = None,
        threshold: float | None = None,
        support_mode: str = "deterministic",
        support_temperature: float = 1.0,
        support_generator: torch.Generator | None = None,
        support_uniforms: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = bio_latent.shape[0]
        tech = (
            tech_latent
            if tech_latent is not None
            else bio_latent.new_zeros(batch_size, self.config.tech_latent_dim)
        )
        project = (
            project_latent
            if project_latent is not None
            else bio_latent.new_zeros(batch_size, self.config.project_latent_dim)
        )
        hidden = self.decoder_input(bio_latent)
        abundance_logits = (
            self.abundance_head(hidden)
            + self.tech_abundance_offset(tech)
            + self.project_abundance_offset(project)
        )
        mask_logits = (
            self.mask_head(hidden)
            + self.tech_mask_offset(tech)
            + self.project_mask_offset(project)
        )
        if read_depth is not None:
            mask_logits = mask_logits + self.depth_mask_offset(
                read_depth.reshape(batch_size, 1).to(bio_latent)
            )
        mask_probability = torch.sigmoid(mask_logits)
        cutoff = (
            float(
                getattr(self, "calibrated_mask_threshold", self.config.mask_threshold)
            )
            if threshold is None
            else float(threshold)
        )
        if support_mode == "deterministic":
            support_probability = mask_probability
            decoded_mask = (support_probability >= cutoff).to(bio_latent.dtype)
        elif support_mode == "bernoulli":
            support_probability = mask_probability
            if support_uniforms is None:
                decoded_mask = torch.bernoulli(
                    support_probability, generator=support_generator
                ).to(bio_latent.dtype)
            else:
                if support_uniforms.shape != support_probability.shape:
                    raise ValueError(
                        "support_uniforms must match support probability shape."
                    )
                decoded_mask = (
                    support_uniforms.to(support_probability) < support_probability
                ).to(bio_latent.dtype)
        elif support_mode == "temperature_bernoulli":
            if not math.isfinite(support_temperature) or support_temperature <= 0:
                raise ValueError("support_temperature must be finite and positive.")
            bounded = min(max(cutoff, 1e-6), 1.0 - 1e-6)
            threshold_logit = math.log(bounded / (1.0 - bounded))
            support_probability = torch.sigmoid(
                (mask_logits - threshold_logit) / support_temperature
            )
            if support_uniforms is None:
                decoded_mask = torch.bernoulli(
                    support_probability, generator=support_generator
                ).to(bio_latent.dtype)
            else:
                if support_uniforms.shape != support_probability.shape:
                    raise ValueError(
                        "support_uniforms must match support probability shape."
                    )
                decoded_mask = (
                    support_uniforms.to(support_probability) < support_probability
                ).to(bio_latent.dtype)
        else:
            raise ValueError(f"Unsupported support mode: {support_mode}")
        fallback = torch.zeros_like(decoded_mask)
        fallback.scatter_(1, support_probability.argmax(dim=-1, keepdim=True), 1.0)
        decoded_mask = torch.where(
            decoded_mask.sum(dim=-1, keepdim=True) > 0, decoded_mask, fallback
        )
        decoded_abundance = masked_softmax(abundance_logits, decoded_mask)
        reconstruction_mask = (
            decoded_mask if teacher_forced_mask is None else teacher_forced_mask
        )
        reconstruction = (
            decoded_abundance
            if teacher_forced_mask is None
            else masked_softmax(abundance_logits, teacher_forced_mask)
        )
        return {
            "mask_logits": mask_logits,
            "mask_prob": mask_probability,
            "support_probability": support_probability,
            "abundance_logits": abundance_logits,
            "decoded_mask": decoded_mask,
            "recon_mask": reconstruction_mask,
            "decoded_abundance": decoded_abundance,
            "recon_abundance": reconstruction,
        }

    def reconstruct(
        self,
        abundance: torch.Tensor,
        mask: torch.Tensor,
        categorical_conditions: dict[str, torch.Tensor] | None = None,
        condition_validity: dict[str, torch.Tensor] | None = None,
        read_depth: torch.Tensor | None = None,
        teacher_force_mask: bool = True,
        deterministic_posterior: bool = False,
        grl_lambda_override: float | None = None,
    ) -> dict[str, torch.Tensor]:
        categorical = categorical_conditions or {}
        validity = condition_validity or {}
        if deterministic_posterior and isinstance(self, HurdleVAE):
            encoded = self.encode_distribution(abundance, mask)
            encoded.update(
                {
                    "bio_latent": encoded["bio_mu"],
                    "bio_representation_latent": encoded["bio_mu"],
                }
            )
        else:
            encoded = self.encode(abundance, mask)
        bio = encoded["bio_latent"]
        representation = encoded.get("bio_representation_latent", bio)
        tech = self.encode_technical(
            categorical, len(abundance), abundance.device, validity
        )
        project = self.project_latent(
            categorical.get(self.config.project_field),
            len(abundance),
            abundance.device,
            validity.get(self.config.project_field),
        )
        output = self.decode(
            bio, tech, project, read_depth, mask if teacher_force_mask else None
        )
        coefficient = (
            self.config.grl_lambda
            if grl_lambda_override is None
            else float(grl_lambda_override)
        )
        reversed_bio = (
            representation
            if coefficient == 0
            else gradient_reverse(representation, coefficient)
        )
        output.update(encoded)
        output.update(
            {
                "tech_latent": tech,
                "project_latent": project,
                "batch_latent": torch.cat([tech, project], dim=-1),
                "batch_logits": {
                    field: head(tech if field != self.config.project_field else project)
                    for field, head in self.tech_heads.items()
                },
                "bio_batch_logits": {
                    field: head(reversed_bio)
                    for field, head in self.bio_nuisance_heads.items()
                },
                "bio_supervision_logits": {
                    **{
                        field: head(representation)
                        for field, head in self.bio_categorical_heads.items()
                    },
                    **{
                        field: head(representation)
                        for field, head in self.bio_multilabel_heads.items()
                    },
                },
                "project_shrinkage_loss": self.centered_project_embeddings()
                .square()
                .sum(dim=-1)
                .mean(),
            }
        )
        return output

    forward = reconstruct


class HurdleVAE(HurdleAE):
    def __init__(self, config: TokenizerConfig) -> None:
        super().__init__(config)
        self.bio_mu_head = nn.Linear(config.encoder_hidden_dim, config.bio_latent_dim)
        self.bio_logvar_head = nn.Linear(
            config.encoder_hidden_dim, config.bio_latent_dim
        )
        del self.bio_latent_head

    @property
    def model_type(self) -> str:
        return "vae"

    def encode_distribution(
        self, abundance: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(self._encoder_input(abundance, mask))
        mu = self.bio_mu_head(hidden)
        logvar = self.bio_logvar_head(hidden).clamp(
            self.config.logvar_min, self.config.logvar_max
        )
        return {"bio_mu": mu, "bio_logvar": logvar, "mu": mu, "logvar": logvar}

    def encode(
        self, abundance: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        distribution = self.encode_distribution(abundance, mask)
        latent = distribution["bio_mu"] + torch.randn_like(
            distribution["bio_mu"]
        ) * torch.exp(0.5 * distribution["bio_logvar"])
        return {
            **distribution,
            "bio_latent": latent,
            "bio_representation_latent": distribution["bio_mu"],
            "latent": latent,
        }

    def encode_to_mu(self, abundance: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encode_distribution(abundance, mask)["bio_mu"]


def build_tokenizer(config: TokenizerConfig, model_type: str) -> HurdleAE:
    if model_type == "ae":
        return HurdleAE(config)
    if model_type == "vae":
        return HurdleVAE(config)
    raise ValueError(f"Unsupported tokenizer model type: {model_type}")
