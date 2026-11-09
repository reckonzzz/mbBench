from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class DiffusionConfig:
    latent_dim: int = 128
    token_dim: int = 64
    num_latent_tokens: int = 12
    num_layers: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    num_steps: int = 500
    beta_start: float = 1e-4
    beta_end: float = 0.02
    condition_hidden_dim: int = 256
    dropout: float = 0.1
    condition_dropout: float = 0.05
    primary_condition_dropout: float = 0.10
    secondary_condition_dropout: float = 0.25
    prediction_target: str = "v"
    primary_fields: tuple[str, ...] = ("Phenotype_Tokens", "Body_Site_Coarse")
    secondary_fields: tuple[str, ...] = ()
    primary_interactions: tuple[tuple[str, str], ...] = ()
    secondary_condition_mode: str = "gated"


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.embedding_dim // 2
        exponent = -math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=timesteps.device) * exponent)
        arguments = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)
        if self.embedding_dim % 2:
            embedding = nn.functional.pad(embedding, (0, 1))
        return self.projection(embedding)


class ConditionTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        categorical_vocabulary_sizes: dict[str, int],
        multilabel_dims: dict[str, int],
        primary_fields: tuple[str, ...],
        secondary_fields: tuple[str, ...],
        primary_interactions: tuple[tuple[str, str], ...],
        token_dim: int,
        hidden_dim: int,
        dropout: float,
        secondary_condition_mode: str = "gated",
    ) -> None:
        super().__init__()
        if secondary_condition_mode not in {"gated", "direct"}:
            raise ValueError("secondary_condition_mode must be gated or direct.")
        self.categorical_fields = list(categorical_vocabulary_sizes)
        self.multilabel_fields = list(multilabel_dims)
        self.multilabel_dims = dict(multilabel_dims)
        self.primary_fields = list(primary_fields)
        self.secondary_fields = list(secondary_fields)
        self.primary_interactions = list(primary_interactions)
        self.secondary_condition_mode = secondary_condition_mode
        self.token_dim = token_dim
        self.categorical_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(size + 2, token_dim)
                for field, size in categorical_vocabulary_sizes.items()
            }
        )
        self.cfg_null_indices = {
            field: size for field, size in categorical_vocabulary_sizes.items()
        }
        self.masked_indices = {
            field: size + 1 for field, size in categorical_vocabulary_sizes.items()
        }
        self.multilabel_projections = nn.ModuleDict(
            {
                field: nn.Sequential(
                    nn.Linear(size, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, token_dim),
                )
                for field, size in multilabel_dims.items()
            }
        )
        fields = [*self.categorical_fields, *self.multilabel_fields]
        self.field_bias = nn.ParameterDict(
            {field: nn.Parameter(torch.zeros(token_dim)) for field in fields}
        )
        self.multilabel_cfg_null = nn.ParameterDict(
            {
                field: nn.Parameter(torch.zeros(token_dim))
                for field in self.multilabel_fields
            }
        )
        self.role_embeddings = nn.ParameterDict(
            {
                role: nn.Parameter(torch.zeros(token_dim))
                for role in ("time", "primary", "interaction", "secondary")
            }
        )
        self.interaction_projections = nn.ModuleDict(
            {
                self._interaction_key(pair): nn.Sequential(
                    nn.Linear(token_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, token_dim),
                )
                for pair in primary_interactions
            }
        )
        self.secondary_gate = nn.Sequential(
            nn.Linear(token_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.Sigmoid(),
        )

    @staticmethod
    def _interaction_key(fields: tuple[str, str]) -> str:
        return "__interaction__".join(fields)

    def _encode_field(
        self,
        field: str,
        categorical: dict[str, torch.Tensor],
        multilabel: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        if field in self.categorical_embeddings:
            values = categorical[field]
            embedded = (
                self.categorical_embeddings[field](values) + self.field_bias[field]
            )
            return torch.where(
                values.eq(self.masked_indices[field])[:, None],
                torch.zeros_like(embedded),
                embedded,
            )
        if field in self.multilabel_projections:
            values = multilabel[field].float()
            cfg_null = (values == -1).all(dim=-1, keepdim=True)
            masked = (values == -2).all(dim=-1, keepdim=True)
            projected = self.multilabel_projections[field](values.clamp_min(0.0))
            null = self.multilabel_cfg_null[field][None].expand_as(projected)
            encoded = torch.where(cfg_null, null, projected) + self.field_bias[field]
            return torch.where(masked, torch.zeros_like(encoded), encoded)
        return None

    def forward(
        self,
        categorical: dict[str, torch.Tensor],
        multilabel: dict[str, torch.Tensor],
        timestep_token: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(timestep_token)
        primary: list[torch.Tensor] = []
        secondary: list[torch.Tensor] = []
        encoded_primary: dict[str, torch.Tensor] = {}
        for field in self.primary_fields:
            token = self._encode_field(field, categorical, multilabel)
            if token is not None:
                encoded_primary[field] = token
                primary.append(token + self.role_embeddings["primary"])
        for pair in self.primary_interactions:
            interaction = self.interaction_projections[self._interaction_key(pair)](
                torch.cat([encoded_primary[pair[0]], encoded_primary[pair[1]]], dim=-1)
            )
            primary.append(interaction + self.role_embeddings["interaction"])
        primary_summary = (
            torch.stack(primary, dim=1).mean(dim=1)
            if primary
            else timestep_token.new_zeros(batch_size, self.token_dim)
        )
        for field in self.secondary_fields:
            token = self._encode_field(field, categorical, multilabel)
            if token is None:
                continue
            token = token + self.role_embeddings["secondary"]
            if self.secondary_condition_mode == "gated":
                token = token * self.secondary_gate(
                    torch.cat([token, primary_summary, timestep_token], dim=-1)
                )
            secondary.append(token)
        time = (timestep_token + self.role_embeddings["time"])[:, None]
        primary_stack = (
            torch.stack(primary, dim=1)
            if primary
            else timestep_token.new_zeros(batch_size, 0, self.token_dim)
        )
        secondary_stack = (
            torch.stack(secondary, dim=1)
            if secondary
            else timestep_token.new_zeros(batch_size, 0, self.token_dim)
        )
        return torch.cat([time, primary_stack, secondary_stack], dim=1)


class TransformerMLP(nn.Module):
    def __init__(self, token_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(token_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, token_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class LatentTransformerBlock(nn.Module):
    def __init__(
        self, token_dim: int, num_heads: int, mlp_ratio: float, dropout: float
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(token_dim)
        self.self_attn = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(token_dim)
        self.cross_attn = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(token_dim)
        self.mlp = TransformerMLP(token_dim, mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(latent)
        attended, _ = self.self_attn(
            normalized, normalized, normalized, need_weights=False
        )
        latent = latent + self.dropout(attended)
        normalized = self.cross_norm(latent)
        attended, _ = self.cross_attn(normalized, context, context, need_weights=False)
        latent = latent + self.dropout(attended)
        return latent + self.dropout(self.mlp(self.mlp_norm(latent)))


class LatentDenoiser(nn.Module):
    """Transformer denoiser for the standardized biological latent."""

    def __init__(
        self,
        config: DiffusionConfig,
        categorical_vocabulary_sizes: dict[str, int],
        multilabel_dims: dict[str, int],
    ) -> None:
        super().__init__()
        self.config = config
        self.categorical_fields = list(categorical_vocabulary_sizes)
        self.multilabel_fields = list(multilabel_dims)
        self.multilabel_dims = dict(multilabel_dims)
        self.primary_fields = list(config.primary_fields)
        self.secondary_fields = list(config.secondary_fields)
        self.condition_encoder = ConditionTokenEncoder(
            categorical_vocabulary_sizes=categorical_vocabulary_sizes,
            multilabel_dims=multilabel_dims,
            primary_fields=config.primary_fields,
            secondary_fields=config.secondary_fields,
            primary_interactions=config.primary_interactions,
            token_dim=config.token_dim,
            hidden_dim=config.condition_hidden_dim,
            dropout=config.dropout,
            secondary_condition_mode=config.secondary_condition_mode,
        )
        self.time_encoder = TimestepEmbedding(config.token_dim)
        self.input_projection = nn.Linear(
            config.latent_dim, config.num_latent_tokens * config.token_dim
        )
        self.output_projection = nn.Linear(
            config.num_latent_tokens * config.token_dim, config.latent_dim
        )
        self.latent_positions = nn.Parameter(
            torch.zeros(1, config.num_latent_tokens, config.token_dim)
        )
        self.blocks = nn.ModuleList(
            [
                LatentTransformerBlock(
                    config.token_dim, config.num_heads, config.mlp_ratio, config.dropout
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.token_dim)

    def _scoped_conditions(
        self,
        categorical: dict[str, torch.Tensor],
        multilabel: dict[str, torch.Tensor],
        latent: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch = len(latent)
        scoped_categorical = {
            field: categorical.get(
                field,
                torch.full(
                    (batch,),
                    self.condition_encoder.cfg_null_indices[field],
                    dtype=torch.long,
                    device=latent.device,
                ),
            )
            for field in self.categorical_fields
        }
        scoped_multilabel = {
            field: multilabel.get(
                field,
                torch.full(
                    (batch, self.multilabel_dims[field]),
                    -1.0,
                    dtype=latent.dtype,
                    device=latent.device,
                ),
            )
            for field in self.multilabel_fields
        }
        return scoped_categorical, scoped_multilabel

    @staticmethod
    def _drop(
        tensor: torch.Tensor, mask: torch.Tensor, null_index: int = 0
    ) -> torch.Tensor:
        if tensor.ndim == 1:
            return torch.where(mask, torch.full_like(tensor, null_index), tensor)
        return torch.where(mask[:, None], torch.full_like(tensor, -1), tensor)

    def apply_condition_dropout(
        self,
        categorical: dict[str, torch.Tensor],
        multilabel: dict[str, torch.Tensor],
        latent: torch.Tensor,
        *,
        enable_dropout: bool,
        force_drop_primary: bool = False,
        force_drop_secondary: bool = False,
        force_drop_fields: Collection[str] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        forced = set(force_drop_fields or ())
        declared = set(self.categorical_fields) | set(self.multilabel_fields)
        if forced - declared:
            raise ValueError(
                f"Cannot drop undeclared fields: {sorted(forced - declared)}"
            )
        if (
            not enable_dropout
            and not force_drop_primary
            and not force_drop_secondary
            and not forced
        ):
            return categorical, multilabel
        batch = len(latent)
        full = (
            torch.rand(batch, device=latent.device) < self.config.condition_dropout
            if enable_dropout and self.config.condition_dropout > 0
            else torch.zeros(batch, dtype=torch.bool, device=latent.device)
        )
        masks: dict[str, torch.Tensor] = {}
        for field in declared:
            probability = (
                self.config.primary_condition_dropout
                if field in self.primary_fields
                else self.config.secondary_condition_dropout
            )
            mask = (
                torch.rand(batch, device=latent.device) < probability
                if enable_dropout and probability > 0
                else torch.zeros(batch, dtype=torch.bool, device=latent.device)
            )
            if (
                (force_drop_primary and field in self.primary_fields)
                or (force_drop_secondary and field in self.secondary_fields)
                or field in forced
            ):
                mask = torch.ones(batch, dtype=torch.bool, device=latent.device)
            masks[field] = full | mask
        return (
            {
                field: self._drop(
                    value, masks[field], self.condition_encoder.cfg_null_indices[field]
                )
                for field, value in categorical.items()
            },
            {
                field: self._drop(value, masks[field])
                for field, value in multilabel.items()
            },
        )

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        categorical_conditions: dict[str, torch.Tensor],
        multilabel_conditions: dict[str, torch.Tensor],
        *,
        drop_conditioning: bool = True,
        force_drop_primary: bool = False,
        force_drop_secondary: bool = False,
        force_drop_fields: Collection[str] | None = None,
    ) -> torch.Tensor:
        categorical, multilabel = self._scoped_conditions(
            categorical_conditions, multilabel_conditions, noisy_latent
        )
        categorical, multilabel = self.apply_condition_dropout(
            categorical,
            multilabel,
            noisy_latent,
            enable_dropout=self.training and drop_conditioning,
            force_drop_primary=force_drop_primary,
            force_drop_secondary=force_drop_secondary,
            force_drop_fields=force_drop_fields,
        )
        time = self.time_encoder(timesteps)
        context = self.condition_encoder(categorical, multilabel, time)
        latent = self.input_projection(noisy_latent).view(
            len(noisy_latent), self.config.num_latent_tokens, self.config.token_dim
        )
        latent = latent + self.latent_positions
        for block in self.blocks:
            latent = block(latent, context)
        return self.output_projection(
            self.final_norm(latent).reshape(len(noisy_latent), -1)
        )


class GaussianDiffusion:
    def __init__(self, config: DiffusionConfig) -> None:
        if config.prediction_target not in {"v", "eps"}:
            raise ValueError(
                f"Unsupported prediction target: {config.prediction_target}"
            )
        self.config = config
        self.betas = torch.linspace(
            config.beta_start, config.beta_end, config.num_steps
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.config.num_steps, (batch_size,), device=device)

    def _alpha_sigma(
        self, timesteps: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_bar = self.alpha_bars.to(device)[timesteps, None]
        return alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt()

    def q_sample(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(latent) if noise is None else noise
        alpha, sigma = self._alpha_sigma(timesteps, latent.device)
        return alpha * latent + sigma * noise, noise

    def training_target(
        self, latent: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        if self.config.prediction_target == "eps":
            return noise
        alpha, sigma = self._alpha_sigma(timesteps, latent.device)
        return alpha * noise - sigma * latent

    def predict_x0_and_eps(
        self, noisy: torch.Tensor, prediction: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha, sigma = self._alpha_sigma(timesteps, noisy.device)
        if self.config.prediction_target == "v":
            return (
                alpha * noisy - sigma * prediction,
                sigma * noisy + alpha * prediction,
            )
        epsilon = prediction
        return (noisy - sigma * epsilon) / alpha.clamp_min(1e-6), epsilon

    def posterior_variance(
        self, timesteps: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        beta = self.betas.to(device)[timesteps]
        bars = self.alpha_bars.to(device)
        current = bars[timesteps]
        previous = torch.where(
            timesteps > 0, bars[(timesteps - 1).clamp_min(0)], torch.ones_like(current)
        )
        return (beta * (1.0 - previous) / (1.0 - current).clamp_min(1e-12)).clamp_min(
            0.0
        )

    @torch.no_grad()
    def sample(
        self,
        denoiser: LatentDenoiser,
        categorical: dict[str, torch.Tensor],
        multilabel: dict[str, torch.Tensor],
        num_samples: int,
        guidance_scale: float = 1.25,
        generator: torch.Generator | None = None,
        sample_seeds: Sequence[int] | None = None,
    ) -> torch.Tensor:
        device = next(denoiser.parameters()).device
        if sample_seeds is not None:
            if generator is not None or len(sample_seeds) != num_samples:
                raise ValueError(
                    "sample_seeds must match num_samples and cannot be combined with generator."
                )
            trajectories = []
            for seed in sample_seeds:
                row_generator = torch.Generator(device=device.type).manual_seed(
                    int(seed)
                )
                trajectories.append(
                    torch.randn(
                        self.config.num_steps,
                        denoiser.config.latent_dim,
                        device=device,
                        generator=row_generator,
                    )
                )
            noise_trajectory = torch.stack(trajectories, dim=1)
            latent = noise_trajectory[0]
        else:
            noise_trajectory = None
            latent = torch.randn(
                num_samples,
                denoiser.config.latent_dim,
                device=device,
                generator=generator,
            )
        categorical = {field: value.to(device) for field, value in categorical.items()}
        multilabel = {field: value.to(device) for field, value in multilabel.items()}
        betas, alphas, bars = (
            self.betas.to(device),
            self.alphas.to(device),
            self.alpha_bars.to(device),
        )
        for step in reversed(range(self.config.num_steps)):
            timestep = torch.full((num_samples,), step, device=device, dtype=torch.long)
            conditional = denoiser(
                latent, timestep, categorical, multilabel, drop_conditioning=False
            )
            if guidance_scale == 1.0:
                prediction = conditional
            else:
                unconditional = denoiser(
                    latent,
                    timestep,
                    categorical,
                    multilabel,
                    drop_conditioning=False,
                    force_drop_primary=True,
                    force_drop_secondary=True,
                )
                prediction = unconditional + guidance_scale * (
                    conditional - unconditional
                )
            _, epsilon = self.predict_x0_and_eps(latent, prediction, timestep)
            if step:
                noise = (
                    noise_trajectory[self.config.num_steps - step]
                    if noise_trajectory is not None
                    else torch.randn(
                        latent.shape,
                        device=device,
                        dtype=latent.dtype,
                        generator=generator,
                    )
                )
            else:
                noise = torch.zeros_like(latent)
            latent = (
                latent - betas[step] / torch.sqrt(1.0 - bars[step]) * epsilon
            ) / torch.sqrt(alphas[step])
            latent = (
                latent
                + self.posterior_variance(timestep, device)[:, None].sqrt() * noise
            )
        return latent


def normalize_latent(
    latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (latent - mean.to(latent)) / std.to(latent).clamp_min(1e-6)


def denormalize_latent(
    latent: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return latent * std.to(latent).clamp_min(1e-6) + mean.to(latent)
