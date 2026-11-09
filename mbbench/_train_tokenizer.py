from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .common import (
    capture_rng_state,
    load_yaml,
    new_output_dir,
    require_bool,
    restore_rng_state,
    select_device,
    set_seed,
    write_training_progress,
    write_yaml,
)
from .data import (
    MicrobiomeDataset,
    PreparedData,
    _split_tokens,
)
from .defaults import load_default_config
from .losses import (
    abundance_reconstruction_loss,
    focal_bce_with_logits,
    kl_beta_schedule,
    kl_divergence,
    masked_compositional_loss,
)
from .tokenizer import TokenizerConfig, build_tokenizer


def _vocabulary_size(vocabulary: dict[str, Any]) -> int:
    return len(vocabulary.get("values", vocabulary.get("tokens", [])))


def _apply_loss_overrides(
    config: dict[str, Any], overrides: Mapping[str, object] | None
) -> None:
    if overrides is None:
        return
    if not isinstance(overrides, Mapping):
        raise TypeError(
            "loss_overrides must be a mapping of loss setting names to values."
        )
    loss_config = config.get("loss")
    if not isinstance(loss_config, dict):
        raise ValueError("Tokenizer configuration must contain a loss mapping.")
    unknown = sorted(set(map(str, overrides)) - set(loss_config))
    if unknown:
        raise ValueError("Unknown loss settings: " + ", ".join(unknown) + ".")
    for raw_name, value in overrides.items():
        name = str(raw_name)
        if isinstance(value, bool):
            raise ValueError(f"loss_overrides[{name!r}] must be numeric.")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"loss_overrides[{name!r}] must be numeric.") from error
        if not np.isfinite(numeric) or numeric < 0:
            raise ValueError(
                f"loss_overrides[{name!r}] must be finite and nonnegative."
            )
        if name == "support_pos_weight_cap" and numeric < 1:
            raise ValueError("support_pos_weight_cap must be at least 1.")
        if name.endswith(("_epoch", "_epochs")):
            if not numeric.is_integer():
                raise ValueError(f"loss_overrides[{name!r}] must be an integer.")
            loss_config[name] = int(numeric)
        else:
            loss_config[name] = numeric


def _center_weights(
    dataset: MicrobiomeDataset, fields: list[str]
) -> dict[str, tuple[float, ...]]:
    sample_weights = torch.zeros(len(dataset), dtype=torch.float64)
    project = dataset.categorical.get("Project_Group")
    complete = dataset.nuisance_complete.bool()
    if project is None:
        sample_weights.fill_(1.0)
    elif complete.any():
        project_counts = torch.bincount(
            project[complete],
            minlength=_vocabulary_size(dataset.vocabularies["Project_Group"]),
        ).to(torch.float64)
        sample_weights[complete] = (
            project_counts[project[complete]].clamp_min(1.0).rsqrt()
        )
    weights: dict[str, tuple[float, ...]] = {}
    for field in fields:
        values = dataset.categorical.get(field)
        if values is None:
            continue
        size = _vocabulary_size(dataset.vocabularies[field])
        validity = dataset.validity.get(field, torch.ones(len(dataset))).to(
            torch.float64
        )
        counts = torch.bincount(
            values,
            weights=sample_weights * validity,
            minlength=size,
        )
        if counts.sum() <= 0:
            counts.fill_(1.0)
        weights[field] = tuple(float(value) for value in counts)
    return weights


def _configuration(
    config: dict[str, Any], data: PreparedData, train: MicrobiomeDataset
) -> TokenizerConfig:
    model = config["model"]
    technical = [str(field) for field in model["technical_fields"]]
    biological = [str(field) for field in model["biological_fields"]]
    missing_technical = [field for field in technical if field not in data.vocabularies]
    if missing_technical:
        raise ValueError(
            f"Prepared data is missing requested technical vocabularies: {missing_technical}"
        )
    missing_biological = [
        field for field in biological if field not in data.vocabularies
    ]
    if missing_biological:
        raise ValueError(
            f"Prepared data is missing requested biological vocabularies: {missing_biological}"
        )
    overlap = sorted(set(biological) & set(technical))
    if overlap:
        raise ValueError(f"Fields cannot be both biological and technical: {overlap}")
    bio_categorical = {
        field: _vocabulary_size(data.vocabularies[field])
        for field in biological
        if data.vocabularies[field]["type"] == "categorical"
    }
    bio_multilabel = {
        field: _vocabulary_size(data.vocabularies[field])
        for field in biological
        if data.vocabularies[field]["type"] == "multilabel"
    }
    project_field = "Project_Group"
    return TokenizerConfig(
        input_dim=len(data.taxa),
        bio_latent_dim=int(model.get("bio_latent_dim", 128)),
        tech_latent_dim=int(model.get("tech_latent_dim", 32)),
        project_latent_dim=int(model.get("project_latent_dim", 16)),
        encoder_hidden_dim=int(model.get("encoder_hidden_dim", 1024)),
        decoder_hidden_dim=int(model.get("decoder_hidden_dim", 1024)),
        condition_embedding_dim=int(model.get("condition_embedding_dim", 32)),
        dropout=float(model.get("dropout", 0.1)),
        mask_threshold=float(model.get("mask_threshold", 0.40)),
        eps=float(model.get("clr_eps", 1e-6)),
        logvar_min=float(model.get("logvar_min", -12.0)),
        logvar_max=float(model.get("logvar_max", 12.0)),
        technical_field_sizes={
            field: _vocabulary_size(data.vocabularies[field]) for field in technical
        },
        project_field=project_field,
        project_field_size=(
            _vocabulary_size(data.vocabularies[project_field])
            if project_field in data.vocabularies
            else 0
        ),
        bio_categorical_field_sizes=bio_categorical,
        bio_multilabel_field_dims=bio_multilabel,
        grl_lambda=float(model.get("grl_lambda", 0.12)),
        technical_center_weights=_center_weights(train, technical),
        project_center_weights=_center_weights(train, [project_field]).get(
            project_field, ()
        ),
    )


def _schedule(epoch: int, start: int, ramp: int) -> float:
    if epoch + 1 < max(start, 1):
        return 0.0
    if ramp <= 1:
        return 1.0
    return float(min(max((epoch + 2 - max(start, 1)) / ramp, 0.0), 1.0))


def _orthogonality(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if len(left) <= 1:
        return torch.tensor(0.0, device=left.device, dtype=left.dtype)
    left = left - left.mean(0, keepdim=True)
    right = right - right.mean(0, keepdim=True)
    left = left * torch.rsqrt(left.square().mean(0, keepdim=True) + 1e-6)
    right = right * torch.rsqrt(right.square().mean(0, keepdim=True) + 1e-6)
    return (left.T @ right / len(left)).square().mean()


def _hierarchy_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: list[list[int]],
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not groups:
        return torch.tensor(0.0, device=prediction.device)
    per_group_prediction = torch.stack(
        [prediction[:, group].sum(dim=1) for group in groups], dim=1
    )
    per_group_target = torch.stack(
        [target[:, group].sum(dim=1) for group in groups], dim=1
    )
    active_groups = per_group_target > 0
    per_group_loss = F.smooth_l1_loss(
        per_group_prediction, per_group_target, reduction="none"
    )
    per_sample = (per_group_loss * active_groups).sum(dim=1) / active_groups.sum(
        dim=1
    ).clamp_min(1)
    valid = active_groups.any(dim=1)
    if not valid.any():
        return prediction.sum() * 0.0
    if sample_weights is None:
        return per_sample[valid].mean()
    weights = sample_weights.to(per_sample).clamp_min(0) * valid.to(per_sample)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def _mean_cross_entropy(
    logits: dict[str, torch.Tensor],
    conditions: dict[str, torch.Tensor],
    validity: dict[str, torch.Tensor],
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = []
    for field, value in logits.items():
        if field not in conditions:
            continue
        weight = validity.get(field, torch.ones(len(value), device=value.device))
        if sample_weight is not None:
            weight = weight * sample_weight
        if float(weight.sum().item()) <= 0.0:
            continue
        per_sample = F.cross_entropy(value, conditions[field], reduction="none")
        losses.append((per_sample * weight).sum() / weight.sum())
    if losses:
        return torch.stack(losses).mean()
    if logits:
        first = next(iter(logits.values()))
        return torch.tensor(0.0, device=first.device, dtype=first.dtype)
    return torch.tensor(0.0)


def _training_sampler(
    dataset: MicrobiomeDataset, config: dict[str, Any]
) -> WeightedRandomSampler | None:
    policy = config.get("balanced_sampling", {})
    if not policy.get("enabled", False):
        return None
    metadata = dataset.metadata
    body_site = (
        metadata["Body_Site_Coarse"].fillna("<NULL>").astype(str)
        if "Body_Site_Coarse" in metadata
        else pd.Series("all", index=metadata.index)
    )
    complete = dataset.nuisance_complete.numpy()
    weights = np.ones(len(metadata))
    if "Project_Group" in metadata:
        project = metadata["Project_Group"].fillna("<NULL>").astype(str)
        project_counts = project[complete].value_counts()
        project_weight = (
            project[complete].map(project_counts).to_numpy(dtype=float) ** -0.5
        )
    else:
        project_weight = np.ones(int(complete.sum()))
    body_counts = body_site.value_counts()
    body_weight = body_site[complete].map(body_counts).to_numpy(dtype=float) ** -float(
        policy.get("body_site_exponent", 0.15)
    )
    weights[complete] = project_weight * body_weight
    weights /= weights.mean()
    lower, upper = policy.get("clip_range", [0.25, 4.0])
    weights[complete] = weights[complete].clip(float(lower), float(upper))
    weights = torch.tensor(weights, dtype=torch.double)
    generator = torch.Generator().manual_seed(
        int(policy.get("seed", config.get("seed", 42)))
    )
    return WeightedRandomSampler(
        weights, len(dataset), replacement=True, generator=generator
    )


def _set_adversarial_weights(
    dataset: MicrobiomeDataset, config: dict[str, Any]
) -> None:
    policy = config.get("bio_batch_adversarial_sample_weighting", {})
    if not policy.get("enabled", False):
        return
    metadata = dataset.metadata
    if "Project_Group" not in metadata:
        dataset.adversarial_weights.zero_()
        return
    if not {"Phenotype_Tokens", "Body_Site_Coarse"}.issubset(metadata.columns):
        dataset.adversarial_weights = dataset.nuisance_complete.float()
        return
    minimum_projects = int(policy.get("min_projects", 2))
    minimum_samples = int(policy.get("min_samples_per_project", 5))
    projects = metadata["Project_Group"].fillna("<NULL>").astype(str)
    sites = metadata["Body_Site_Coarse"].fillna("<NULL>").astype(str)
    eligible_sites = {
        site
        for site, rows in metadata.assign(_site=sites, _project=projects).groupby(
            "_site"
        )
        if (rows["_project"].value_counts() >= minimum_samples).sum()
        >= minimum_projects
    }
    token_projects: dict[str, dict[str, int]] = {}
    for value, project in zip(metadata["Phenotype_Tokens"], projects, strict=True):
        for token in _split_tokens(value):
            token_projects.setdefault(token, {})[project] = (
                token_projects.setdefault(token, {}).get(project, 0) + 1
            )
    eligible_tokens = {
        token
        for token, counts in token_projects.items()
        if sum(count >= minimum_samples for count in counts.values())
        >= minimum_projects
    }
    weights = []
    for site, value in zip(sites, metadata["Phenotype_Tokens"], strict=True):
        tokens = set(_split_tokens(value))
        weights.append(
            float(
                site in eligible_sites
                and (not tokens or bool(tokens & eligible_tokens))
            )
        )
    dataset.adversarial_weights = (
        torch.tensor(weights, dtype=torch.float32) * dataset.nuisance_complete
    )


def _set_reconstruction_weights(
    dataset: MicrobiomeDataset, config: dict[str, Any]
) -> None:
    incomplete_weight = float(
        config.get("nuisance_incomplete_reconstruction_weight", 0.5)
    )
    if not 0.0 <= incomplete_weight <= 1.0:
        raise ValueError(
            "nuisance_incomplete_reconstruction_weight must be between 0 and 1."
        )
    dataset.reconstruction_weights = torch.where(
        dataset.nuisance_complete,
        torch.ones(len(dataset), dtype=torch.float32),
        torch.full((len(dataset),), incomplete_weight, dtype=torch.float32),
    )


def _calibrate_support_threshold(probability: np.ndarray, truth: np.ndarray) -> float:
    """Select the hurdle decoder threshold on validation data."""

    true_richness = truth.sum(axis=1)
    true_sparsity = 1.0 - truth.mean()
    true_prevalence = truth.mean(axis=0)
    rows = []
    for threshold in np.arange(0.05, 0.951, 0.01):
        predicted = probability >= threshold
        empty = ~predicted.any(axis=1)
        predicted[empty, probability[empty].argmax(axis=1)] = True
        scores = []
        for column in range(truth.shape[1]):
            target = truth[:, column].astype(bool)
            if target.all() or not target.any():
                continue
            estimate = predicted[:, column]
            tp = np.logical_and(target, estimate).sum()
            denominator = (
                2 * tp
                + np.logical_and(~target, estimate).sum()
                + np.logical_and(target, ~estimate).sum()
            )
            scores.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
        macro_f1 = float(np.mean(scores)) if scores else 0.0
        quantiles = np.linspace(0.0, 1.0, max(len(truth), 2))
        richness_error = np.abs(
            np.quantile(true_richness, quantiles)
            - np.quantile(predicted.sum(axis=1), quantiles)
        ).mean() / max(float(true_richness.mean()), 1.0)
        ecological = (
            richness_error
            + abs(float(1.0 - predicted.mean()) - true_sparsity)
            + np.abs(predicted.mean(axis=0) - true_prevalence).mean()
        ) / 3.0
        rows.append((float(threshold), macro_f1, float(ecological)))
    best_f1 = max(row[1] for row in rows)
    eligible = [row for row in rows if row[1] >= 0.95 * best_f1]
    return min(eligible, key=lambda row: (row[2], -row[1], row[0]))[0]


def _loss(
    model,
    batch: dict[str, Any],
    *,
    model_type: str,
    loss_config: dict[str, Any],
    epoch: int,
    support_pos_weight: torch.Tensor,
    multilabel_pos_weights: dict[str, torch.Tensor],
    hierarchy_groups: list[list[int]],
) -> tuple[torch.Tensor, dict[str, float]]:
    abundance, mask = batch["abundance"], batch["mask"]
    categorical, multilabel = batch["categorical"], batch["multilabel"]
    reconstruction_weights = batch.get("reconstruction_weight")
    adv_factor = _schedule(
        epoch,
        int(loss_config.get("bio_batch_adversarial_start_epoch", 5)),
        int(loss_config.get("bio_batch_adversarial_ramp_epochs", 8)),
    )
    output = model(
        abundance,
        mask,
        categorical_conditions=categorical,
        condition_validity=batch["validity"],
        read_depth=batch.get("read_depth"),
        teacher_force_mask=True,
        deterministic_posterior=False,
        grl_lambda_override=model.config.grl_lambda
        * _schedule(
            epoch,
            int(loss_config.get("grl_start_epoch", 5)),
            int(loss_config.get("grl_ramp_epochs", 8)),
        ),
    )
    mask_loss = focal_bce_with_logits(
        output["mask_logits"],
        mask,
        float(loss_config["focal_gamma"]),
        support_pos_weight,
        reconstruction_weights,
    )
    reconstruction = abundance_reconstruction_loss(
        output["recon_abundance"],
        abundance,
        output["recon_mask"],
        reconstruction_weights,
    )
    composition = masked_compositional_loss(
        output["recon_abundance"],
        abundance,
        output["recon_mask"],
        sample_weights=reconstruction_weights,
    )
    hierarchy = _hierarchy_loss(
        output["recon_abundance"], abundance, hierarchy_groups, reconstruction_weights
    )
    kl = (
        kl_divergence(output["mu"], output["logvar"])
        if model_type == "vae"
        else abundance.sum() * 0.0
    )
    batch_supervision = _mean_cross_entropy(
        output["batch_logits"],
        categorical,
        batch["validity"],
        batch["nuisance_complete"],
    ).to(abundance.device)
    adversarial = _mean_cross_entropy(
        output["bio_batch_logits"],
        categorical,
        batch["validity"],
        batch["adversarial_weight"],
    ).to(abundance.device)
    bio_losses: list[torch.Tensor] = []
    for field, logits in output["bio_supervision_logits"].items():
        if field not in categorical:
            continue
        per_sample = F.cross_entropy(logits, categorical[field], reduction="none")
        weight = batch["validity"][field]
        if float(weight.sum().item()) > 0.0:
            bio_losses.append((per_sample * weight).sum() / weight.sum())
    for field, logits in output["bio_supervision_logits"].items():
        if field not in multilabel:
            continue
        per_sample = F.binary_cross_entropy_with_logits(
            logits,
            multilabel[field],
            pos_weight=multilabel_pos_weights[field].to(abundance),
            reduction="none",
        ).mean(dim=1)
        weight = batch["validity"][field]
        if float(weight.sum().item()) > 0.0:
            bio_losses.append((per_sample * weight).sum() / weight.sum())
    bio = (
        torch.stack(bio_losses).mean()
        if bio_losses
        else torch.tensor(0.0, device=abundance.device, dtype=abundance.dtype)
    )
    nuisance_complete = batch["nuisance_complete"] > 0
    representation = output["bio_representation_latent"][nuisance_complete]
    tech_latent = output["tech_latent"][nuisance_complete]
    project_latent = output["project_latent"][nuisance_complete]
    batch_factor = _schedule(
        epoch,
        int(loss_config.get("batch_supervision_start_epoch", 1)),
        int(loss_config.get("batch_supervision_ramp_epochs", 2)),
    )
    bio_factor = _schedule(
        epoch,
        int(loss_config.get("bio_supervision_start_epoch", 2)),
        int(loss_config.get("bio_supervision_ramp_epochs", 4)),
    )
    orth_factor = _schedule(
        epoch,
        int(loss_config.get("orthogonality_start_epoch", 2)),
        int(loss_config.get("orthogonality_ramp_epochs", 6)),
    )
    if orth_factor > 0.0:
        orthogonality = torch.stack(
            [
                _orthogonality(representation, tech_latent),
                _orthogonality(representation, project_latent),
            ]
        ).mean()
    else:
        with torch.no_grad():
            orthogonality = torch.stack(
                [
                    _orthogonality(representation, tech_latent),
                    _orthogonality(representation, project_latent),
                ]
            ).mean()
    total = (
        float(loss_config["mask_weight"]) * mask_loss
        + float(loss_config["recon_weight"]) * reconstruction
        + float(loss_config["compositional_weight"]) * composition
        + float(loss_config["hierarchy_weight"]) * hierarchy
        + kl_beta_schedule(
            epoch,
            int(loss_config["kl_warmup_epochs"]),
            float(loss_config["kl_beta_max"]),
        )
        * kl
        + float(loss_config["batch_supervision_weight"])
        * batch_factor
        * batch_supervision
        + float(loss_config["bio_batch_adversarial_weight"]) * adv_factor * adversarial
        + float(loss_config["bio_supervision_weight"]) * bio_factor * bio
        + float(loss_config["orthogonality_weight"]) * orth_factor * orthogonality
        + float(loss_config["project_shrinkage_weight"])
        * output["project_shrinkage_loss"]
    )
    return total, {
        "total_loss": float(total.detach()),
        "support_loss": float(mask_loss.detach()),
        "reconstruction_loss": float(reconstruction.detach()),
        "composition_loss": float(composition.detach()),
        "kl_loss": float(kl.detach()),
    }


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "abundance": batch["abundance"].to(device),
        "mask": batch["mask"].to(device),
        "categorical": {
            field: value.to(device) for field, value in batch["categorical"].items()
        },
        "multilabel": {
            field: value.to(device) for field, value in batch["multilabel"].items()
        },
        "validity": {
            field: value.to(device) for field, value in batch["validity"].items()
        },
        "adversarial_weight": batch["adversarial_weight"].to(device),
        "nuisance_complete": batch["nuisance_complete"].to(device),
        "reconstruction_weight": batch["reconstruction_weight"].to(device),
        **(
            {"read_depth": batch["read_depth"].to(device)}
            if "read_depth" in batch
            else {}
        ),
    }


def train_tokenizer(
    *,
    config_path: str | Path | None = None,
    model_type: str | None = None,
    data_dir: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    seed: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    loss_overrides: Mapping[str, object] | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float | None = None,
    early_stopping_min_epochs: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    """Train an AE or VAE tokenizer."""

    resume = require_bool(resume, "resume")
    overwrite = require_bool(overwrite, "overwrite")
    if config_path is None:
        selected_type = "ae" if model_type is None else str(model_type).lower()
        if selected_type not in {"ae", "vae"}:
            raise ValueError("model_type must be 'ae' or 'vae'.")
        config = load_default_config(f"tokenizer_{selected_type}.yaml")
    else:
        config = load_yaml(config_path)
        if (
            model_type is not None
            and str(config.get("model_type", "")).lower() != str(model_type).lower()
        ):
            raise ValueError("model_type does not match config_path.")
    _apply_loss_overrides(config, loss_overrides)
    repeated_fields = sorted(
        set(config.get("model", {})) & {"biological_fields", "technical_fields"}
    )
    if repeated_fields:
        raise ValueError(
            "Tokenizer condition fields come from prepared data; remove these model settings: "
            + ", ".join(repeated_fields)
            + "."
        )
    if seed is not None:
        config["seed"] = seed
    for key, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("learning_rate", learning_rate),
    ):
        if value is not None:
            config["training"][key] = value
    for key, value in (
        ("early_stopping_patience", early_stopping_patience),
        ("early_stopping_min_delta", early_stopping_min_delta),
        ("early_stopping_min_epochs", early_stopping_min_epochs),
    ):
        if value is not None:
            config["training"][key] = value
    if resume and overwrite:
        raise ValueError("resume and overwrite cannot be used together.")
    training = config["training"]
    epochs = int(training["epochs"])
    patience = int(training["early_stopping_patience"])
    minimum_delta = float(training.get("early_stopping_min_delta", 0.0))
    minimum_epochs = int(training["early_stopping_min_epochs"])
    selection_start_epoch = int(
        training.get("checkpoint_selection_start_epoch", minimum_epochs)
    )
    if (
        epochs <= 0
        or patience <= 0
        or minimum_epochs <= 0
        or selection_start_epoch <= 0
        or minimum_delta < 0
    ):
        raise ValueError(
            "epochs and early-stopping counts must be positive; min_delta must be nonnegative."
        )
    model_type = str(config["model_type"]).lower()
    set_seed(int(config.get("seed", 42)))
    device = select_device(device_name)
    data = PreparedData.load(data_dir)
    config["model"]["biological_fields"] = list(data.biological_fields)
    config["model"]["technical_fields"] = list(data.technical_fields)
    train_data, validation_data = MicrobiomeDataset(data, "train"), MicrobiomeDataset(
        data, "validation"
    )
    if not len(train_data) or not len(validation_data):
        raise ValueError(
            "Tokenizer training requires nonempty train and validation splits."
        )
    tokenizer_config = _configuration(config, data, train_data)
    model = build_tokenizer(tokenizer_config, model_type).to(device)
    loss_config = config["loss"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    support = train_data.mask.sum(0)
    ratio = (len(train_data) - support) / support.clamp_min(1)
    support_pos_weight = (
        ratio.pow(float(loss_config["support_pos_weight_exponent"]))
        .clamp(1, float(loss_config["support_pos_weight_cap"]))
        .to(device)
    )
    multilabel_pos_weights = {}
    for field, values in train_data.multilabel.items():
        validity = train_data.validity[field][:, None]
        positives = (values * validity).sum(0)
        negatives = validity.sum() - positives
        multilabel_pos_weights[field] = (negatives / positives.clamp_min(1.0)).clamp(
            1.0, 20.0
        )
    taxonomy_path = Path(data_dir) / "taxonomy.csv"
    hierarchy_groups: list[list[int]] = []
    if taxonomy_path.is_file():
        taxonomy = (
            pd.read_csv(taxonomy_path)
            .set_index("taxon")
            .reindex(data.taxa)
            .fillna("unknown")
        )
        hierarchy_groups = [
            indices.index.tolist()
            for _, indices in taxonomy.reset_index().groupby("family")
        ]
    _set_adversarial_weights(train_data, training)
    _set_reconstruction_weights(train_data, training)
    _set_reconstruction_weights(validation_data, training)
    sampler = _training_sampler(train_data, training)
    train_loader = DataLoader(
        train_data,
        batch_size=int(training["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=int(training["batch_size"]), shuffle=False
    )
    output_path = Path(output_dir).expanduser().resolve()
    if resume:
        checkpoint_path = output_path / "checkpoints" / "last.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {checkpoint_path}"
            )
        output = output_path
    else:
        output = new_output_dir(output_path, overwrite=overwrite)
    (output / "checkpoints").mkdir(exist_ok=True)
    start_clock = time.time()
    start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    progress_dir = output / "progress"
    base_status = {
        "start_time": start_time,
        "model_type": model_type,
        "run_dir": str(output),
        "total_steps": epochs * len(train_loader),
    }
    history: list[dict[str, float | int]] = []
    best, best_for_early_stopping, best_epoch, stale = float("inf"), float("inf"), 0, 0
    start_epoch = 0
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_type = str(checkpoint.get("model_type", "")).removeprefix(
            "protocol_hurdle_"
        )
        if checkpoint_type != model_type:
            raise ValueError(
                f"Resume checkpoint model type is {checkpoint_type!r}, not {model_type!r}."
            )
        expected_resume = {
            "tokenizer_config": tokenizer_config.__dict__,
            "taxa": data.taxa,
            "vocabularies": data.vocabularies,
            "biological_fields": list(data.biological_fields),
            "technical_fields": list(data.technical_fields),
            "seed": int(config.get("seed", 42)),
        }
        missing_resume = [name for name in expected_resume if name not in checkpoint]
        if missing_resume:
            raise ValueError(
                "Resume checkpoint lacks compatibility fields: "
                + ", ".join(missing_resume)
                + ". Start a new run."
            )
        mismatched = [
            name for name, value in expected_resume.items() if checkpoint[name] != value
        ]
        if mismatched:
            raise ValueError(
                "Resume settings do not match the checkpoint: "
                + ", ".join(mismatched)
                + "."
            )
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        best = float(
            checkpoint.get("best_validation_loss", checkpoint["validation_loss"])
        )
        best_for_early_stopping = float(
            checkpoint.get("best_early_stopping_validation_loss", best)
        )
        best_epoch = int(checkpoint.get("best_epoch", start_epoch))
        stale = int(checkpoint.get("epochs_without_improvement", 0))
        if (
            sampler is not None
            and checkpoint.get("sampler_generator_state") is not None
        ):
            sampler.generator.set_state(checkpoint["sampler_generator_state"])
        if checkpoint.get("rng_state") is not None:
            restore_rng_state(checkpoint["rng_state"])
        if start_epoch >= epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch}; --epochs must be larger."
            )
    write_training_progress(
        progress_dir,
        "training resumed" if resume else "training started",
        {
            **base_status,
            "epoch": start_epoch,
            "step": start_epoch * len(train_loader),
            "progress_percent": 100.0 * start_epoch / epochs,
            "current_loss": None,
            "best_val_loss": None if best == float("inf") else best,
            "eta_seconds": None,
            "state": "running",
        },
    )
    for epoch in range(start_epoch, epochs):
        model.train()
        train_losses: list[float] = []
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            total, _ = _loss(
                model,
                batch,
                model_type=model_type,
                loss_config=loss_config,
                epoch=epoch,
                support_pos_weight=support_pos_weight,
                multilabel_pos_weights=multilabel_pos_weights,
                hierarchy_groups=hierarchy_groups,
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["max_grad_norm"])
            )
            optimizer.step()
            train_losses.append(float(total.detach()))
            step = epoch * len(train_loader) + batch_index
            if batch_index == len(train_loader) or batch_index % 25 == 0:
                elapsed = time.time() - start_clock
                eta = (
                    elapsed / step * (base_status["total_steps"] - step)
                    if step
                    else None
                )
                write_training_progress(
                    progress_dir,
                    f"epoch {epoch + 1} batch {batch_index}/{len(train_loader)}",
                    {
                        **base_status,
                        "epoch": epoch + 1,
                        "step": step,
                        "progress_percent": 100.0 * step / base_status["total_steps"],
                        "current_loss": train_losses[-1],
                        "best_val_loss": None if best == float("inf") else best,
                        "eta_seconds": eta,
                        "state": "running",
                    },
                )
                print(
                    f"epoch {epoch + 1}/{epochs} batch {batch_index}/{len(train_loader)} loss={train_losses[-1]:.6f}",
                    flush=True,
                )
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for raw_batch in validation_loader:
                batch = _to_device(raw_batch, device)
                total, _ = _loss(
                    model,
                    batch,
                    model_type=model_type,
                    loss_config=loss_config,
                    epoch=epoch,
                    support_pos_weight=support_pos_weight,
                    multilabel_pos_weights=multilabel_pos_weights,
                    hierarchy_groups=hierarchy_groups,
                )
                validation_losses.append(float(total))
        train_loss = sum(train_losses) / len(train_losses)
        validation_loss = sum(validation_losses) / len(validation_losses)
        print(
            f"epoch {epoch + 1}/{epochs} validation_loss={validation_loss:.6f}",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_total_loss": train_loss,
                "validation_total_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        payload = {
            "state_dict": model.state_dict(),
            "model_type": model_type,
            "tokenizer_config": tokenizer_config.__dict__,
            "data_dir": str(Path(data_dir).resolve()),
            "read_depth_stats": data.read_depth_stats,
            "taxa": data.taxa,
            "vocabularies": data.vocabularies,
            "biological_fields": list(data.biological_fields),
            "technical_fields": list(data.technical_fields),
            "epoch": epoch + 1,
            "validation_loss": validation_loss,
            "seed": int(config.get("seed", 42)),
        }
        eligible = epoch + 1 >= selection_start_epoch
        if eligible and validation_loss < best:
            best, best_epoch = validation_loss, epoch + 1
            torch.save(payload, output / "checkpoints" / "best.pt")
        if eligible and validation_loss < best_for_early_stopping - minimum_delta:
            best_for_early_stopping, stale = validation_loss, 0
        elif eligible:
            stale += 1
        torch.save(
            {
                **payload,
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_validation_loss": best,
                "best_early_stopping_validation_loss": best_for_early_stopping,
                "best_epoch": best_epoch,
                "epochs_without_improvement": stale,
                "rng_state": capture_rng_state(),
                "sampler_generator_state": (
                    sampler.generator.get_state() if sampler is not None else None
                ),
            },
            output / "checkpoints" / "last.pt",
        )
        if epoch + 1 >= minimum_epochs and stale >= patience:
            break
    pd.DataFrame(history).to_csv(output / "metrics_history.csv", index=False)
    best_path = output / "checkpoints" / "best.pt"
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_payload["state_dict"])
    model.eval()
    probabilities, targets = [], []
    with torch.no_grad():
        for raw_batch in validation_loader:
            batch = _to_device(raw_batch, device)
            decoded = model(
                batch["abundance"],
                batch["mask"],
                categorical_conditions=batch["categorical"],
                condition_validity=batch["validity"],
                read_depth=batch.get("read_depth"),
                deterministic_posterior=True,
            )
            probabilities.append(decoded["mask_prob"].cpu().numpy())
            targets.append(batch["mask"].cpu().numpy())
    calibrated_threshold = _calibrate_support_threshold(
        np.concatenate(probabilities), np.concatenate(targets)
    )
    best_payload["calibrated_mask_threshold"] = calibrated_threshold
    torch.save(best_payload, best_path)
    config["result"] = {
        "best_epoch": best_epoch,
        "best_validation_loss": best,
        "stopped_epoch": len(history),
        "calibrated_mask_threshold": calibrated_threshold,
    }
    write_yaml(output / "training_config.yaml", config)
    write_training_progress(
        progress_dir,
        "training finished",
        {
            **base_status,
            "epoch": len(history),
            "step": len(history) * len(train_loader),
            "progress_percent": 100.0,
            "current_loss": history[-1]["train_total_loss"],
            "best_val_loss": best,
            "eta_seconds": 0.0,
            "state": "completed",
        },
    )
    return output
