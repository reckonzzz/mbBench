from __future__ import annotations

from copy import deepcopy
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from .data import MicrobiomeDataset, PreparedData
from .defaults import load_default_config
from .diffusion import (
    DiffusionConfig,
    GaussianDiffusion,
    LatentDenoiser,
    normalize_latent,
)
from .tokenizer import TokenizerConfig, build_tokenizer


def _load_tokenizer(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = TokenizerConfig(**payload["tokenizer_config"])
    model = build_tokenizer(config, str(payload["model_type"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def _validate_tokenizer_data(payload: dict[str, Any], data: PreparedData) -> None:
    taxa = payload.get("taxa")
    if taxa is None:
        raise ValueError(
            "Tokenizer checkpoint is missing its taxa roster; train or export the tokenizer again."
        )
    if list(taxa) != list(data.taxa):
        raise ValueError(
            "Tokenizer checkpoint taxa do not match the prepared data or their order."
        )
    config = payload.get("tokenizer_config")
    if not isinstance(config, dict):
        raise ValueError("Tokenizer checkpoint is missing tokenizer_config.")
    if int(config.get("input_dim", -1)) != len(data.taxa):
        raise ValueError(
            "Tokenizer checkpoint input dimension does not match the prepared taxa table."
        )


def _condition_sizes(
    data: PreparedData, fields: tuple[str, ...]
) -> tuple[dict[str, int], dict[str, int]]:
    categorical: dict[str, int] = {}
    multilabel: dict[str, int] = {}
    for field in fields:
        vocabulary = data.vocabularies[field]
        if vocabulary["type"] == "multilabel":
            multilabel[field] = len(vocabulary["tokens"])
        else:
            categorical[field] = len(vocabulary["values"])
    return categorical, multilabel


def _conditions(batch: dict[str, Any], denoiser: LatentDenoiser, device: torch.device):
    categorical = {
        field: batch["categorical"][field].to(device)
        for field in denoiser.categorical_fields
    }
    multilabel = {
        field: batch["multilabel"][field].to(device)
        for field in denoiser.multilabel_fields
    }
    return categorical, multilabel


@torch.no_grad()
def _latent_statistics(
    tokenizer, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, int]:
    values: list[torch.Tensor] = []
    for batch in loader:
        abundance, mask = batch["abundance"].to(device), batch["mask"].to(device)
        values.append(tokenizer.encode_to_mu(abundance, mask).cpu())
    latent = torch.cat(values)
    mean = latent.mean(0)
    variance = latent.square().mean(0) - mean.square()
    std = variance.clamp_min(1e-6).sqrt()
    return mean, std, len(latent)


def _ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_parameter, parameter in zip(
            ema.parameters(), model.parameters(), strict=True
        ):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema.buffers(), model.buffers(), strict=True):
            ema_buffer.copy_(buffer)


def train_diffusion(
    *,
    config_path: str | Path | None = None,
    data_dir: str | Path,
    tokenizer_checkpoint: str | Path,
    output_dir: str | Path,
    use_body_system: bool = False,
    device_name: str = "auto",
    seed: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float | None = None,
    early_stopping_min_epochs: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    """Train the latent diffusion model."""

    resume = require_bool(resume, "resume")
    overwrite = require_bool(overwrite, "overwrite")
    config = (
        load_default_config("diffusion.yaml")
        if config_path is None
        else load_yaml(config_path)
    )
    if "biological_fields" in config.get("model", {}):
        raise ValueError(
            "Diffusion condition fields come from prepared data; remove model.biological_fields."
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
    epochs_total = int(training["epochs"])
    patience = int(training["early_stopping_patience"])
    minimum_delta = float(training.get("early_stopping_min_delta", 0.0))
    minimum_epochs = int(training["early_stopping_min_epochs"])
    if epochs_total <= 0 or patience <= 0 or minimum_epochs <= 0 or minimum_delta < 0:
        raise ValueError(
            "epochs and early-stopping counts must be positive; min_delta must be nonnegative."
        )
    training_seed = int(config.get("seed", 42))
    set_seed(training_seed)
    device = select_device(device_name)
    data = PreparedData.load(data_dir)
    config["model"]["biological_fields"] = list(data.biological_fields)
    tokenizer, tokenizer_payload = _load_tokenizer(tokenizer_checkpoint, device)
    _validate_tokenizer_data(tokenizer_payload, data)
    fields = tuple(config["model"]["biological_fields"])
    if use_body_system and "Body_System_Model" not in fields:
        fields = (*fields, "Body_System_Model")
    missing = [field for field in fields if field not in data.vocabularies]
    if missing:
        raise ValueError(
            f"Prepared data is missing diffusion condition vocabularies: {missing}"
        )
    train_data = MicrobiomeDataset(
        data, "train", required_fields=fields, require_nuisance_complete=True
    )
    validation_data = MicrobiomeDataset(
        data, "validation", required_fields=fields, require_nuisance_complete=True
    )
    if not len(train_data) or not len(validation_data):
        raise ValueError(
            "Diffusion training requires nonempty train and validation samples with valid conditions."
        )
    train_loader = DataLoader(
        train_data, batch_size=int(training["batch_size"]), shuffle=True
    )
    validation_loader = DataLoader(
        validation_data, batch_size=int(training["batch_size"]), shuffle=False
    )
    statistics_loader = DataLoader(
        train_data, batch_size=int(training["batch_size"]), shuffle=False
    )
    mean, std, latent_count = _latent_statistics(tokenizer, statistics_loader, device)
    categorical_sizes, multilabel_dims = _condition_sizes(data, fields)
    model_config = config["model"]
    diffusion_config = DiffusionConfig(
        latent_dim=int(model_config.get("latent_dim", 128)),
        token_dim=int(model_config.get("token_dim", 64)),
        num_latent_tokens=int(model_config.get("num_latent_tokens", 12)),
        num_layers=int(model_config.get("num_layers", 6)),
        num_heads=int(model_config.get("num_heads", 4)),
        mlp_ratio=float(model_config.get("mlp_ratio", 4.0)),
        num_steps=int(model_config.get("diffusion_steps", 500)),
        beta_start=float(model_config.get("beta_start", 1e-4)),
        beta_end=float(model_config.get("beta_end", 0.02)),
        condition_hidden_dim=int(model_config.get("condition_hidden_dim", 256)),
        dropout=float(model_config.get("dropout", 0.1)),
        condition_dropout=float(model_config.get("condition_dropout", 0.05)),
        primary_condition_dropout=float(
            model_config.get("primary_condition_dropout", 0.10)
        ),
        secondary_condition_dropout=float(
            model_config.get("secondary_condition_dropout", 0.25)
        ),
        prediction_target=str(model_config.get("prediction_target", "v")),
        primary_fields=fields,
    )
    model = LatentDenoiser(diffusion_config, categorical_sizes, multilabel_dims).to(
        device
    )
    ema = deepcopy(model).eval()
    scheduler = GaussianDiffusion(diffusion_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    ema_decay = float(training.get("ema_decay", 0.999))
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
        "model_type": "diffusion",
        "run_dir": str(output),
        "total_steps": epochs_total * len(train_loader),
    }
    history: list[dict[str, float | int]] = []
    best, best_for_early_stopping, best_epoch, stale = float("inf"), float("inf"), 0, 0
    start_epoch = 0
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if bool(checkpoint.get("use_body_system", False)) != use_body_system:
            raise ValueError(
                "Resume checkpoint Body-System setting does not match this run."
            )
        expected_resume = {
            "model_config": diffusion_config.__dict__,
            "tokenizer_config": tokenizer_payload["tokenizer_config"],
            "tokenizer_model_type": tokenizer_payload["model_type"],
            "taxa": data.taxa,
            "vocabularies": data.vocabularies,
            "condition_fields": list(fields),
            "seed": training_seed,
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
        model.load_state_dict(checkpoint["raw_state_dict"])
        ema.load_state_dict(checkpoint["state_dict"])
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
        if checkpoint.get("rng_state") is not None:
            restore_rng_state(checkpoint["rng_state"])
        if start_epoch >= epochs_total:
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
            "progress_percent": 100.0 * start_epoch / epochs_total,
            "current_loss": None,
            "best_val_loss": None if best == float("inf") else best,
            "eta_seconds": None,
            "state": "running",
        },
    )

    def latent_from(batch):
        with torch.no_grad():
            latent = tokenizer.encode_to_mu(
                batch["abundance"].to(device), batch["mask"].to(device)
            )
        return normalize_latent(latent, mean, std)

    def validation_loss() -> float:
        ema.eval()
        generator = torch.Generator(device=device.type).manual_seed(1042)
        losses: list[float] = []
        with torch.no_grad():
            for batch in validation_loader:
                latent = latent_from(batch)
                categorical, multilabel = _conditions(batch, ema, device)
                timestep = torch.randint(
                    0,
                    diffusion_config.num_steps,
                    (len(latent),),
                    device=device,
                    generator=generator,
                )
                noise = torch.randn(latent.shape, device=device, generator=generator)
                noisy, noise = scheduler.q_sample(latent, timestep, noise)
                target = scheduler.training_target(latent, noise, timestep)
                losses.append(
                    float(
                        F.mse_loss(
                            ema(noisy, timestep, categorical, multilabel), target
                        )
                    )
                )
        return sum(losses) / len(losses)

    for epoch in range(start_epoch, epochs_total):
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            latent = latent_from(batch)
            categorical, multilabel = _conditions(batch, model, device)
            timestep = scheduler.sample_timesteps(len(latent), device)
            noisy, noise = scheduler.q_sample(latent, timestep)
            target = scheduler.training_target(latent, noise, timestep)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(noisy, timestep, categorical, multilabel), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["max_grad_norm"])
            )
            optimizer.step()
            _ema_update(ema, model, ema_decay)
            losses.append(float(loss.detach()))
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
                        "current_loss": losses[-1],
                        "best_val_loss": None if best == float("inf") else best,
                        "eta_seconds": eta,
                        "state": "running",
                    },
                )
                print(
                    f"epoch {epoch + 1}/{epochs_total} batch {batch_index}/{len(train_loader)} loss={losses[-1]:.6f}",
                    flush=True,
                )
        train_loss = sum(losses) / len(losses)
        val_loss = validation_loss()
        print(
            f"epoch {epoch + 1}/{epochs_total} validation_loss={val_loss:.6f}",
            flush=True,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_denoising_loss": train_loss,
                "validation_denoising_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        payload = {
            "state_dict": ema.state_dict(),
            "raw_state_dict": model.state_dict(),
            "model_config": diffusion_config.__dict__,
            "tokenizer_config": tokenizer_payload["tokenizer_config"],
            "tokenizer_model_type": tokenizer_payload["model_type"],
            "tokenizer_checkpoint": str(Path(tokenizer_checkpoint).resolve()),
            "taxa": data.taxa,
            "vocabularies": data.vocabularies,
            "condition_fields": list(fields),
            "latent_stats": {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "sample_count": latent_count,
            },
            "use_body_system": use_body_system,
            "epoch": epoch + 1,
            "validation_loss": val_loss,
            "validation_model": "ema",
            "seed": training_seed,
        }
        if val_loss < best:
            best, best_epoch = val_loss, epoch + 1
            torch.save(payload, output / "checkpoints" / "best.pt")
        if val_loss < best_for_early_stopping - minimum_delta:
            best_for_early_stopping, stale = val_loss, 0
        else:
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
            },
            output / "checkpoints" / "last.pt",
        )
        if epoch + 1 >= minimum_epochs and stale >= patience:
            break
    pd.DataFrame(history).to_csv(output / "metrics_history.csv", index=False)
    config["result"] = {
        "best_epoch": best_epoch,
        "best_validation_loss": best,
        "stopped_epoch": len(history),
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
            "current_loss": history[-1]["train_denoising_loss"],
            "best_val_loss": best,
            "eta_seconds": 0.0,
            "state": "completed",
        },
    )
    return output
