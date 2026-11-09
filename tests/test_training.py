from __future__ import annotations

from pathlib import Path

import torch
import yaml
import pytest

from mbbench.data import MicrobiomeDataset, PreparedData, prepare_data
from mbbench.defaults import load_default_config
from mbbench.losses import (
    abundance_reconstruction_loss,
    focal_bce_with_logits,
    masked_compositional_loss,
)
from mbbench._package_model import package_model
from mbbench.simulator import Simulator
from mbbench._train_diffusion import (
    _latent_statistics,
    _validate_tokenizer_data,
    train_diffusion,
)
from mbbench._train_tokenizer import (
    _apply_loss_overrides,
    _center_weights,
    _mean_cross_entropy,
    _orthogonality,
    _set_reconstruction_weights,
    train_tokenizer,
)


def test_tokenizer_loss_overrides_are_checked():
    config = load_default_config("tokenizer_ae.yaml")
    _apply_loss_overrides(
        config,
        {"mask_weight": 1.5, "bio_batch_adversarial_weight": 0.02},
    )
    assert config["loss"]["mask_weight"] == 1.5
    assert config["loss"]["bio_batch_adversarial_weight"] == 0.02
    with pytest.raises(ValueError, match="Unknown loss settings"):
        _apply_loss_overrides(config, {"made_up_weight": 1.0})
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _apply_loss_overrides(config, {"mask_weight": -1.0})


def test_reconstruction_weights_match_nuisance_completeness(tmp_path):
    data_dir = prepare_data(
        abundance_path=Path(__file__).parents[1] / "examples" / "abundance_small.csv",
        metadata_path=Path(__file__).parents[1] / "examples" / "metadata_small.csv",
        output_dir=tmp_path / "weighted_data",
        split_column="split",
    )
    dataset = MicrobiomeDataset(PreparedData.load(data_dir), "train")
    _set_reconstruction_weights(
        dataset, {"nuisance_incomplete_reconstruction_weight": 0.5}
    )
    assert torch.equal(
        dataset.reconstruction_weights,
        torch.where(
            dataset.nuisance_complete,
            torch.ones(len(dataset)),
            torch.full((len(dataset),), 0.5),
        ),
    )


def test_training_defaults_can_be_selected_without_yaml(tmp_path):
    assert load_default_config("tokenizer_ae.yaml")["model_type"] == "ae"
    assert load_default_config("tokenizer_vae.yaml")["model_type"] == "vae"
    diffusion = load_default_config("diffusion.yaml")
    assert "model" in diffusion
    assert diffusion["training"]["early_stopping_patience"] == 12
    with pytest.raises(ValueError, match="model_type must be"):
        train_tokenizer(
            model_type="unsupported",
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "run",
        )


def test_diffusion_latent_statistics_apply_the_production_variance_floor():
    class Tokenizer:
        def encode_to_mu(self, abundance, mask):
            return abundance

    loader = [
        {
            "abundance": torch.tensor([[1.0, 2.0], [1.0, 4.0]]),
            "mask": torch.ones(2, 2),
        }
    ]
    mean, std, count = _latent_statistics(Tokenizer(), loader, torch.device("cpu"))
    assert count == 2
    assert torch.equal(mean, torch.tensor([1.0, 3.0]))
    assert torch.allclose(std, torch.tensor([1.0e-3, 1.0]))


def test_diffusion_rejects_tokenizer_taxa_mismatch_before_training():
    data = type("Prepared", (), {"taxa": ["Taxon_A", "Taxon_B"]})()
    matching = {
        "taxa": ["Taxon_A", "Taxon_B"],
        "tokenizer_config": {"input_dim": 2},
    }
    _validate_tokenizer_data(matching, data)
    with pytest.raises(ValueError, match="taxa do not match"):
        _validate_tokenizer_data(
            {**matching, "taxa": ["Taxon_B", "Taxon_A"]}, data
        )
    with pytest.raises(ValueError, match="missing its taxa roster"):
        _validate_tokenizer_data({"tokenizer_config": {"input_dim": 2}}, data)


def test_training_configs_cannot_redefine_prepared_condition_fields(tmp_path):
    tokenizer_config = load_default_config("tokenizer_ae.yaml")
    tokenizer_config["model"]["biological_fields"] = ["Cohort"]
    tokenizer_path = tmp_path / "tokenizer.yaml"
    tokenizer_path.write_text(
        yaml.safe_dump(tokenizer_config, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="come from prepared data"):
        train_tokenizer(
            config_path=tokenizer_path,
            data_dir=tmp_path / "missing-data",
            output_dir=tmp_path / "tokenizer-run",
        )

    diffusion_config = load_default_config("diffusion.yaml")
    diffusion_config["model"]["biological_fields"] = ["Cohort"]
    diffusion_path = tmp_path / "diffusion.yaml"
    diffusion_path.write_text(
        yaml.safe_dump(diffusion_config, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="come from prepared data"):
        train_diffusion(
            config_path=diffusion_path,
            data_dir=tmp_path / "missing-data",
            tokenizer_checkpoint=tmp_path / "missing-tokenizer.pt",
            output_dir=tmp_path / "diffusion-run",
        )


def test_public_reconstruction_losses_accept_sample_weights():
    logits = torch.tensor([[0.2, -0.4], [1.0, -1.0]])
    target_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    weights = torch.tensor([1.0, 0.0])
    assert torch.allclose(
        focal_bce_with_logits(logits, target_mask, gamma=1.0, sample_weights=weights),
        focal_bce_with_logits(logits[:1], target_mask[:1], gamma=1.0),
    )
    prediction = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    target = torch.tensor([[0.6, 0.4], [0.9, 0.1]])
    support = torch.ones_like(target)
    assert torch.allclose(
        abundance_reconstruction_loss(prediction, target, support, weights),
        abundance_reconstruction_loss(prediction[:1], target[:1], support[:1]),
    )
    assert torch.allclose(
        masked_compositional_loss(prediction, target, support, sample_weights=weights),
        masked_compositional_loss(prediction[:1], target[:1], support[:1]),
    )


def test_empty_supervision_and_singleton_orthogonality_do_not_create_gradients():
    logits = torch.nn.Parameter(torch.tensor([[0.2, -0.4], [1.0, -1.0]]))
    empty_loss = _mean_cross_entropy(
        {"field": logits},
        {"field": torch.tensor([0, 1])},
        {"field": torch.zeros(2)},
        torch.ones(2),
    )
    assert not empty_loss.requires_grad

    left = torch.nn.Parameter(torch.ones(1, 2))
    right = torch.nn.Parameter(torch.ones(1, 2))
    orthogonality = _orthogonality(left, right)
    assert not orthogonality.requires_grad


def test_project_balanced_center_weights(tmp_path):
    data_dir = prepare_data(
        abundance_path=Path(__file__).parents[1] / "examples" / "abundance_small.csv",
        metadata_path=Path(__file__).parents[1] / "examples" / "metadata_small.csv",
        output_dir=tmp_path / "balanced_data",
        split_column="split",
    )
    data = PreparedData.load(data_dir)
    train = MicrobiomeDataset(data, "train")
    weights = _center_weights(train, ["Project_Group", "Instrument"])
    project = train.categorical["Project_Group"]
    complete = train.nuisance_complete.bool()
    counts = torch.bincount(
        project[complete], minlength=len(weights["Project_Group"])
    ).float()
    expected_samples = torch.zeros(len(train))
    expected_samples[complete] = counts[project[complete]].clamp_min(1.0).rsqrt()
    expected_project = torch.bincount(
        project,
        weights=expected_samples * train.validity["Project_Group"],
        minlength=len(counts),
    )
    assert torch.allclose(torch.tensor(weights["Project_Group"]), expected_project)
    assert weights["Project_Group"][0] == 0.0
    assert weights["Project_Group"][1] == 0.0


def test_tiny_training_to_custom_generation(tmp_path):
    root = Path(__file__).parents[1]
    data_dir = prepare_data(
        abundance_path=root / "examples" / "abundance_small.csv",
        metadata_path=root / "examples" / "metadata_small.csv",
        output_dir=tmp_path / "data",
        split_column="split",
    )
    tokenizer_config = yaml.safe_load(
        (root / "mbbench" / "defaults" / "tokenizer_ae.yaml").read_text()
    )
    tokenizer_config["model"].update(
        bio_latent_dim=4,
        tech_latent_dim=2,
        project_latent_dim=2,
        encoder_hidden_dim=8,
        decoder_hidden_dim=8,
        condition_embedding_dim=2,
        dropout=0.0,
    )
    tokenizer_config["training"].update(
        epochs=1, batch_size=4, early_stopping_min_epochs=1
    )
    tokenizer_config["training"]["balanced_sampling"]["enabled"] = False
    tokenizer_config["training"]["bio_batch_adversarial_sample_weighting"][
        "enabled"
    ] = False
    tokenizer_yaml = tmp_path / "tokenizer.yaml"
    tokenizer_yaml.write_text(yaml.safe_dump(tokenizer_config), encoding="utf-8")
    tokenizer_run = train_tokenizer(
        config_path=tokenizer_yaml,
        data_dir=data_dir,
        output_dir=tmp_path / "tokenizer",
        device_name="cpu",
    )
    train_tokenizer(
        config_path=tokenizer_yaml,
        data_dir=data_dir,
        output_dir=tokenizer_run,
        device_name="cpu",
        epochs=2,
        resume=True,
    )
    tokenizer_last = torch.load(
        tokenizer_run / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert tokenizer_last["epoch"] == 2
    assert len(tokenizer_last["history"]) == 2
    tokenizer_full = train_tokenizer(
        config_path=tokenizer_yaml,
        data_dir=data_dir,
        output_dir=tmp_path / "tokenizer_full",
        device_name="cpu",
        epochs=2,
    )
    tokenizer_full_last = torch.load(
        tokenizer_full / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    for name, value in tokenizer_last["state_dict"].items():
        assert torch.equal(value, tokenizer_full_last["state_dict"][name])

    diffusion_config = yaml.safe_load(
        (root / "mbbench" / "defaults" / "diffusion.yaml").read_text()
    )
    diffusion_config["model"].update(
        latent_dim=4,
        token_dim=4,
        num_latent_tokens=2,
        num_layers=1,
        num_heads=1,
        mlp_ratio=2.0,
        diffusion_steps=2,
        condition_hidden_dim=8,
        dropout=0.0,
    )
    diffusion_config["training"].update(epochs=1, batch_size=4)
    diffusion_yaml = tmp_path / "diffusion.yaml"
    diffusion_yaml.write_text(yaml.safe_dump(diffusion_config), encoding="utf-8")
    diffusion_run = train_diffusion(
        config_path=diffusion_yaml,
        data_dir=data_dir,
        tokenizer_checkpoint=tokenizer_run / "checkpoints" / "best.pt",
        output_dir=tmp_path / "diffusion",
        device_name="cpu",
    )
    train_diffusion(
        config_path=diffusion_yaml,
        data_dir=data_dir,
        tokenizer_checkpoint=tokenizer_run / "checkpoints" / "best.pt",
        output_dir=diffusion_run,
        device_name="cpu",
        epochs=2,
        resume=True,
    )
    diffusion_last = torch.load(
        diffusion_run / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert diffusion_last["epoch"] == 2
    assert len(diffusion_last["history"]) == 2
    assert diffusion_last["validation_model"] == "ema"
    diffusion_full = train_diffusion(
        config_path=diffusion_yaml,
        data_dir=data_dir,
        tokenizer_checkpoint=tokenizer_run / "checkpoints" / "best.pt",
        output_dir=tmp_path / "diffusion_full",
        device_name="cpu",
        epochs=2,
    )
    diffusion_full_last = torch.load(
        diffusion_full / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    for state_name in ("state_dict", "raw_state_dict"):
        for name, value in diffusion_last[state_name].items():
            assert torch.equal(value, diffusion_full_last[state_name][name])
    model_dir = package_model(
        tokenizer_checkpoint=tokenizer_run / "checkpoints" / "best.pt",
        diffusion_checkpoint=diffusion_run / "checkpoints" / "best.pt",
        data_dir=data_dir,
        output_dir=tmp_path / "model",
    )
    model_config = yaml.safe_load(
        (model_dir / "model.yaml").read_text(encoding="utf-8")
    )
    assert "default_guidance_scale" not in model_config
    assert "default_support_temperature" not in model_config
    assert model_config["read_depth_stats"]["transform"] == "log1p_train_median_iqr"
    assert model_config["read_depth_stats"]["train_sample_count"] > 0
    result = Simulator.from_directory(model_dir, "cpu").generate(n_samples=2, seed=42)
    assert result.abundance.iloc[:, 1:].sum(axis=1).round(6).eq(1.0).all()

    vae_config = yaml.safe_load(
        (root / "mbbench" / "defaults" / "tokenizer_vae.yaml").read_text()
    )
    vae_config["model"].update(tokenizer_config["model"])
    vae_config["training"].update(epochs=1, batch_size=4, early_stopping_min_epochs=1)
    vae_config["training"]["balanced_sampling"]["enabled"] = False
    vae_config["training"]["bio_batch_adversarial_sample_weighting"]["enabled"] = False
    vae_yaml = tmp_path / "vae.yaml"
    vae_yaml.write_text(yaml.safe_dump(vae_config), encoding="utf-8")
    vae_run = train_tokenizer(
        config_path=vae_yaml,
        data_dir=data_dir,
        output_dir=tmp_path / "vae",
        device_name="cpu",
    )
    train_tokenizer(
        config_path=vae_yaml,
        data_dir=data_dir,
        output_dir=vae_run,
        device_name="cpu",
        epochs=2,
        resume=True,
    )
    assert (vae_run / "checkpoints" / "best.pt").is_file()
