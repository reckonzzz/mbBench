from __future__ import annotations

import shutil
from pathlib import Path

import torch

from .common import new_output_dir, write_yaml
from .data import PreparedData, _split_tokens


def package_model(
    *,
    tokenizer_checkpoint: str | Path,
    diffusion_checkpoint: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    supports_body_system: bool = False,
    body_system_map: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Assemble training outputs into the model directory consumed by Simulator."""

    tokenizer = torch.load(tokenizer_checkpoint, map_location="cpu", weights_only=False)
    diffusion = torch.load(diffusion_checkpoint, map_location="cpu", weights_only=False)
    tokenizer_type = str(tokenizer.get("model_type", "")).removeprefix(
        "protocol_hurdle_"
    )
    diffusion_type = str(
        diffusion.get("tokenizer_model_type", tokenizer_type)
    ).removeprefix("protocol_hurdle_")
    if tokenizer_type != diffusion_type:
        raise ValueError(
            "Tokenizer and diffusion checkpoints use different tokenizer model types."
        )
    tokenizer_config = tokenizer.get("tokenizer_config")
    diffusion_tokenizer_config = diffusion.get("tokenizer_config")
    if (
        diffusion_tokenizer_config is not None
        and tokenizer_config != diffusion_tokenizer_config
    ):
        raise ValueError(
            "Tokenizer and diffusion checkpoints contain different tokenizer configurations."
        )
    diffusion_fields = tuple(diffusion["model_config"].get("primary_fields", ()))
    if supports_body_system != ("Body_System_Model" in diffusion_fields):
        raise ValueError(
            "Body-System packaging flag does not match the diffusion checkpoint conditions."
        )
    data = PreparedData.load(data_dir)
    expected_fields = tuple(data.biological_fields)
    if supports_body_system:
        expected_fields = (*expected_fields, "Body_System_Model")
    if diffusion_fields != expected_fields:
        raise ValueError(
            "Diffusion condition fields do not match the prepared biological fields."
        )
    if tokenizer_config is None:
        raise ValueError("Tokenizer checkpoint is missing tokenizer_config.")
    if int(tokenizer_config.get("input_dim", -1)) != len(data.taxa):
        raise ValueError(
            "Tokenizer input dimension does not match the prepared taxa table."
        )
    diffusion_latent_dim = diffusion["model_config"].get(
        "latent_dim", diffusion["model_config"].get("bio_latent_dim")
    )
    if int(diffusion_latent_dim) != int(tokenizer_config.get("bio_latent_dim", -1)):
        raise ValueError(
            "Diffusion latent dimension does not match the tokenizer checkpoint."
        )
    for name, checkpoint in (("tokenizer", tokenizer), ("diffusion", diffusion)):
        if checkpoint.get("taxa") is not None and list(checkpoint["taxa"]) != list(
            data.taxa
        ):
            raise ValueError(
                f"{name.capitalize()} checkpoint taxa do not match prepared data."
            )
        if (
            checkpoint.get("vocabularies") is not None
            and checkpoint["vocabularies"] != data.vocabularies
        ):
            raise ValueError(
                f"{name.capitalize()} checkpoint vocabularies do not match prepared data."
            )
    if supports_body_system:
        if body_system_map is None or not Path(body_system_map).is_file():
            raise ValueError(
                "Body-System packaging requires an existing body_system_map file."
            )
    output = new_output_dir(output_dir, overwrite=overwrite)
    torch.save(
        {
            "state_dict": tokenizer["state_dict"],
            "model_type": tokenizer.get(
                "model_type", diffusion.get("tokenizer_model_type")
            ),
            "tokenizer_config": tokenizer.get(
                "tokenizer_config", diffusion.get("tokenizer_config")
            ),
            "calibrated_mask_threshold": tokenizer.get("calibrated_mask_threshold"),
        },
        output / "tokenizer.pt",
    )
    torch.save(
        {
            "state_dict": diffusion["state_dict"],
            "model_config": diffusion["model_config"],
            "tokenizer_model_type": diffusion.get("tokenizer_model_type"),
            "latent_stats": diffusion["latent_stats"],
        },
        output / "diffusion.pt",
    )
    shutil.copy2(Path(data_dir) / "taxa.csv", output / "taxa.csv")
    shutil.copy2(Path(data_dir) / "vocabularies.json", output / "vocabularies.json")
    fields = list(
        diffusion["model_config"].get(
            "primary_fields", ("Phenotype_Tokens", "Body_Site_Coarse")
        )
    )
    model_type = tokenizer_type
    train_metadata = data.metadata[data.metadata["split"] == "train"].copy()
    technical_fields = list(
        tokenizer["tokenizer_config"].get("technical_field_sizes", {})
    )
    donor_fields = [
        field
        for field in (*fields, *technical_fields, "Project_Group", "raw_total_reads")
        if field in train_metadata
    ]
    supported_modes = ["relative"]
    if {"Project_Group", "raw_total_reads"}.issubset(donor_fields):
        donors = train_metadata[donor_fields].dropna()
        if len(donors):
            donors.to_csv(output / "donors.csv", index=False)
            supported_modes.append("observed")
    combinations = train_metadata[fields].copy()
    for field in fields:
        if data.vocabularies[field]["type"] == "multilabel":
            combinations[field] = combinations[field].map(
                lambda value: "|".join(sorted(_split_tokens(value)))
            )
    combinations.dropna().drop_duplicates().to_csv(
        output / "condition_combinations.csv", index=False
    )
    if supports_body_system:
        shutil.copy2(body_system_map, output / "body_site_to_system.csv")
    write_yaml(
        output / "model.yaml",
        {
            "method": "MicrobiomeBench",
            "model_type": model_type,
            "training_seed": int(tokenizer.get("seed", diffusion.get("seed", 42))),
            "condition_fields": fields,
            "supports_body_system": supports_body_system,
            "supported_output_modes": supported_modes,
            "read_depth_stats": data.read_depth_stats,
        },
    )
    return output
