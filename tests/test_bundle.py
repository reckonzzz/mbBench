from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from mbbench.bundle import ModelBundle


def _write_bundle(path, *, supports_body_system=False):
    path.mkdir()
    config = {
        "model_type": "ae",
        "condition_fields": ["Phenotype_Tokens", "Body_Site_Coarse"],
        "supports_body_system": supports_body_system,
        "supported_output_modes": ["relative", "observed"],
    }
    (path / "model.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (path / "vocabularies.json").write_text(
        json.dumps(
            {
                "Phenotype_Tokens": {"tokens": ["<NULL>", "Healthy", "IBD"]},
                "Body_Site_Coarse": {"values": ["<UNK>", "stool_feces", "saliva"]},
            }
        ),
        encoding="utf-8",
    )
    torch.save({}, path / "tokenizer.pt")
    torch.save({}, path / "diffusion.pt")
    pd.DataFrame({"taxon": ["g1", "g2"]}).to_csv(path / "taxa.csv", index=False)
    pd.DataFrame(
        {
            "Phenotype_Tokens": ["Healthy"],
            "Body_Site_Coarse": ["stool_feces"],
            "Project_Group": ["project_1"],
            "raw_total_reads": [10000],
        }
    ).to_csv(path / "donors.csv", index=False)
    if supports_body_system:
        pd.DataFrame({"body_site": ["saliva"], "body_system": ["oral"]}).to_csv(
            path / "body_site_to_system.csv", index=False
        )


def test_model_bundle_lists_conditions(tmp_path):
    root = tmp_path / "model"
    _write_bundle(root)
    bundle = ModelBundle.load(root)
    assert bundle.condition_values("Phenotype_Tokens") == ["Healthy", "IBD"]
    assert bundle.condition_values("Body_Site_Coarse") == ["stool_feces", "saliva"]


def test_body_system_bundle_requires_mapping(tmp_path):
    root = tmp_path / "model"
    _write_bundle(root, supports_body_system=True)
    (root / "body_site_to_system.csv").unlink()
    with pytest.raises(ValueError, match="body_site_to_system"):
        ModelBundle.load(root)
