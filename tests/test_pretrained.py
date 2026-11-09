from __future__ import annotations

import inspect
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import zipfile

import pandas as pd
import pytest
import torch
import yaml

from mbbench.pretrained import (
    OFFICIAL_MODEL_URLS,
    download_model,
    prompt_install_directory,
    resolve_model_directory,
)
from mbbench.models import download_model as public_download_model


@pytest.mark.parametrize("archive_format", ["tar.gz", "zip"])
def test_download_installs_complete_official_archive(tmp_path, archive_format):
    source = tmp_path / "source" / "ae"
    source.mkdir(parents=True)
    (source / "model.yaml").write_text(
        yaml.safe_dump(
            {
                "model_type": "ae",
                "condition_fields": ["Phenotype_Tokens", "Body_Site_Coarse"],
                "supported_output_modes": ["relative"],
            }
        ),
        encoding="utf-8",
    )
    (source / "vocabularies.json").write_text(
        json.dumps(
            {
                "Phenotype_Tokens": {"tokens": ["Healthy"]},
                "Body_Site_Coarse": {"values": ["stool_feces"]},
            }
        ),
        encoding="utf-8",
    )
    torch.save({}, source / "tokenizer.pt")
    torch.save({}, source / "diffusion.pt")
    pd.DataFrame({"taxon": ["g1"]}).to_csv(source / "taxa.csv", index=False)
    archive = tmp_path / "content"
    if archive_format == "zip":
        with zipfile.ZipFile(archive, "w") as handle:
            for path in source.iterdir():
                handle.write(path, arcname=f"ae/{path.name}")
    else:
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(source, arcname="ae")
    installed = download_model(url=archive.as_uri(), target_dir=tmp_path / "installed")
    assert installed.model_type == "ae"


def test_model_download_rejects_zip_path_traversal(tmp_path):
    archive = tmp_path / "content"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../unexpected.txt", "unsafe")
    target = tmp_path / "installed"
    with pytest.raises(ValueError, match="unsafe path"):
        download_model(url=archive.as_uri(), target_dir=target)
    assert not target.exists()


def test_model_download_rejects_wrong_official_checksum(tmp_path, monkeypatch):
    archive = tmp_path / "wrong-content"
    archive.write_bytes(b"unexpected response")
    monkeypatch.setattr("mbbench.pretrained.urlopen", lambda url, timeout: archive.open("rb"))
    target = tmp_path / "installed"
    with pytest.raises(ValueError, match="SHA-256"):
        download_model(url=OFFICIAL_MODEL_URLS["ae"], target_dir=target)
    assert not target.exists()


def test_install_directory_prompt_uses_default_or_user_path(tmp_path, monkeypatch):
    default = tmp_path / "default"
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert prompt_install_directory(default) == default.resolve()
    selected = tmp_path / "selected"
    monkeypatch.setattr("builtins.input", lambda _: str(selected))
    assert prompt_install_directory(default) == selected.resolve()


def test_model_resolver_uses_cache_and_never_prompts_noninteractive(
    tmp_path, monkeypatch
):
    cached = tmp_path / "cached"
    cached.mkdir()
    monkeypatch.setattr("mbbench.pretrained.model_cache_dir", lambda _: cached)
    monkeypatch.setattr(
        "mbbench.pretrained.ModelBundle.load",
        lambda path: SimpleNamespace(root=Path(path).resolve()),
    )
    assert resolve_model_directory("ae", None, False) == cached.resolve()

    missing = tmp_path / "missing"
    monkeypatch.setattr("mbbench.pretrained.model_cache_dir", lambda _: missing)
    monkeypatch.setitem(
        __import__(
            "mbbench.pretrained", fromlist=["OFFICIAL_MODEL_URLS"]
        ).OFFICIAL_MODEL_URLS,
        "ae",
        "https://example.invalid/model.tar.gz",
    )
    monkeypatch.setattr("mbbench.pretrained.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        "builtins.input", lambda _: pytest.fail("noninteractive resolver read stdin")
    )
    with pytest.raises(FileNotFoundError, match="download-model"):
        resolve_model_directory("ae", None, False)


def test_model_resolver_confirms_download_then_asks_install_directory(
    tmp_path, monkeypatch
):
    default = tmp_path / "default"
    selected = tmp_path / "selected"
    monkeypatch.setattr("mbbench.pretrained.model_cache_dir", lambda _: default)
    monkeypatch.setitem(
        __import__(
            "mbbench.pretrained", fromlist=["OFFICIAL_MODEL_URLS"]
        ).OFFICIAL_MODEL_URLS,
        "ae",
        "https://example.invalid/model.tar.gz",
    )
    monkeypatch.setattr("mbbench.pretrained.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("mbbench.pretrained.sys.stdout.isatty", lambda: True)
    answers = iter(["y", str(selected)])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    calls = []

    def fake_download(*, url, target_dir, timeout=120):
        calls.append((url, Path(target_dir)))
        return SimpleNamespace(root=Path(target_dir))

    monkeypatch.setattr("mbbench.pretrained.download_model", fake_download)
    assert resolve_model_directory("ae", None, False) == selected.resolve()
    assert calls == [("https://example.invalid/model.tar.gz", selected.resolve())]


def test_public_download_returns_a_ready_model(tmp_path, monkeypatch):
    target = tmp_path / "model"
    calls = []
    monkeypatch.setitem(
        __import__(
            "mbbench.models", fromlist=["OFFICIAL_MODEL_URLS"]
        ).OFFICIAL_MODEL_URLS,
        "ae",
        "https://example.test/ae.tar.gz",
    )
    monkeypatch.setattr("mbbench.models.sys.stdin.isatty", lambda: False)
    monkeypatch.setattr(
        "mbbench.models.download_model_archive",
        lambda *, url, target_dir: SimpleNamespace(root=Path(target_dir)),
    )

    def load(path, device):
        calls.append((Path(path), device))
        return SimpleNamespace(root=Path(path), device=device)

    monkeypatch.setattr("mbbench.models.Simulator.from_directory", load)
    result = public_download_model("ae", install_dir=target, device="cpu")
    assert result.root == target.resolve()
    assert calls == [(target.resolve(), "cpu")]
    assert "use_body_system" not in inspect.signature(public_download_model).parameters
