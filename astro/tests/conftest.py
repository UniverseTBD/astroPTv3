import os
from pathlib import Path

import pytest
import torch
from legacy_fixture import make_record

from astropt3.config_io import load_model_config
from astropt3.configuration_astropt3 import DEFAULT_MODALITIES, AstroPT3Config
from astropt3.data.packing import ObjectSequencer, PackedCollator

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "model"

# gpu-marked tests spawn nanotron trainer subprocesses ({**os.environ}); a
# test run must never require (or spam) a wandb login — real runs set
# WANDB_MODE=online explicitly in their launcher
os.environ.setdefault("WANDB_MODE", "disabled")


@pytest.fixture(scope="session")
def tiny_config():
    config, _ = load_model_config(CONFIG_DIR / "test-tiny.yaml")
    # Scalar regression coverage is independent of the four-survey YAML recipes.
    return AstroPT3Config(**{**config.to_dict(), "modalities": DEFAULT_MODALITIES})


@pytest.fixture
def categorical_label(monkeypatch):
    from astropt3.data.scalar_registry import CATEGORICAL

    monkeypatch.setitem(CATEGORICAL, "test_label", 4)
    return {
        "name": "test_label", "family": "scalar", "source": "legacy", "record_keys": ["test_label"],
        "token_ids": [64, 65, 66], "input_size": 4, "patch_size": 1, "pos_type": "index",
        "max_positions": 1, "categories": 4,
    }


@pytest.fixture(scope="session")
def sequencer(tiny_config):
    return ObjectSequencer(tiny_config)


@pytest.fixture(scope="session")
def collator(tiny_config):
    return PackedCollator(tiny_config, seq_len=896)


@pytest.fixture(scope="session")
def full_record():
    # seed 3 has both image and spectrum
    record = make_record(3)
    assert "spectrum" in record
    return record


@pytest.fixture(scope="session")
def image_only_record():
    # scan for a deterministic image-only record
    for i in range(64):
        record = make_record(i)
        if "spectrum" not in record:
            return record
    raise RuntimeError("no image-only record found in first 64 seeds")


@pytest.fixture(scope="session")
def spectrum_only_record():
    # ADR 0005: a non-crossmatched DESI row — spectrum, no image
    record = make_record(3, image_only_fraction=0.0, spectrum_only_fraction=1.0)
    assert "image" not in record and "spectrum" in record
    return record


@pytest.fixture(scope="session")
def tiny_model(tiny_config):
    torch.manual_seed(0)
    from astropt3 import AstroPT3Model

    model = AstroPT3Model(tiny_config)
    model.eval()
    return model
