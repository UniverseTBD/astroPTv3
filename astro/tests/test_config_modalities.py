"""Keep every shipped model and training YAML on the current four surveys."""

from pathlib import Path

import pytest
import yaml

ASTRO_DIR = Path(__file__).resolve().parents[1]
REFERENCE = ASTRO_DIR / "configs/nanotron/astropt3-70m-four-modal.yaml"
CONFIGS = sorted((ASTRO_DIR / "configs").rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: str(path.relative_to(ASTRO_DIR)))
def test_yaml_uses_current_four_modalities(path):
    reference = yaml.safe_load(REFERENCE.read_text())
    expected = reference["model"]["model_config"]["modalities"]
    assert [modality["name"] for modality in expected] == [
        "images", "spectra", "sdss_spectra", "hsc_images"
    ]
    raw = yaml.safe_load(path.read_text())
    model = raw["model"]["model_config"] if "model" in raw else raw
    assert model.get("modalities") == expected
    assert model.get("loss_aggregation") == "family"
    for stage in raw.get("data_stages", []):
        catalogs = stage["data"]["dataset"]["catalog_stream"]["catalogs"]
        assert catalogs == reference["data_stages"][0]["data"]["dataset"]["catalog_stream"]["catalogs"]
