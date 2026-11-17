"""Keep every shipped training YAML on the current modality and catalog recipe."""

from pathlib import Path

import pytest
import yaml

ASTRO_DIR = Path(__file__).resolve().parents[1]
REFERENCE = ASTRO_DIR / "configs/nanotron/astropt3-70m.yaml"
CONFIGS = sorted((ASTRO_DIR / "configs/nanotron").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: str(path.relative_to(ASTRO_DIR)))
def test_yaml_uses_current_modalities(path):
    reference = yaml.safe_load(REFERENCE.read_text())
    expected = reference["model"]["model_config"]["modalities"]
    assert [modality["name"] for modality in expected][:5] == [
        "images", "spectra", "sdss_spectra", "hsc_images", "galaxies"
    ]
    raw = yaml.safe_load(path.read_text())
    assert raw["model"]["model_config"]["modalities"] == expected
    for stage in raw["data_stages"]:
        catalogs = stage["data"]["dataset"]["catalog_stream"]["catalogs"]
        assert catalogs == reference["data_stages"][0]["data"]["dataset"]["catalog_stream"]["catalogs"]
