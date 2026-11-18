"""Keep every shipped training YAML on the current modality and catalog recipe."""

from pathlib import Path

import pytest
import yaml

from astropt3.configuration_astropt3 import LEGACY_SOUTH_IMAGES

ASTRO_DIR = Path(__file__).resolve().parents[1]
CONTROL = ASTRO_DIR / "configs/controls/astropt3-70m-north.yaml"
CONFIGS = sorted((ASTRO_DIR / "configs/nanotron").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: str(path.relative_to(ASTRO_DIR)))
def test_yaml_uses_current_modalities(path):
    control = yaml.safe_load(CONTROL.read_text())
    expected = control["model"]["model_config"]["modalities"]
    assert [modality["name"] for modality in expected][:5] == [
        "images", "spectra", "sdss_spectra", "hsc_images", "galaxies"
    ]
    raw = yaml.safe_load(path.read_text())
    assert Path(raw["checkpoints"]["checkpoints_path"]).name == f"{path.stem}-dual-anchor"
    model = raw["model"]["model_config"]
    assert model["modalities"][:-1] == expected
    assert model["modalities"][-1] == LEGACY_SOUTH_IMAGES
    assert model["vocab_size"] == control["model"]["model_config"]["vocab_size"] == 64
    original = control["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    assert original["split"] == "train"
    for stage in raw["data_stages"]:
        stream = stage["data"]["dataset"]["catalog_stream"]
        north, south = stream["anchors"]
        assert [north] == original["anchors"] and north["draw_weight"] == 1
        assert south["draw_weight"] == 2 and south["record_key"] == "legacy_south_image"
        assert stream == {**original, "anchors": [north, south]}
