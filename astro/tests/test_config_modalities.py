"""Keep every shipped training YAML on the current modality and catalog recipe."""

from pathlib import Path

import pytest
import yaml

from astropt3.configuration_astropt3 import LEGACY_SOUTH_MODALITIES
from astropt3.modalities import ModalityRegistry

ASTRO_DIR = Path(__file__).resolve().parents[1]
CONTROL = ASTRO_DIR / "configs/controls/astropt3-70m-north.yaml"
CONFIGS = sorted((ASTRO_DIR / "configs/nanotron").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: str(path.relative_to(ASTRO_DIR)))
def test_yaml_uses_current_modalities(path):
    control = yaml.safe_load(CONTROL.read_text())
    expected = control["model"]["model_config"]["modalities"]
    images = {m["name"]: m for m in expected if m["family"] == "image"}
    assert set(images) == {
        "legacy_north_g", "legacy_north_r", "legacy_north_z",
        "hsc_g", "hsc_r", "hsc_i", "hsc_z", "hsc_y", "galaxies",
    }
    rgb = images["galaxies"]
    assert rgb["image_format"] == "rgb" and "band" not in rgb
    assert rgb["token_ids"] == [23, 24, 25]
    assert (rgb["input_size"], rgb["patch_size"], rgb["max_positions"]) == (768, 16, 256)
    raw = yaml.safe_load(path.read_text())
    assert Path(raw["checkpoints"]["checkpoints_path"]).name == f"{path.stem}-dual-anchor"
    model = raw["model"]["model_config"]
    assert model["modalities"][: len(expected)] == expected
    assert ModalityRegistry(model["modalities"][len(expected) :]).to_list() == (
        ModalityRegistry(LEGACY_SOUTH_MODALITIES).to_list()
    )
    assert model["vocab_size"] == control["model"]["model_config"]["vocab_size"] == 200
    assert model["vocab_size"] % raw["parallelism"]["tp"] == 0
    worst_case = 1 + sum(m["max_positions"] + 2 for m in model["modalities"])
    assert worst_case <= raw["tokens"]["sequence_length"]
    original = control["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    assert original["split"] == "train"
    for stage in raw["data_stages"]:
        stream = stage["data"]["dataset"]["catalog_stream"]
        north, south = stream["anchors"]
        assert [north] == original["anchors"] and north["draw_weight"] == 1
        assert south["draw_weight"] == 2 and south["record_key"] == "legacy_south_image"
        assert stream == {**original, "anchors": [north, south]}
