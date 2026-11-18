"""Keep every shipped training YAML on the current modality and catalog recipe."""

from pathlib import Path

import pytest
import yaml

from astropt3.configuration_astropt3 import (
    DEFAULT_MODALITIES,
    LEGACY_SOUTH_MODALITIES,
    AstroPT3Config,
)
from astropt3.data.catalog_stream import CatalogStream
from astropt3.modalities import ModalityRegistry

ASTRO_DIR = Path(__file__).resolve().parents[1]
CONTROL = ASTRO_DIR / "configs/controls/astropt3-70m-north.yaml"
CONFIGS = sorted((ASTRO_DIR / "configs/nanotron").glob("*.yaml"))


REFERENCE = ASTRO_DIR / "configs/nanotron/astropt3-70m.yaml"


@pytest.mark.parametrize("backend", ["hf", "nanotron"])
@pytest.mark.parametrize(("overrides", "message"), [
    ({"band": "des-r"}, "legacy_north_g.*legacy_north.*des-g"),
    ({"source": "legacy_south"}, "legacy_north_g.*legacy_north.*des-g"),
    ({"record_keys": [""]}, "nonempty record key"),
])
def test_invalid_fixed_band_config_is_rejected(backend, overrides, message):
    config_cls = AstroPT3Config if backend == "hf" else pytest.importorskip("nanotron.config").AstroPT3Config
    modality = {**DEFAULT_MODALITIES[0], **overrides}
    with pytest.raises(ValueError, match=message):
        config_cls(modalities=[modality], vocab_size=256)


@pytest.mark.parametrize("backend", ["hf", "nanotron"])
@pytest.mark.parametrize("name, sigma", [
    ("legacy_north_g", -1.0), ("legacy_north_g", float("nan")),
    ("legacy_north_g", float("inf")), ("Z", 0.1),
])
def test_invalid_pre_flow_noise(backend, name, sigma):
    config_cls = AstroPT3Config if backend == "hf" else pytest.importorskip("nanotron.config").AstroPT3Config
    modality = next(m for m in DEFAULT_MODALITIES if m["name"] == name)
    with pytest.raises(ValueError, match="pre_flow_noise_max"):
        config_cls(modalities=[{**modality, "pre_flow_noise_max": sigma}], vocab_size=256)


def test_catalog_cannot_feed_south_flux_to_north_marker():
    recipe = yaml.safe_load(REFERENCE.read_text())
    spec = recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    modality = {**DEFAULT_MODALITIES[0], "record_keys": ["legacy_south_image"]}
    with pytest.raises(ValueError, match="legacy_north_g.*source disagrees with its image decoder"):
        CatalogStream(spec, AstroPT3Config(modalities=[modality]))


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
    # Noise amplitude is a training setting, not part of the data contract.
    assert [
        {key: value for key, value in m.items() if key != "pre_flow_noise_max"}
        for m in model["modalities"][: len(expected)]
    ] == expected
    assert ModalityRegistry(model["modalities"][len(expected) :]).to_list() == (
        ModalityRegistry(LEGACY_SOUTH_MODALITIES).to_list()
    )
    assert model["vocab_size"] == control["model"]["model_config"]["vocab_size"] == 256
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


def test_70m_recipe_uses_pre_flow_noise_on_images_and_spectra():
    config = yaml.safe_load(REFERENCE.read_text())["model"]["model_config"]
    assert config["jetformer_noise_min"] == config["jetformer_noise_max"] == 0.0
    assert {m["name"]: m["pre_flow_noise_max"] for m in config["modalities"] if "pre_flow_noise_max" in m} == {
        **{f"legacy_north_{band}": 0.4 for band in "grz"},
        **{f"hsc_{band}": 0.35 for band in "grizy"},
        "galaxies": 0.04, "spectra": 0.1, "sdss_spectra": 0.1,
    }
