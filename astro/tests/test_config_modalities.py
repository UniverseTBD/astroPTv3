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


def test_catalog_cannot_feed_south_flux_to_north_marker():
    recipe = yaml.safe_load((ASTRO_DIR / "configs/nanotron/astropt3-70m.yaml").read_text())
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
    assert (rgb["input_size"], rgb["patch_size"], rgb["max_positions"], rgb["image_resize"]) == (192, 8, 256, 128)
    raw = yaml.safe_load(path.read_text())
    assert Path(raw["checkpoints"]["checkpoints_path"]).name == f"{path.stem}-dual-anchor"
    model = raw["model"]["model_config"]
    # Noise amplitude is a training setting, not part of the data contract.
    definitions = [
        {key: value for key, value in m.items() if key != "pre_flow_noise_max"}
        for m in model["modalities"]
    ]
    assert len(definitions) == 57
    assert {"gz10_label", "gz10_images"}.isdisjoint(m["name"] for m in definitions)
    assert definitions[: len(expected)] == expected
    south = definitions[len(expected) :]
    assert ModalityRegistry(south[: len(LEGACY_SOUTH_MODALITIES)]).to_list() == (
        ModalityRegistry(LEGACY_SOUTH_MODALITIES).to_list()
    )
    assert [(m["name"], m["source"], m["record_keys"]) for m in south[len(LEGACY_SOUTH_MODALITIES) :]] == [
        ("legacy_south_ebv", "legacy_south", ["legacy_south_ebv"]),
        ("legacy_south_photometry", "legacy_south", [f"legacy_south_flux_{band}" for band in "griz"]),
    ]
    assert model["vocab_size"] == control["model"]["model_config"]["vocab_size"] == 256
    assert model["vocab_size"] % raw["parallelism"]["tp"] == 0
    config = AstroPT3Config(**model)
    # Each record can carry at most one Legacy anchor's bands and scalars.
    for absent in ({"legacy_north", "legacy"}, {"legacy_south"}):
        worst_case = 1 + sum(m["max_positions"] + 2 for m in model["modalities"] if m["source"] not in absent)
        assert worst_case <= raw["tokens"]["sequence_length"]
    original = control["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    assert original["split"] == "train"
    assert "gz10" not in {catalog["name"] for catalog in original["catalogs"]}
    sdss = next(catalog for catalog in original["catalogs"] if catalog["name"] == "sdss")
    assert sdss["repo"] == "UniverseTBD/mmu_sdss_dr17@091f56aa923837b7672586f6076fda490a6062dc"
    for stage in raw["data_stages"]:
        stream = stage["data"]["dataset"]["catalog_stream"]
        north, south = stream["anchors"]
        assert [north] == original["anchors"] and north["draw_weight"] == 1
        assert south["draw_weight"] == 2 and south["record_key"] == "legacy_south_image"
        assert stream == {**original, "anchors": [north, south]}
        CatalogStream(stream, config)


def test_70m_recipe_combines_pre_flow_and_constant_latent_noise():
    config = yaml.safe_load((ASTRO_DIR / "configs/nanotron/astropt3-70m.yaml").read_text())["model"]["model_config"]
    assert config["jetformer_noise_min"] == config["jetformer_noise_max"] == 0.1
    assert {m["name"]: m["pre_flow_noise_max"] for m in config["modalities"] if "pre_flow_noise_max" in m} == {
        "legacy_north_g": 0.21, "legacy_north_r": 0.37, "legacy_north_z": 1.08,
        "legacy_south_g": 0.15, "legacy_south_r": 0.14, "legacy_south_i": 0.45, "legacy_south_z": 0.64,
        "hsc_g": 0.15, "hsc_r": 0.20, "hsc_i": 0.32, "hsc_z": 0.49, "hsc_y": 0.93,
        "galaxies": 0.20, "spectra": 0.1, "sdss_spectra": 0.1,
    }
