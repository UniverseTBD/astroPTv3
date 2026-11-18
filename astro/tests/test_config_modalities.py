"""Keep every shipped training YAML on the current modality and catalog recipe."""

from pathlib import Path

import pytest
import yaml

ASTRO_DIR = Path(__file__).resolve().parents[1]
REFERENCE = ASTRO_DIR / "configs/nanotron/astropt3-70m.yaml"
CONFIGS = sorted((ASTRO_DIR / "configs/nanotron").glob("*.yaml"))


def modality_definitions(recipe):
    # Noise amplitude is a training setting, not part of the data contract.
    return [
        {key: value for key, value in modality.items() if key != "pre_flow_noise_max"}
        for modality in recipe["model"]["model_config"]["modalities"]
    ]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: str(path.relative_to(ASTRO_DIR)))
def test_yaml_uses_current_modalities(path):
    reference = yaml.safe_load(REFERENCE.read_text())
    expected = modality_definitions(reference)
    assert [modality["name"] for modality in expected][:5] == [
        "images", "spectra", "sdss_spectra", "hsc_images", "galaxies"
    ]
    raw = yaml.safe_load(path.read_text())
    assert modality_definitions(raw) == expected
    expected_stream = reference["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    for stage in raw["data_stages"]:
        assert stage["data"]["dataset"]["catalog_stream"] == expected_stream


def test_70m_recipe_uses_pre_flow_noise_on_images_and_spectra():
    config = yaml.safe_load(REFERENCE.read_text())["model"]["model_config"]
    assert config["jetformer_noise_min"] == config["jetformer_noise_max"] == 0.0
    assert {m["name"]: m["pre_flow_noise_max"] for m in config["modalities"] if "pre_flow_noise_max" in m} == {
        "images": 0.4, "hsc_images": 0.35, "galaxies": 0.04, "spectra": 0.1, "sdss_spectra": 0.1,
    }
