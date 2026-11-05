"""CPU-only schema/save tests, also runnable in the optional Nanotron environment."""

from pathlib import Path

import pytest
import yaml

nanotron_config = pytest.importorskip("nanotron.config")
AstroPT3StreamingDatasetsArgs = pytest.importorskip(
    "nanotron.config.astropt3_config"
).AstroPT3StreamingDatasetsArgs


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_echo_factor_round_trips_nanotron_yaml(tmp_path, factor):
    baseline = Path(__file__).resolve().parents[1] / "configs/nanotron/astropt3-70m-dp1.yaml"
    raw = yaml.safe_load(baseline.read_text())
    raw["data_stages"][0]["data"]["dataset"]["echo_factor"] = factor
    config = nanotron_config.get_config_from_dict(raw)
    path = tmp_path / "config.yaml"
    config.save_as_yaml(path)
    restored = nanotron_config.get_config_from_file(path)
    assert restored.data_stages[0].data.dataset.echo_factor == factor


def test_legacy_dataset_config_defaults_to_no_echoing():
    args = AstroPT3StreamingDatasetsArgs()
    assert args.echo_factor == 1
    assert args.catalog_stream is None


def test_four_modal_default_round_trips_catalog_stream(tmp_path):
    baseline = Path(__file__).resolve().parents[1] / "configs/nanotron/astropt3-70m-four-modal.yaml"
    raw = yaml.safe_load(baseline.read_text())
    config = nanotron_config.get_config_from_dict(raw)
    path = tmp_path / "config.yaml"
    config.save_as_yaml(path)
    restored = nanotron_config.get_config_from_file(path)
    dataset = restored.data_stages[0].data.dataset
    assert dataset.echo_factor == 4
    assert dataset.catalog_stream == raw["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    assert len(restored.model.model_config.modalities) == 4


@pytest.mark.parametrize("factor", [0, -1, True, False, 1.5, "2", None])
def test_nanotron_rejects_invalid_echo_factor(factor):
    with pytest.raises(ValueError, match="positive integer"):
        AstroPT3StreamingDatasetsArgs(echo_factor=factor)
