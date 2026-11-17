"""CPU-only schema/save tests, also runnable in the optional Nanotron environment."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from legacy_fixture import TINY_CATALOG_STREAM

nanotron_config = pytest.importorskip("nanotron.config")
AstroPT3StreamingDatasetsArgs = pytest.importorskip(
    "nanotron.config.astropt3_config"
).AstroPT3StreamingDatasetsArgs


@pytest.mark.parametrize("factor", [1, 4])
def test_echo_factor_round_trips_nanotron_yaml(tmp_path, factor):
    baseline = Path(__file__).resolve().parents[1] / "configs/nanotron/astropt3-70m.yaml"
    raw = yaml.safe_load(baseline.read_text())
    raw["data_stages"][0]["data"]["dataset"]["echo_factor"] = factor
    config = nanotron_config.get_config_from_dict(raw)
    path = tmp_path / "config.yaml"
    config.save_as_yaml(path)
    restored = nanotron_config.get_config_from_file(path)
    assert restored.data_stages[0].data.dataset.echo_factor == factor
    assert restored.data_stages[0].data.dataset.catalog_stream == raw["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    assert len(restored.model.model_config.modalities) == 14


def test_explicit_dataset_config_defaults_to_no_echoing():
    args = AstroPT3StreamingDatasetsArgs(catalog_stream=TINY_CATALOG_STREAM)
    assert args.echo_factor == 1
    assert args.catalog_stream == TINY_CATALOG_STREAM


@pytest.mark.parametrize("kwargs", [
    {}, {"catalog_stream": None}, {"catalog_stream": {}},
    {"catalog_stream": {"catalogs": []}}, {"catalog_stream": []},
])
def test_nanotron_requires_explicit_catalogs(kwargs):
    with pytest.raises(ValueError, match="catalog_stream.catalogs"):
        AstroPT3StreamingDatasetsArgs(**kwargs)


@pytest.mark.parametrize("selected", [False, True])
def test_nanotron_rejects_retired_crossmatch_option(selected):
    with pytest.raises(TypeError, match="crossmatch_desi"):
        AstroPT3StreamingDatasetsArgs(
            catalog_stream=TINY_CATALOG_STREAM, crossmatch_desi=selected
        )


def test_default_recipe_round_trips_catalog_stream(tmp_path):
    baseline = Path(__file__).resolve().parents[1] / "configs/nanotron/astropt3-70m.yaml"
    raw = yaml.safe_load(baseline.read_text())
    config = nanotron_config.get_config_from_dict(raw)
    path = tmp_path / "config.yaml"
    config.save_as_yaml(path)
    restored = nanotron_config.get_config_from_file(path)
    dataset = restored.data_stages[0].data.dataset
    assert dataset.echo_factor == 4
    assert dataset.catalog_stream == raw["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    assert len(restored.model.model_config.modalities) == 14


def test_derived_gpu_smoke_keeps_tiny_training_contract(tmp_path):
    from test_nanotron_gpu import smoke_config

    raw = smoke_config(SimpleNamespace(config_cls=nanotron_config.AstroPT3Config), tmp_path)
    config = nanotron_config.get_config_from_dict(yaml.safe_load(yaml.safe_dump(raw)))
    model = config.model.model_config
    assert (model.hidden_size, model.num_hidden_layers, model.jetformer_flow_hidden) == (64, 4, 128)
    assert len(model.modalities) == 14 and not model.use_cache
    assert not model._fused_rms_norm and not model._fused_rotary_emb
    data = config.data_stages[0].data
    assert data.num_loading_workers == 0 and data.dataset.echo_factor == 1
    assert (config.tokens.train_steps, config.tokens.micro_batch_size, config.tokens.sequence_length) == (50, 4, 896)
    scheduler = config.optimizer.learning_rate_scheduler
    assert (scheduler.lr_warmup_steps, scheduler.lr_decay_steps) == (10, 40)
    assert config.optimizer.zero_stage == 0
    assert config.parallelism.dp == 1
    assert config.checkpoints.checkpoint_interval == 50
    assert config.checkpoints.checkpoint_schedule is None


@pytest.mark.parametrize("factor", [0, -1, True, False, 1.5, "2", None])
def test_nanotron_rejects_invalid_echo_factor(factor):
    with pytest.raises(ValueError, match="positive integer"):
        AstroPT3StreamingDatasetsArgs(echo_factor=factor)
