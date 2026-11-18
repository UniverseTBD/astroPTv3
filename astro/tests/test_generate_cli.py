"""Generation selects templates through the configured upstream-backed stream."""

import copy
import sys
from pathlib import Path

import pytest
import yaml

from astropt3.data.nanotron_loader import hf_config_from_modalities

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate as cli  # pyright: ignore[reportMissingImports] -- scripts added above


@pytest.fixture
def recipe():
    path = Path(__file__).resolve().parents[1] / "configs/nanotron/astropt3-70m.yaml"
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("count", [1, 2, 4])
def test_templates_use_selected_catalogs_and_close_after_one_draw(monkeypatch, recipe, count):
    # Start with SDSS so single-modality generation cannot fall back to Legacy.
    modalities = {m["name"]: m for m in recipe["model"]["model_config"]["modalities"]}
    names = ["sdss_spectra", "spectra", "hsc_g", "legacy_north_g"]
    modalities = [modalities[name] for name in names[:count]]
    config = hf_config_from_modalities(modalities)
    keys = [modality["record_keys"][0] for modality in modalities]
    spec = recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    available = spec["anchors"] + spec["catalogs"]
    selected = [next(cat for cat in available if cat["record_key"] == key).copy() for key in keys]
    spec["anchors"] = [{**selected[0], "draw_weight": 1}]
    spec["catalogs"] = [{k: v for k, v in cat.items() if k != "draw_weight"} for cat in selected[1:]]
    spec["min_catalogs"] = min(count, 3)
    before = copy.deepcopy(spec)
    sparse = {"object_id": "sparse", keys[0]: {}}
    full = {"object_id": "full", **dict.fromkeys(keys, {})}
    closed = []

    def draws(stream, **kwargs):
        assert [cat["record_key"] for cat in stream.spec["anchors"] + stream.spec["catalogs"]] == keys
        assert kwargs == {"seed": 17, "rank": 0, "worker_id": 0}
        try:
            yield 0, iter([sparse, full])
            raise AssertionError("generation fetched a second draw")
        finally:
            closed.append(True)

    monkeypatch.setattr(cli.CatalogStream, "draws", draws)
    records = cli._draw_live_records(config, spec, [0, 1], 17)
    assert records == ([sparse, full] if count == 1 else [full, sparse])
    assert closed == [True]
    assert spec == before
    with pytest.raises(ValueError, match="index 2"):
        cli._draw_live_records(config, spec, [2], 17)
    assert closed == [True, True]


@pytest.mark.parametrize("rows", [[], [-1], [0, -2]])
def test_invalid_template_indices_fail_before_opening_catalogs(rows):
    with pytest.raises(ValueError, match="nonnegative"):
        cli._draw_live_records(None, {}, rows, 0)


@pytest.mark.parametrize("invalid", ["missing", "legacy", "multiple", "split"])
def test_cli_rejects_old_or_ambiguous_recipes_before_loading_checkpoint(monkeypatch, tmp_path, recipe, invalid):
    from transformers import AutoConfig

    if invalid == "missing":
        del recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    elif invalid == "legacy":
        recipe["data_stages"][0]["data"]["dataset"]["crossmatch_desi"] = False
    elif invalid == "split":
        recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"]["split"] = "typo"
    else:
        recipe["data_stages"] *= 2
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    output = tmp_path / "output"
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args: pytest.fail("checkpoint loaded"))
    monkeypatch.setattr(sys, "argv", ["generate.py", "--checkpoint", "unused", "--config-file", str(path), "--out", str(output)])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("configured,override,expected", [
    ("train", [], "validation"), ("train", ["--split", "train"], "train"), (None, [], None),
])
def test_cli_forwards_explicit_stream_to_template_selection(monkeypatch, tmp_path, recipe, configured, override, expected):
    from transformers import AutoConfig

    from astropt3.eval import samples

    training_stream = recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    if configured is None:
        training_stream.pop("split", None)
    else:
        training_stream["split"] = configured
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    config = hf_config_from_modalities(recipe["model"]["model_config"]["modalities"])
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args: config)
    records = [{"object_id": "template"}]

    def draw(actual_config, spec, rows, seed):
        assert actual_config is config
        assert spec == (training_stream if expected is None else {**training_stream, "split": expected})
        assert rows == [1, 0] and seed == 17
        return records

    def sample(checkpoint, actual_records, **kwargs):
        assert checkpoint == "unused" and actual_records is records
        assert kwargs["band"] == "hsc_g"
        return {}

    monkeypatch.setattr(cli, "_draw_live_records", draw)
    monkeypatch.setattr(samples, "sample_checkpoint", sample)
    monkeypatch.setattr(sys, "argv", ["generate.py", "--checkpoint", "unused", "--config-file", str(path),
                                     "--rows", "1,0", "--stream-seed", "17", "--band", "hsc_g",
                                     *override, "--out", str(tmp_path / "output")])
    cli.main()
