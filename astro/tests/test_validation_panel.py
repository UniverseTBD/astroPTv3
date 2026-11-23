"""The held-out sample is independent of training settings and DP sharding."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from astropt3.data import validation_panel as panel
from astropt3.data.nanotron_loader import hf_config_from_modalities
from legacy_fixture import make_record


@pytest.fixture
def panel_config():
    config_path = Path(__file__).parents[1] / "configs/nanotron/astropt3-70m.yaml"
    recipe = yaml.safe_load(config_path.read_text())
    return hf_config_from_modalities(recipe["model"]["model_config"]["modalities"])


def test_validation_panel_replays_and_keeps_every_target(monkeypatch, tmp_path, panel_config):
    records = [make_record(i, image_only_fraction=0.0) for i in range(3)]
    closed = []

    def draws(self, *, seed, rank, worker_id, rows_per_draw):
        assert self.spec["split"] == "validation"
        assert (seed, rank, worker_id) == (panel.PANEL_SEED, 0, 0)
        assert rows_per_draw > 1
        try:
            yield 9, records[:2]
            yield 42, records[1:]
            pytest.fail("read beyond the fixed panel")
        finally:
            closed.append(True)

    monkeypatch.setattr(panel.CatalogStream, "draws", draws)
    ids, shards = panel.load_validation_panel(panel_config, 3, 500, 2, 1)
    other_ids, other_shards = panel.load_validation_panel(panel_config, 3, 500, 1, 4)
    assert ids == other_ids == [record["object_id"] for record in records]
    assert closed == [True, True]
    assert [batch["input_ids"].shape[0] for batch in shards[0]] == [2, 1]
    assert [len(shard) for shard in other_shards] == [1, 1, 1, 0]
    def values(by_rank, name):
        return torch.cat([batch[f"{name}_values"] for shard in by_rank for batch in shard])

    for name in ("legacy_north_g", "spectra", "ebv"):
        torch.testing.assert_close(values(shards, name), values(other_shards, name), rtol=0, atol=0)
        assert values(shards, name).shape[0] > 0
    for row_idx, shard in enumerate(other_shards[:3]):
        longer = shards[0][row_idx // 2]["input_ids"][row_idx % 2]
        shorter = shard[0]["input_ids"][0]
        torch.testing.assert_close(longer[:len(shorter)], shorter)

    reference = tmp_path / "first" / panel.PANEL_FILENAME
    destination = tmp_path / "later" / panel.PANEL_FILENAME
    panel.save_validation_manifest(reference, ids)
    panel.save_validation_manifest(destination, ids, reference=reference)
    assert json.loads(destination.read_text()) == {"seed": panel.PANEL_SEED, "object_ids": ids}
    with pytest.raises(ValueError, match="does not match"):
        panel.save_validation_manifest(destination, ids[::-1], reference=reference)


def test_validation_panel_fails_if_too_few_unique_objects(monkeypatch, panel_config):
    def draws(self, **kwargs):
        for epoch in range(2):
            yield epoch, [make_record(0)]

    monkeypatch.setattr(panel.CatalogStream, "draws", draws)
    with pytest.raises(ValueError, match="only 1 of 2 unique objects"):
        panel.load_validation_panel(panel_config, 2, 500, 1, 1)
