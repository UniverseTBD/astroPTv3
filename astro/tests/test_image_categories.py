"""Source/filter identity and token budget for the training recipe."""

from pathlib import Path

import pytest
import torch
import yaml

from astropt3.data.band_registry import physical_normalize
from astropt3.data.nanotron_loader import (
    flatten_packed_batch,
    hf_config_from_modalities,
)
from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.tokenization import PAD_ID


@pytest.fixture(scope="module")
def config():
    path = Path(__file__).resolve().parents[1] / "configs/nanotron/astropt3-70m.yaml"
    model = yaml.safe_load(path.read_text())["model"]["model_config"]
    return hf_config_from_modalities(model["modalities"], vocab_size=model["vocab_size"])


def image(bands, side):
    return {
        "band": bands,
        "flux": torch.tensor([0.03, 0.02, 0.01, 0.04, 0.05][: len(bands)])
        .reshape(-1, 1, 1)
        .expand(-1, side, side),
    }


def test_explicit_spans_follow_labels_not_channel_order(config):
    north = image(["des-z", "des-r", "des-g"], 152)
    hsc = image(["hsc-g", "hsc-r", "hsc-i", "hsc-z", "hsc-y"], 96)
    obj = ObjectSequencer(config).build({"image": north, "hsc_image": hsc})
    assert set(obj.masks) == {
        "legacy_north_g", "legacy_north_r", "legacy_north_z",
        "hsc_g", "hsc_r", "hsc_i", "hsc_z", "hsc_y",
    }
    registry = config.modality_registry()
    for name in obj.masks:
        mod = registry.get_config(name)
        assert mod.token_ids is not None and mod.band is not None
        begin, placeholder, end = mod.token_ids
        (indices,) = obj.masks[name].nonzero(as_tuple=True)
        assert obj.input_ids[indices[0] - 1] == begin
        assert (obj.input_ids[indices] == placeholder).all()
        assert obj.input_ids[indices[-1] + 1] == end
        assert torch.equal(obj.positions[name], torch.arange(144))
        payload = north if mod.source == "legacy_north" else hsc
        channel = payload["band"].index(mod.band)
        flux = payload["flux"][channel : channel + 1, :1, :1]
        expected = physical_normalize(flux, [mod.band], divisor=config.image_norm_divisor)
        assert torch.allclose(obj.values[name], expected.reshape(1, 1).expand(144, 64))
    assert len(set(obj.input_ids.tolist())) == 1 + 3 * len(obj.masks)


def test_image_token_budget_and_packed_utilization(config):
    sequencer = ObjectSequencer(config)
    sources = {
        "image": image(["des-g", "des-r", "des-z"], 152),
        "legacy_south_image": image(["des-g", "des-r", "des-i", "des-z"], 160),
        "hsc_image": image(["hsc-g", "hsc-r", "hsc-i", "hsc-z", "hsc-y"], 96),
        "galaxy_image": image(["rgb-r", "rgb-g", "rgb-b"], 256),
    }
    by_source = [sequencer.build({key: payload}) for key, payload in sources.items()]
    rgb = by_source[-1]
    assert set(rgb.masks) == {"galaxies"}
    assert rgb.values["galaxies"].shape == (256, 768)
    assert torch.equal(rgb.values["galaxies"][0, :3], sources["galaxy_image"]["flux"][:, 0, 0])
    objects = [*by_source, sequencer.build(sources)]
    # Each calibrated band adds 144 patches + 2 delimiters; RGB adds 256 + 2 once.
    assert [len(obj) for obj in objects] == [439, 585, 731, 259, 2011]
    collator = PackedCollator(config, seq_len=4096)
    batch = collator(objects * 3)
    ids = batch["input_ids"]
    assert ids.shape == (3, 4096)
    assert (ids != PAD_ID).sum().item() / ids.numel() == pytest.approx(4025 / 4096)
    flat = flatten_packed_batch(collator([by_source[0]]), config, 4096)
    assert flat["legacy_north_g_values"].shape == (144, 64)
    assert not flat["legacy_south_g_mask"].any()
    assert flat["legacy_south_g_values"].shape == (0, 64)
