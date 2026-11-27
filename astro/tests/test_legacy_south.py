"""Legacy DR10 South decoder, packed-token ABI and pinned Parquet canary."""

import io
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from astropt3 import AstroPT3Model
from astropt3.configuration_astropt3 import (
    DEFAULT_MODALITIES,
    LEGACY_SOUTH_MODALITIES,
    AstroPT3Config,
)
from astropt3.data.nanotron_loader import (
    decode_legacy_image,
    decode_legacy_south_image,
    flatten_packed_batch,
    regroup_micro_batch,
)
from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.tokenization import unpatchify_image


def _payload():
    shape = (4, 160, 160)
    return {
        "band": ["des-g", "des-r", "des-i", "des-z"],
        "flux": np.full(shape, np.arange(1, 5, dtype=np.float32)[:, None, None] * 0.01),
    }


@pytest.mark.parametrize("field,bad", [
    ("flux", np.zeros((3, 152, 152))), ("flux", None),
    ("band", ["des-g", "des-r", "des-z", "des-i"]),
    ("band", ["des-g", "des-r", "des-z", "des-z"]),
])
def test_rejects_drift(field, bad):
    with pytest.raises(ValueError, match=field):
        decode_legacy_south_image({**_payload(), field: bad})


def test_packed_south_fills_zeros_and_keeps_all_four_bands(tiny_config):
    image = _payload()
    image["flux"][0, 32, 32] = 0
    image["flux"][1, 32, 32] = np.nan
    image["flux"][1, 33, 33] = -0.01
    decoded = decode_legacy_south_image(image)
    assert set(decoded) == {"flux", "band"}
    assert np.isfinite(decoded["flux"]).all()
    assert np.isnan(image["flux"][1, 32, 32])  # Do not mutate the decoded input.
    with pytest.raises(ValueError, match="flux has shape"):
        decode_legacy_image(image)

    config = AstroPT3Config(**{
        **tiny_config.to_dict(), "modalities": [DEFAULT_MODALITIES[0], *LEGACY_SOUTH_MODALITIES],
        "vocab_size": None, "spiral": False,
    })
    sequencer = ObjectSequencer(config)
    record = {"object_id": "legacy_south:example", "legacy_south_image": decoded}
    obj = sequencer.build(record)
    ids = obj.input_ids.tolist()
    repeat = sequencer.build(record)
    assert ids[0] == 1 and len(ids) == 1 + 4 * 146
    planes = []
    for band, begin in zip("griz", (169, 172, 175, 178), strict=True):
        name = f"legacy_south_{band}"
        start = ids.index(begin)
        assert ids[start:start + 146] == [begin, *([begin + 1] * 144), begin + 2]
        tokens = obj.values[name]
        assert tokens.shape == (144, 64) and tokens.dtype == torch.float32
        assert torch.equal(tokens, repeat.values[name])
        planes.append(unpatchify_image(tokens, 8, 1, 96))
    pixels = torch.cat(planes)
    assert torch.isfinite(pixels).all()
    torch.testing.assert_close(pixels[:, -1, -1], torch.arange(1, 5, dtype=torch.float32).asinh())
    assert pixels[1, 1, 1] == torch.asinh(torch.tensor(-1.0))
    expected = torch.from_numpy(decoded["flux"][:, 32:128, 32:128] / 0.01).asinh()
    assert int((expected == 0).sum()) == 2
    torch.testing.assert_close(pixels[expected != 0], expected[expected != 0])
    assert torch.all(pixels[expected == 0] != 0)

    flat = flatten_packed_batch(PackedCollator(config, len(obj))([obj]), config, len(obj))
    assert config.vocab_size == 181 and flat["legacy_north_g_values"].shape == (0, 64)
    assert not flat["legacy_north_g_mask"].any()
    with torch.no_grad():
        out = AstroPT3Model(config).eval()(**regroup_micro_batch(flat, config.modality_registry().names()))
    assert set(out.modality_losses) == {f"legacy_south_{band}" for band in "griz"}
    assert torch.isfinite(out.loss)


@pytest.mark.gpu
def test_south_conversion_and_forward_parity():
    from test_nanotron_gpu import (
        REL_TOL,
        assert_loss_metrics,
        build_nt_model,
        make_nt,
        tiny_nt_config,
    )

    nt = make_nt()
    config = tiny_nt_config(nt, modalities=[DEFAULT_MODALITIES[0], *LEGACY_SOUTH_MODALITIES],
                            jetformer_flow_steps=2, jetformer_flow_hidden=32)
    hf_config = nt.to_hf.get_hf_config(config)
    nt_back = nt.from_hf.get_nanotron_config(hf_config)
    for name, band in {
        "legacy_north_g": "des-g", **{f"legacy_south_{b}": f"des-{b}" for b in "griz"},
    }.items():
        assert hf_config.modality_registry().get_config(name).band == band
        assert nt_back.modality(name)["band"] == band
    hf = AstroPT3Model(hf_config)
    torch.nn.Module.cuda(hf)
    hf.bfloat16().eval()
    model = build_nt_model(nt, config).eval()
    nt.from_hf.convert_hf_to_nt(hf, model, config)
    obj = ObjectSequencer(hf_config).build({"legacy_south_image": decode_legacy_south_image(_payload())})
    seq_len = 640
    flat = flatten_packed_batch(PackedCollator(hf_config, seq_len)([obj]), hf_config, seq_len)
    flat = {k: v.cuda() for k, v in flat.items()}
    assert flat["legacy_north_g_mask"].shape == (1, seq_len)
    assert not flat["legacy_north_g_mask"].any()
    assert flat["legacy_north_g_values"].shape == (0, 64)
    assert flat["legacy_north_g_positions"].shape == (0,)
    with torch.no_grad():
        out = model(**flat)
        ref = hf(**regroup_micro_batch(flat, config.modality_names()))
    assert set(ref.modality_losses) == {f"legacy_south_{band}" for band in "griz"}
    assert_loss_metrics(out, ref, config, flat)
    assert out["loss"].item() == pytest.approx(ref.loss.item(), rel=REL_TOL)
    back = AstroPT3Model(hf_config)
    torch.nn.Module.cuda(back)
    back.bfloat16()
    nt.to_hf.convert_nt_to_hf(model, back, config)
    for name, value in hf.state_dict().items():
        assert torch.equal(value, back.state_dict()[name]), name


@pytest.mark.network
def test_pinned_south_parquet_row():
    url = ("https://huggingface.co/datasets/hugging-science/mmu_legacysurvey_dr10_south_21/resolve/"
           "982f321784d8daf2a5f983eb3ec560f31b90d667/mmu_legacysurvey_dr10_south_21/dataset/"
           "Norder=10/Dir=11350000/Npix=11359295.parquet")
    with urlopen(url, timeout=30) as response:
        data = response.read(2_802_724)
    assert len(data) == 2_802_723
    table = pq.ParquetFile(io.BytesIO(data)).read(
        columns=["image.band", "image.flux", "object_id", "FLUX_G"],
    )
    assert len(table) == 4 and table["object_id"][0].as_py() == "2462m200-369"
    assert table["FLUX_G"][1].as_py() == pytest.approx(-0.42972362)
    decoded = decode_legacy_south_image(table["image"][0])
    assert decoded["flux"].shape == (4, 160, 160)
    assert np.isfinite(decoded["flux"]).all()
