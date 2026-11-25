import pytest
import torch
from legacy_fixture import record_stream

from astropt3.data.packing import PackedCollator
from astropt3.modeling_astropt3 import left_shift_mask
from astropt3.tokenization import BOS_ID, PAD_ID, modality_token_ids

NORTH_BANDS = ("legacy_north_g", "legacy_north_r", "legacy_north_z")


def test_object_sequence_structure(sequencer, full_record):
    # pin the span order (the ADR 0008 shuffle would otherwise pick it) and
    # omit the scalar spans — their layout is covered by test_scalar_modalities
    order = [*NORTH_BANDS, "spectra"]
    obj = sequencer.build(full_record, modality_order=order, include_scalars=False)
    assert len(obj) == 472  # 1 bos + 3 * (1+144+1) image bands + (1+31+1) spectra
    ids = obj.input_ids
    assert ids[0] == BOS_ID
    offset = 1
    for name in order:
        count, width = (31, 256) if name == "spectra" else (144, 64)
        begin, placeholder, end = modality_token_ids(name)
        assert ids[offset] == begin and ids[offset + count + 1] == end
        assert (ids[offset + 1 : offset + count + 1] == placeholder).all()
        assert torch.equal(obj.masks[name], ids == placeholder)
        assert obj.values[name].shape == (count, width)
        if name in NORTH_BANDS:
            assert obj.positions[name].tolist() == list(range(144))
        offset += count + 2
    # spectra positions are normalized wavelengths in ~[0, 1]
    pos = obj.positions["spectra"]
    assert pos.shape == (31, 1) and (pos > 0).all() and (pos < 1.1).all()


def test_image_only_object(sequencer, image_only_record):
    obj = sequencer.build(image_only_record)
    # 1 bos + 3 * 146 image band tokens + 2 * 3 scalar span tokens (ADR 0008)
    assert len(obj) == 445
    assert set(obj.masks) == {*NORTH_BANDS, "ebv", "photometry"}


def test_spectrum_only_object(sequencer, spectrum_only_record):
    obj = sequencer.build(spectrum_only_record)
    # 1 bos + (1+31+1) spectra block + (1+1+1) Z span (ADR 0008)
    assert len(obj) == 37
    # image bands and image-catalog scalars are absent on non-crossmatched rows
    assert set(obj.masks) == {"spectra", "Z"}
    assert obj.masks["spectra"].sum() == 31 and obj.values["spectra"].shape == (31, 256)
    assert obj.masks["Z"].sum() == 1 and obj.values["Z"].shape == (1, 1)


def test_masked_padded_and_no_data_values_are_noise_not_exact_zeros(sequencer, full_record):
    # exact constants are point masses the likelihood can exploit without bound
    import copy

    from astropt3.data.spectral import spectral_normalize

    record = copy.deepcopy(full_record)
    spec, image = record["spectrum"], record["image"]
    spec["mask"] = spec["mask"].copy()
    spec["mask"][1000:1600] = True
    image["flux"] = image["flux"].copy()
    image["flux"][:, 60:90, 60:90] = 0.0  # no-data block inside the 96x96 crop
    obj = sequencer.build(record)

    flat = obj.values["spectra"].flatten()
    n = len(spec["flux"])
    flux = torch.as_tensor(spec["flux"])
    expected = spectral_normalize(flux, torch.as_tensor(spec["lambda"]))
    keep = ~torch.as_tensor(spec["mask"])
    assert torch.equal(flat[:n][keep], expected[keep])  # real bins untouched
    filled = torch.cat([flat[:n][~keep], flat[n:]])  # masked bins + tail padding
    assert len(flat) > n and torch.isfinite(filled).all()
    assert (filled != 0).all() and len(filled.unique()) == len(filled)
    assert filled.std() > 0.005
    for name in NORTH_BANDS:
        patches = obj.values[name]
        assert (patches != 0).all() and torch.isfinite(patches).all()

    again = sequencer.build(record)
    assert torch.equal(again.values["spectra"], obj.values["spectra"])  # deterministic
    other_epoch = sequencer.build(record, epoch=1)
    assert not torch.equal(other_epoch.values["spectra"].flatten()[n:], flat[n:])


def test_collator_packs_whole_objects(sequencer, tiny_config):
    objs = [sequencer.build(r) for r in record_stream(6)]
    batch = PackedCollator(tiny_config, seq_len=1024)(objs)
    B, T = batch["input_ids"].shape
    assert (B, T) == (3, 1024) and set(NORTH_BANDS) <= set(batch["modality_values"])
    # every object's position_ids run 0..len-1 contiguously: object starts are
    # exactly the positions where position_ids == 0 and the token is not pad
    for b in range(B):
        ids = batch["input_ids"][b]
        pos = batch["position_ids"][b]
        starts = [t for t in range(T) if pos[t] == 0 and ids[t] != PAD_ID]
        for t in starts:
            assert ids[t] == BOS_ID  # objects begin with <|bos|>, never split
    # flattened values align with row-major mask order
    for m, values in batch["modality_values"].items():
        assert batch["modality_masks"][m].sum().item() == len(values)


def test_values_row_major_alignment(sequencer, tiny_config):
    objs = [sequencer.build(r) for r in record_stream(4)]
    batch = PackedCollator(tiny_config, seq_len=1024)(objs)
    # rebuild the concatenation by scanning rows/objects in order and compare
    for m in batch["modality_values"]:
        expected = torch.cat([o.values[m] for o in objs if m in o.values], dim=0)
        assert torch.equal(batch["modality_values"][m], expected)


def test_left_shift_mask():
    mask = torch.tensor([[False, False, True, True, False]])
    shifted = left_shift_mask(mask)
    assert shifted.tolist() == [[False, True, True, False, False]]


def test_begin_token_predicts_first_patch(sequencer, collator, full_record):
    obj = sequencer.build(full_record)
    batch = collator([obj])
    for name in NORTH_BANDS:
        begin, _, _ = modality_token_ids(name)
        shifted = left_shift_mask(batch["modality_masks"][name])
        begin_pos = (batch["input_ids"][0] == begin).nonzero()[0, 0]
        assert shifted[0, begin_pos]  # the band's begin token predicts patch 0


def test_image_band_spans_match_the_labeled_channels(sequencer, image_only_record):
    from astropt3.data.band_registry import physical_normalize
    from astropt3.data.packing import IMAGE_CROP
    from astropt3.tokenization import patchify_image, spiralise

    obj = sequencer.build(image_only_record)
    flux = torch.as_tensor(image_only_record["image"]["flux"], dtype=torch.float32)
    top = (flux.shape[-2] - IMAGE_CROP) // 2
    flux = flux[..., top : top + IMAGE_CROP, top : top + IMAGE_CROP]
    bands = image_only_record["image"]["band"]

    for name, band in zip(NORTH_BANDS, ("des-g", "des-r", "des-z"), strict=True):
        channel = flux[bands.index(band)][None]
        expected = spiralise(patchify_image(physical_normalize(channel, [band]), 8))
        assert torch.allclose(obj.values[name], expected)


def test_rejects_a_record_missing_a_configured_band(sequencer, image_only_record):
    record = dict(image_only_record)
    record["image"] = {
        "flux": torch.as_tensor(image_only_record["image"]["flux"])[:2],
        "band": ["des-g", "des-r"],
    }
    with pytest.raises(ValueError, match="des-z"):
        sequencer.build(record)


def test_image_config_requires_band_and_single_channel_width():
    from astropt3.modalities import ModalityConfig

    base: dict = {
        "name": "legacy_north_g",
        "patch_size": 8,
        "family": "image",
        "source": "legacy_north",
        "record_keys": ["image"],
        "token_ids": [160, 161, 162],
    }
    with pytest.raises(ValueError, match="band"):
        ModalityConfig(input_size=64, **base)
    with pytest.raises(ValueError, match="calibrated band"):
        ModalityConfig(input_size=64, band="rgb-r", **base)
    with pytest.raises(ValueError, match="input_size"):
        ModalityConfig(input_size=192, band="des-g", **base)
    with pytest.raises(ValueError, match="for images"):
        ModalityConfig(**{**base, "family": "spectrum", "input_size": 64, "band": "des-g"})
    with pytest.raises(ValueError, match="for images"):
        ModalityConfig(**{**base, "family": "spectrum", "input_size": 64, "image_resize": 128})
    with pytest.raises(ValueError, match="image_resize"):
        ModalityConfig(input_size=64, band="des-g", image_resize=-8, **base)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        ModalityConfig(input_size=64, band="des-g", image_resize=10, **base)
    with pytest.raises(ValueError, match="max_positions"):
        ModalityConfig(input_size=64, band="des-g", max_positions=16, image_resize=64, **base)


def test_image_resize_downsamples_full_frame_and_propagates_fill_mask():
    from astropt3 import AstroPT3Config
    from astropt3.data.band_registry import RGB_SCALE
    from astropt3.data.packing import ObjectSequencer
    from astropt3.tokenization import antispiralise, unpatchify_image

    rgb = {
        "name": "galaxies",
        "family": "image",
        "source": "galaxies_with_hats",
        "record_keys": ["galaxy_image"],
        "image_format": "rgb",
        "token_ids": [23, 24, 25],
        "input_size": 192,
        "patch_size": 8,
        "max_positions": 256,
        "image_crop": 0,
        "image_resize": 128,
    }
    sequencer = ObjectSequencer(AstroPT3Config(modalities=[rgb], spiral=True))
    # Distinct corner values verify full-frame area averaging rather than a center crop.
    flux = torch.linspace(0.1, 0.9, 3 * 256 * 256, dtype=torch.float32).reshape(3, 256, 256)
    fill_mask = torch.zeros_like(flux, dtype=torch.bool)
    fill_mask[0, 0, 1] = True  # one invalid pixel in the top-left 2x2 block
    seq = sequencer.build({"object_id": "g1", "galaxy_image": {"band": ["rgb-r", "rgb-g", "rgb-b"], "flux": flux, "fill_mask": fill_mask}})
    assert seq.values["galaxies"].shape == (256, 192)
    recovered = unpatchify_image(antispiralise(seq.values["galaxies"]), patch_size=8, channels=3, side=128)
    expected = torch.nn.functional.avg_pool2d(flux, 2) * RGB_SCALE
    torch.testing.assert_close(recovered[1:], expected[1:])
    torch.testing.assert_close(recovered[0, 1:], expected[0, 1:])
    assert not torch.isclose(recovered[0, 0, 0], expected[0, 0, 0])  # invalidated block was noise-filled

    small = torch.full((3, 64, 64), 0.5)
    with pytest.raises(ValueError, match="smaller than image_resize"):
        sequencer.build({"object_id": "g2", "galaxy_image": {"band": ["rgb-r", "rgb-g", "rgb-b"], "flux": small}})
