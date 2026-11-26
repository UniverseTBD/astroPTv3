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
    assert len(obj) == 563  # 1 bos + 3 * (1+144+1) image bands + (1+122+1) spectra
    ids = obj.input_ids
    assert ids[0] == BOS_ID
    offset = 1
    for name in order:
        count, width = (122, 64) if name == "spectra" else (144, 64)
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
    assert pos.shape == (122, 1) and (pos > 0).all() and (pos < 1.1).all()


def test_image_only_object(sequencer, image_only_record):
    obj = sequencer.build(image_only_record)
    # 1 bos + 3 * 146 image band tokens + 2 * 3 scalar span tokens (ADR 0008)
    assert len(obj) == 445
    assert set(obj.masks) == {*NORTH_BANDS, "ebv", "photometry"}


def test_spectrum_only_object(sequencer, spectrum_only_record):
    obj = sequencer.build(spectrum_only_record)
    # 1 bos + (1+122+1) spectra block + (1+1+1) Z span (ADR 0008)
    assert len(obj) == 128
    # image bands and image-catalog scalars are absent on non-crossmatched rows
    assert set(obj.masks) == {"spectra", "Z"}
    assert obj.masks["spectra"].sum() == 122 and obj.values["spectra"].shape == (122, 64)
    assert obj.masks["Z"].sum() == 1 and obj.values["Z"].shape == (1, 1)


def test_masked_zeroed_padded_spectra_follow_continuum_and_images_noise_filled(sequencer, full_record):
    # Masked, exact-zero, and tail-padded spectral bins are filled around the local
    # continuum with seeded pixel noise (no point mass, no zero-flux dropout);
    # no-data image pixels stay noise-filled around zero.
    import copy

    from astropt3.data.spectral import spectral_normalize

    record = copy.deepcopy(full_record)
    spec, image = record["spectrum"], record["image"]
    spec["mask"] = spec["mask"].copy()
    spec["mask"][1000:1600] = True
    spec["flux"] = spec["flux"].copy() + 20.0  # clear nonzero continuum
    spec["flux"][1000:1600] = 1e6  # masked garbage must be ignored
    spec["flux"][2000:2050] = 0.0  # unmasked exact-zero bins
    image["flux"] = image["flux"].copy()
    image["flux"][:, 60:90, 60:90] = 0.0  # no-data block inside the 96x96 crop
    obj = sequencer.build(record)

    flat = obj.values["spectra"].flatten()
    n = len(spec["flux"])
    flux = torch.as_tensor(spec["flux"])
    expected = spectral_normalize(flux, torch.as_tensor(spec["lambda"]))
    keep = ~torch.as_tensor(spec["mask"]) & (flux != 0)
    assert torch.equal(flat[:n][keep], expected[keep])  # real bins untouched
    # Filled interior, exact-zero, and tail-pad bins sit on the local continuum with nonzero spread.
    for filled, ref in (
        (flat[1000:1600], 0.5 * (expected[999] + expected[1600])),
        (flat[2000:2050], 0.5 * (expected[1999] + expected[2050])),
        (flat[n:], expected[:n][keep][-16:].median()),
    ):
        assert torch.isfinite(filled).all()
        assert abs(filled.mean().item() - ref.item()) < 0.25
        assert filled.mean().item() > 1.0  # stays on continuum, not pulled to 0
        assert filled.std().item() >= 0.005  # no constant point mass
    for name in NORTH_BANDS:
        patches = obj.values[name]
        assert (patches != 0).all() and torch.isfinite(patches).all()

    # Garbage in masked bins must not leak into tokens; fill is deterministic per epoch.
    alt = copy.deepcopy(record)
    alt["spectrum"]["flux"][1000:1600] = -1e6
    assert torch.equal(sequencer.build(alt).values["spectra"], obj.values["spectra"])
    other_epoch = sequencer.build(record, epoch=1)
    assert not torch.equal(other_epoch.values["spectra"], obj.values["spectra"])
    assert not torch.equal(other_epoch.values["legacy_north_g"], obj.values["legacy_north_g"])


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
    rgb_base = {**base, "name": "galaxies_r", "source": "galaxies_with_hats", "record_keys": ["galaxy_image"], "token_ids": [217, 218, 219], "image_format": "rgb"}
    assert ModalityConfig(input_size=64, band="rgb-r", **rgb_base).band == "rgb-r"
    with pytest.raises(ValueError, match="rgb-"):
        ModalityConfig(input_size=64, band="des-g", **{**rgb_base, "name": "custom_rgb", "token_ids": [226, 227, 228]})
    with pytest.raises(ValueError, match="input_size"):
        ModalityConfig(input_size=192, band="rgb-r", **rgb_base)


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

    # Single-channel RGB spans select one channel and downsample 256x256 -> 96x96.
    split_mods = [
        {
            "name": f"galaxies_{ch}",
            "family": "image",
            "source": "galaxies_with_hats",
            "record_keys": ["galaxy_image"],
            "image_format": "rgb",
            "band": f"rgb-{ch}",
            "token_ids": [217 + 3 * idx, 218 + 3 * idx, 219 + 3 * idx],
            "input_size": 64,
            "patch_size": 8,
            "max_positions": 144,
            "image_crop": 0,
            "image_resize": 96,
        }
        for idx, ch in enumerate("rgb")
    ]
    split_seq = ObjectSequencer(AstroPT3Config(modalities=split_mods, spiral=True)).build(
        {"object_id": "g1", "galaxy_image": {"band": ["rgb-r", "rgb-g", "rgb-b"], "flux": flux}}
    )
    expected_96 = torch.nn.functional.interpolate(flux.unsqueeze(0), size=(96, 96), mode="area").squeeze(0) * RGB_SCALE
    for idx, ch in enumerate("rgb"):
        assert split_seq.values[f"galaxies_{ch}"].shape == (144, 64)
        rec_ch = unpatchify_image(antispiralise(split_seq.values[f"galaxies_{ch}"]), patch_size=8, channels=1, side=96)
        torch.testing.assert_close(rec_ch, expected_96[idx : idx + 1])

    small = torch.full((3, 64, 64), 0.5)
    with pytest.raises(ValueError, match="smaller than image_resize"):
        sequencer.build({"object_id": "g2", "galaxy_image": {"band": ["rgb-r", "rgb-g", "rgb-b"], "flux": small}})
