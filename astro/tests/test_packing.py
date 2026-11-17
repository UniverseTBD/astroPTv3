from pathlib import Path

import pytest
import torch
from legacy_fixture import record_stream

from astropt3.config_io import load_model_config
from astropt3.data.packing import ObjectSequencer
from astropt3.modeling_astropt3 import left_shift_mask
from astropt3.tokenization import BOS_ID, PAD_ID, modality_token_ids

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "model"


def test_object_sequence_structure(sequencer, full_record):
    # pin the span order (the ADR 0008 shuffle would otherwise pick it) and
    # omit the scalar spans — their layout is covered by test_scalar_modalities
    obj = sequencer.build(
        full_record, modality_order=["images", "spectra"], include_scalars=False
    )
    assert len(obj) == 180  # 1 bos + (1+144+1) images + (1+31+1) spectra
    begin_img, ph_img, end_img = modality_token_ids("images")
    begin_spec, ph_spec, end_spec = modality_token_ids("spectra")
    ids = obj.input_ids
    assert ids[0] == BOS_ID
    assert ids[1] == begin_img and ids[146] == end_img
    assert (ids[2:146] == ph_img).all()
    assert ids[147] == begin_spec and ids[179] == end_spec
    assert (ids[148:179] == ph_spec).all()
    assert obj.masks["images"].sum() == 144 and obj.values["images"].shape == (144, 192)
    assert obj.masks["spectra"].sum() == 31 and obj.values["spectra"].shape == (31, 256)
    # spectra positions are normalized wavelengths in ~[0, 1]
    pos = obj.positions["spectra"]
    assert pos.shape == (31, 1) and (pos > 0).all() and (pos < 1.1).all()


def test_image_only_object(sequencer, image_only_record):
    obj = sequencer.build(image_only_record)
    # 1 bos + 146 images block + (1+1+1) ebv + (1+1+1) photometry (ADR 0008)
    assert len(obj) == 153
    assert "spectra" not in obj.masks and "Z" not in obj.masks
    assert {"ebv", "photometry"} <= set(obj.masks)


def test_spectrum_only_object(sequencer, spectrum_only_record):
    obj = sequencer.build(spectrum_only_record)
    # 1 bos + (1+31+1) spectra block + (1+1+1) Z span (ADR 0008)
    assert len(obj) == 37
    assert "images" not in obj.masks
    # image-catalog scalars are null on non-crossmatched rows
    assert "ebv" not in obj.masks and "photometry" not in obj.masks
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
    images = obj.values["images"]
    assert (images != 0).all() and torch.isfinite(images).all()

    again = sequencer.build(record)
    assert torch.equal(again.values["spectra"], obj.values["spectra"])  # deterministic
    other_epoch = sequencer.build(record, epoch=1)
    assert not torch.equal(other_epoch.values["spectra"].flatten()[n:], flat[n:])


def test_collator_packs_whole_objects(sequencer, collator):
    objs = [sequencer.build(r) for r in record_stream(6)]
    batch = collator(objs)
    B, T = batch["input_ids"].shape
    assert T == 896
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


def test_values_row_major_alignment(sequencer, collator):
    objs = [sequencer.build(r) for r in record_stream(4)]
    batch = collator(objs)
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
    begin_img, _, _ = modality_token_ids("images")
    shifted = left_shift_mask(batch["modality_masks"]["images"])
    begin_pos = (batch["input_ids"][0] == begin_img).nonzero()[0, 0]
    assert shifted[0, begin_pos]  # hidden at <|begin_images|> predicts patch 0


# -- ADR 0014 §8: per-band image tokenisation ---------------------------------


def _per_band_config():
    """A tiny config whose ``images`` modality is factorised per band."""
    from astropt3.configuration_astropt3 import DEFAULT_MODALITIES, AstroPT3Config

    modalities = []
    for modality in DEFAULT_MODALITIES:
        modality = dict(modality)
        if modality["name"] == "images":
            modality.update(
                input_size=64,  # 8*8*1 instead of 8*8*3
                max_positions=432,  # 3 bands x 144 patches
                channel_tokenization="per_band",
                band_order=("des-g", "des-r", "des-z"),
            )
        modalities.append(modality)
    return AstroPT3Config(
        modalities=modalities,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
    )


def test_per_band_gives_three_times_the_tokens_from_the_same_bytes(image_only_record):
    fused = ObjectSequencer(load_model_config(CONFIG_DIR / "test-tiny.yaml")[0])
    per_band = ObjectSequencer(_per_band_config())

    a = fused.build(image_only_record).values["images"]
    b = per_band.build(image_only_record).values["images"]

    assert a.shape == (144, 192)
    assert b.shape == (432, 64)
    # E_values does not move: identical target dimensions, finer factorisation
    assert a.numel() == b.numel() == 27_648


def test_per_band_positions_run_across_the_whole_concatenation(image_only_record):
    obj = ObjectSequencer(_per_band_config()).build(image_only_record)
    positions = obj.positions["images"]
    assert positions.tolist() == list(range(432))
    # the span is one contiguous block of 432 placeholders
    assert obj.masks["images"].sum() == 432


def test_per_band_is_band_major_in_the_configured_order(image_only_record):
    """Band-major, fixed order: band g's 144 patches, then r, then z."""
    from astropt3.data.band_registry import physical_normalize
    from astropt3.data.packing import IMAGE_CROP
    from astropt3.tokenization import patchify_image, spiralise

    config = _per_band_config()
    obj = ObjectSequencer(config).build(image_only_record)
    tokens = obj.values["images"]

    flux = torch.as_tensor(image_only_record["image"]["flux"], dtype=torch.float32)
    top = (flux.shape[-2] - IMAGE_CROP) // 2
    flux = flux[..., top : top + IMAGE_CROP, top : top + IMAGE_CROP]
    bands = [str(b) for b in image_only_record["image"]["band"]]
    flux = physical_normalize(flux, bands)

    for index, band in enumerate(["des-g", "des-r", "des-z"]):
        expected = spiralise(patchify_image(flux[bands.index(band)][None], 8))
        assert torch.allclose(tokens[index * 144 : (index + 1) * 144], expected)


def test_per_band_rejects_a_record_missing_a_named_band(image_only_record):
    record = dict(image_only_record)
    record["image"] = {
        "flux": torch.as_tensor(image_only_record["image"]["flux"])[:2],
        "band": ["des-g", "des-r"],
    }
    with pytest.raises(ValueError, match="missing"):
        ObjectSequencer(_per_band_config()).build(record)


def test_per_band_config_requires_a_band_order_and_a_matching_input_size():
    from astropt3.modalities import ModalityConfig

    base: dict = {
        "name": "images",
        "patch_size": 8,
        "family": "image",
        "source": "legacy",
        "record_keys": ["image"],
        "token_ids": [2, 3, 4],
        "channel_tokenization": "per_band",
    }
    with pytest.raises(ValueError, match="band_order"):
        ModalityConfig(input_size=64, **base)
    with pytest.raises(ValueError, match="input_size must be 64"):
        ModalityConfig(input_size=192, band_order=("des-g",), **base)
    with pytest.raises(ValueError, match="only applies to images"):
        ModalityConfig(
            **{
                **base,
                "family": "spectrum",
                "input_size": 64,
                "band_order": ["des-g"],
            }
        )
