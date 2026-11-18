import pytest
import torch

from astropt3 import AstroPT3Config
from astropt3.tokenization import (
    BOS_ID,
    PAD_ID,
    antispiralise,
    modality_token_ids,
    patchify_image,
    patchify_spectrum,
    special_token_map,
    spiralise,
    unpatchify_image,
    unpatchify_spectrum,
)


def test_special_token_ids_frozen():
    assert PAD_ID == 0 and BOS_ID == 1
    expected_blocks = {
        "legacy_north_g": (160, 161, 162),
        "legacy_north_r": (163, 164, 165),
        "legacy_north_z": (166, 167, 168),
        "spectra": (5, 6, 7),
        "Z": (8, 9, 10),
        "ebv": (11, 12, 13),
        "photometry": (14, 15, 16),
    }
    assert {
        name: modality_token_ids(name) for name in expected_blocks
    } == expected_blocks
    config = AstroPT3Config()
    assert config.vocab_size == 169
    assert all(
        {"family", "source", "record_keys", "token_ids"} <= set(modality)
        for modality in config.modalities
    )
    assert {m["name"]: tuple(m["token_ids"]) for m in config.modalities} == expected_blocks
    active_tokens = special_token_map(config.modalities)
    assert len(set(active_tokens.values())) == 23
    frozen_tokens = special_token_map()
    assert all(frozen_tokens[name] == token_id for name, token_id in active_tokens.items())
    assert modality_token_ids("galaxies") == (23, 24, 25)
    for name in ("images", "hsc_images", "legacy_south_images"):
        assert f"<|{name}|>" not in frozen_tokens


def test_new_token_blocks_fill_reservation_then_enlarge_vocab():
    modalities = [m for m in AstroPT3Config().modalities if m["family"] != "image"] + [
        {
            "name": f"extra_{index}",
            "input_size": 1,
            "patch_size": 1,
            "family": "scalar",
            "source": "test",
            "record_keys": [f"extra_{index}"],
        }
        for index in range(13)
    ]
    config = AstroPT3Config(modalities=modalities)
    by_name = {modality["name"]: modality for modality in config.modalities}
    assert by_name["extra_0"]["token_ids"] == [17, 18, 19]
    assert by_name["extra_1"]["token_ids"] == [26, 27, 28]
    assert by_name["extra_11"]["token_ids"] == [59, 60, 61]
    assert by_name["extra_12"]["token_ids"] == [64, 65, 66]
    assert config.vocab_size == 67
    with pytest.raises(ValueError, match="vocab_size=64"):
        AstroPT3Config(modalities=modalities, vocab_size=64)


@pytest.mark.parametrize("base", [2, 20, 53, 160])
def test_new_token_block_collision_raises(base):
    modalities = AstroPT3Config().modalities + [
        {
            "name": "bad",
            "input_size": 1,
            "patch_size": 1,
            "family": "scalar",
            "source": "test",
            "record_keys": ["bad"],
            "token_ids": list(range(base, base + 3)),
        }
    ]
    with pytest.raises(ValueError, match="collide"):
        AstroPT3Config(modalities=modalities)


def test_image_patchify_roundtrip():
    flux = torch.randn(3, 152, 152)
    patches = patchify_image(flux, patch_size=8)
    expected_shape = (361, 192)
    assert patches.shape == expected_shape
    back = unpatchify_image(patches, patch_size=8, channels=3, side=152)
    assert torch.equal(back, flux)


def test_spectrum_patchify_roundtrip():
    flux = torch.randn(7781)
    lam = torch.linspace(3600.0, 9824.0, 7781)
    patches, lam_mean = patchify_spectrum(flux, lam, patch_size=256)
    expected_patch_shape = (31, 256)
    expected_position_shape = (31,)
    assert patches.shape == expected_patch_shape
    assert lam_mean.shape == expected_position_shape
    back = unpatchify_spectrum(patches, length=7781)
    assert torch.equal(back, flux)


def test_spectrum_last_patch_position_ignores_padding():
    flux = torch.randn(7781)
    lam = torch.linspace(3600.0, 9824.0, 7781)
    _, lam_mean = patchify_spectrum(flux, lam, patch_size=256)
    # last patch holds 7781 - 30*256 = 101 real bins + 155 padded zeros
    expected = lam[30 * 256 :].mean()
    assert torch.allclose(lam_mean[-1], expected)
    # a padded-mean would be dragged far below the true value
    assert lam_mean[-1] > lam_mean[-2]


def test_spiralise_roundtrip():
    patches = torch.randn(361, 192)
    assert torch.equal(antispiralise(spiralise(patches)), patches)
    assert not torch.equal(spiralise(patches), patches)
