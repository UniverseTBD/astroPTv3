"""Uniform random span order, seeded per (object_id, epoch) (ADR 0008)."""

import pytest
import torch
from legacy_fixture import make_record
from test_model import single_band_config as single_band_config
from test_model import single_band_sequencer as single_band_sequencer

from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.tokenization import modality_token_ids


def _first_begin(seq):
    """Name of the modality whose span opens first in the sequence."""
    begins = {modality_token_ids(m)[0]: m for m in seq.masks}
    for tok in seq.input_ids.tolist():
        if tok in begins:
            return begins[tok]
    raise AssertionError("no modality span found")


def _bimodal(i):
    return make_record(i, image_only_fraction=0.0)


def test_shuffle_is_uniform_and_deterministic(single_band_sequencer):
    sequencer = single_band_sequencer
    # N=2 (scalar-free): the shuffle reduces to ADR 0005's 50/50 flip
    firsts = [
        _first_begin(sequencer.build(_bimodal(i), include_scalars=False))
        for i in range(200)
    ]
    frac = firsts.count("spectra") / len(firsts)
    assert 0.35 < frac < 0.65  # seeded shuffle is ~uniform over object ids
    # N=5 (all spans): every modality leads somewhere, roughly uniformly
    full_firsts = [_first_begin(sequencer.build(_bimodal(i))) for i in range(300)]
    counts = {m: full_firsts.count(m) for m in set(full_firsts)}
    assert set(counts) == {"legacy_north_g", "spectra", "Z", "ebv", "photometry"}
    assert all(0.1 < c / 300 < 0.35 for c in counts.values()), counts
    # deterministic per (record, epoch); a new epoch reseeds the shuffle
    for i in range(5):
        assert _first_begin(sequencer.build(_bimodal(i))) == full_firsts[i]
    epoch1 = [
        _first_begin(sequencer.build(_bimodal(i), epoch=1)) for i in range(300)
    ]
    assert epoch1 != full_firsts  # the aggregate order sequence moves
    assert epoch1 == [
        _first_begin(sequencer.build(_bimodal(i), epoch=1)) for i in range(300)
    ]


def test_single_span_records_are_unaffected(single_band_sequencer):
    sequencer = single_band_sequencer
    image_only = make_record(4, image_only_fraction=1.0)
    assert _first_begin(sequencer.build(image_only, include_scalars=False)) == "legacy_north_g"
    spec_only = make_record(4, image_only_fraction=0.0, spectrum_only_fraction=1.0)
    assert _first_begin(sequencer.build(spec_only, include_scalars=False)) == "spectra"


def test_explicit_order_override_and_validation(single_band_sequencer):
    sequencer = single_band_sequencer
    record = _bimodal(3)
    rev = sequencer.build(
        record, modality_order=["spectra", "legacy_north_g"], include_scalars=False
    )
    assert _first_begin(rev) == "spectra"
    fwd = sequencer.build(
        record, modality_order=["legacy_north_g", "spectra"], include_scalars=False
    )
    assert _first_begin(fwd) == "legacy_north_g"
    # per-modality payloads are identical either way — only the skeleton moves
    for m in ("legacy_north_g", "spectra"):
        assert torch.equal(rev.values[m], fwd.values[m])
    with pytest.raises(ValueError, match="exactly"):
        sequencer.build(record, modality_order=["spectra"], include_scalars=False)
    with pytest.raises(ValueError, match="exactly"):
        sequencer.build(
            make_record(4, image_only_fraction=1.0),
            modality_order=["spectra", "legacy_north_g"],
            include_scalars=False,
        )


def test_model_forward_on_mixed_order_batch(single_band_config):
    """The mask-based loss alignment must not care about span order."""
    from astropt3.modeling_astropt3 import AstroPT3Model

    torch.manual_seed(0)
    model = AstroPT3Model(single_band_config).eval()
    seq = ObjectSequencer(single_band_config)
    objs = [
        seq.build(_bimodal(3), modality_order=["spectra", "legacy_north_g"], include_scalars=False),
        seq.build(_bimodal(5), modality_order=["legacy_north_g", "spectra"], include_scalars=False),
    ]
    batch = PackedCollator(single_band_config, seq_len=896)(objs)
    out = model(**batch)
    assert torch.isfinite(out.loss)
    assert set(out.modality_losses) == {"legacy_north_g", "spectra"}


def test_spectra_to_images_mode(single_band_config):
    """spectra-to-images samples an image span conditioned on the spectrum."""
    from astropt3.eval.samples import build_template, sample_template
    from astropt3.modeling_astropt3 import AstroPT3Model

    config = single_band_config
    torch.manual_seed(0)
    model = AstroPT3Model(config).eval()
    seq = ObjectSequencer(config)
    record = _bimodal(3)
    template = build_template(seq, record, "spectra-to-images")
    assert set(template.masks) == {"legacy_north_g", "spectra"}  # scalar-free (ADR 0008)
    generator = torch.Generator().manual_seed(0)
    sampled = sample_template(model, template, "spectra-to-images", n=2, generator=generator)
    assert set(sampled) == {"legacy_north_g"}
    assert sampled["legacy_north_g"].shape == (2, *template.values["legacy_north_g"].shape)
    assert torch.isfinite(sampled["legacy_north_g"]).all()
    # an images-first template must be rejected — no conditioning flows
    images_first = seq.build(
        record, modality_order=["legacy_north_g", "spectra"], include_scalars=False
    )
    with pytest.raises(ValueError, match="spectra-first"):
        sample_template(model, images_first, "spectra-to-images", n=1)


def test_default_modes_includes_spectra_to_images(single_band_config):
    from astropt3.eval.samples import default_modes

    assert default_modes(single_band_config) == [
        "unconditional",
        "image-to-spectra",
        "spectra-to-images",
    ]
