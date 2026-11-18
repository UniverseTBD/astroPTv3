"""ADR 0008 scalar modalities: registry, spans, loss, generation, metrics."""

import pytest
import torch
from legacy_fixture import make_record
from test_model import single_band_collator as single_band_collator
from test_model import single_band_config as single_band_config
from test_model import single_band_model as single_band_model
from test_model import single_band_sequencer as single_band_sequencer

from astropt3.data.scalar_registry import scalar_inverse, scalar_normalize
from astropt3.modalities import gmm_nll


def test_scalar_registry_roundtrip_and_unknown_raises():
    x = torch.tensor([0.0, 0.03, 0.7, 1.5, 42.0])
    for name in (
        "Z",
        "sdss_Z",
        "provabgs_Z_HP",
        "provabgs_LOG_MSTAR",
        "provabgs_Z_MW",
        "provabgs_TAGE_MW",
        "provabgs_AVG_SFR",
        "gwh_smooth-or-featured_smooth_fraction",
        "ebv",
        "photometry",
        # ADR 0014 A8 free scalars
        "fiberflux",
        "psf_fwhm",
        "hsc_extendedness",
        "hsc_shape",
        "hsc_psf_shape",
    ):
        rt = scalar_inverse(name, scalar_normalize(name, x))
        assert torch.allclose(rt, x, atol=1e-5), name
    for unknown in ("sSFR", "gwh_undocumented_fraction"):
        with pytest.raises(NotImplementedError):
            scalar_normalize(unknown, x)


def test_a8_transforms_land_o1_on_measured_ranges():
    """ADR 0014 A8 pins each knee to published units; check it lands O(1).

    The percentiles are the ones measured on real partitions and recorded in
    A8 — a transform that pushed them far off O(1) would be the wrong knee.
    """
    measured = {
        # name -> (p1, p50, p99) from A8's tables
        "fiberflux": (0.133, 3.371, 51.324),
        "psfdepth": (49.315, 178.508, 915.720),
        "psf_fwhm": (1.050, 1.730, 2.238),
        "hsc_cmodel_mag": (17.579, 21.571, 25.464),
        "hsc_shape": (-2.345, 0.282, 24.729),
        "hsc_psf_shape": (-0.004, 0.106, 0.166),
    }
    for name, values in measured.items():
        out = scalar_normalize(name, torch.tensor(values))
        # arcsinh is log-like, so a bright object legitimately reaches ~9 —
        # the same place `photometry` already lands. The bound catches a wrong
        # knee (orders of magnitude off), not the tail.
        assert out.abs().max() < 12.0, (name, out)
        rt = scalar_inverse(name, out)
        assert torch.allclose(rt, torch.tensor(values), atol=1e-4), name

    # fiberflux is nMgy exactly like photometry, so it must share the knee —
    # aperture flux and image pixels have to stay in one unit system
    flux = torch.tensor([0.133, 3.371, 51.324])
    assert torch.equal(
        scalar_normalize("fiberflux", flux), scalar_normalize("photometry", flux)
    )


def test_psfdepth_handles_the_zero_that_means_no_coverage():
    """log10(1 + x), not log10(x): 0 is a real value upstream, not an error."""
    out = scalar_normalize("psfdepth", torch.tensor([0.0]))
    assert torch.isfinite(out).all()
    assert torch.allclose(scalar_inverse("psfdepth", out), torch.tensor([0.0]), atol=1e-4)


def test_hsc_shape_keeps_the_sign_of_the_cross_moment():
    """shape12 is signed; a log transform would silently drop half the field."""
    signed = torch.tensor([-2.345, -0.069, 0.061, 1.029])
    out = scalar_normalize("hsc_shape", signed)
    assert torch.equal(out.sign(), signed.sign())
    assert torch.allclose(scalar_inverse("hsc_shape", out), signed, atol=1e-5)


def test_scalar_spans_and_gating(single_band_sequencer, full_record):
    sequencer = single_band_sequencer
    obj = sequencer.build(full_record)
    assert {"Z", "ebv", "photometry"} <= set(obj.masks)
    assert obj.values["Z"].shape == (1, 1)
    assert obj.values["photometry"].shape == (1, 3)
    # normalized truth round-trips to the record value
    z = float(scalar_inverse("Z", obj.values["Z"][0, 0]))
    assert abs(z - full_record["Z"]) < 1e-5
    # ZWARN != 0 suppresses the Z span (ADR 0005's reliability cut, reused)
    bad = dict(full_record, ZWARN=True)
    assert "Z" not in sequencer.build(bad).masks
    # missing scalar fields are absent spans, not errors
    partial = {k: v for k, v in full_record.items() if k != "flux_r"}
    assert "photometry" not in sequencer.build(partial).masks
    # include_scalars=False strips every scalar span (probe sequences)
    assert set(sequencer.build(full_record, include_scalars=False).masks) == {
        "legacy_north_g",
        "spectra",
    }


def test_scalar_loss_matches_manual_gmm_nll(
    single_band_model, single_band_sequencer, single_band_collator, full_record
):
    """Scalar losses are gmm_nll on the raw normalized value, no logdet."""
    batch = single_band_collator(
        [
            single_band_sequencer.build(
                full_record,
                modality_order=["legacy_north_g", "spectra", "Z", "ebv", "photometry"],
            )
        ]
    )
    with torch.no_grad():
        out = single_band_model(**batch)
        from astropt3.modeling_astropt3 import left_shift_mask

        for name in ("Z", "ebv", "photometry"):
            mask = batch["modality_masks"][name]
            hidden = out.last_hidden_state[left_shift_mask(mask)]
            logits_pi, mu, log_sigma = single_band_model.decoders[name](hidden)
            manual = gmm_nll(
                batch["modality_values"][name], logits_pi, mu, log_sigma
            ).mean()
            assert torch.allclose(out.modality_losses[name], manual, atol=1e-6), name


def test_scalar_heads_are_gmm_and_never_flow(single_band_config):
    from astropt3 import AstroPT3Model
    from astropt3.modalities import GMMHead

    model = AstroPT3Model(single_band_config)
    for name in ("Z", "ebv", "photometry"):
        assert isinstance(model.decoders[name], GMMHead)
        assert model.decoders[name].k == single_band_config.scalar_gmm_k
        assert name not in model.flows  # scalars never flow


def test_unconditional_generation_covers_scalars(single_band_config):
    from astropt3 import AstroPT3Model
    from astropt3.data.packing import ObjectSequencer
    from astropt3.generation import generate

    config = single_band_config
    torch.manual_seed(0)
    model = AstroPT3Model(config).eval()
    template = ObjectSequencer(config).build(make_record(3, image_only_fraction=0.0))
    sampled = generate(
        model,
        template,
        set(template.masks),
        n=1,
        generator=torch.Generator().manual_seed(0),
    )
    assert {"Z", "ebv", "photometry"} <= set(sampled)
    assert sampled["Z"].shape == (1, 1, 1)
    assert sampled["photometry"].shape == (1, 1, 3)
    assert all(torch.isfinite(v).all() for v in sampled.values())


def test_scalar_head_metrics(single_band_model, single_band_sequencer):
    """Model-side: given objects whose Z span is pinned last, score the head."""
    from legacy_fixture import record_stream

    from astropt3.eval.scalar_head import scalar_head_metrics

    sequencer = single_band_sequencer
    objects, targets = [], []
    for record in record_stream(64):
        if record.get("Z") is None or "spectrum" not in record:
            continue
        built = sequencer.build(record)
        others = sorted(m for m in built.masks if m != "Z")
        if not others:
            continue
        obj = sequencer.build(record, modality_order=others + ["Z"])
        objects.append(obj)
        targets.append(float(obj.values["Z"][0, 0]))
        if len(objects) >= 16:
            break
    assert len(objects) == 16
    # the target span is pinned last so every observation conditions it
    for obj in objects:
        assert int(obj.masks["Z"].nonzero()[0]) > max(
            int(obj.masks[m].nonzero()[-1]) for m in obj.masks if m != "Z"
        )
    result = scalar_head_metrics(single_band_model, objects, targets, target="Z")
    assert result["n_objects"] == 16
    for key in ("nmad", "outlier_frac", "coverage_1sig", "bias", "r2"):
        assert torch.isfinite(torch.tensor(result[key])), key
    assert result["r2"] <= 1.0
