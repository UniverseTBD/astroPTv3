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
    x = torch.tensor([0.03, 0.7, 1.5, 42.0])  # positive: several maps are log10
    for name in (
        "Z",
        "sdss_Z",
        "provabgs_Z_HP",
        "provabgs_LOG_MSTAR",
        "provabgs_Z_MW",
        "provabgs_TAGE_MW",
        "provabgs_AVG_SFR",
        "gwh_gz_smooth-or-featured",
        "gwh_W50",
        "sdss_vdisp",
        "provabgs_MAG_W1",
        "ebv",
        "photometry",
        # ADR 0014 A8 free scalars
        "fiberflux",
        "psf_fwhm",
        "hsc_shape",
        "hsc_psf_shape",
    ):
        rt = scalar_inverse(name, scalar_normalize(name, x))
        assert torch.allclose(rt, x, atol=1e-5), name
    for unknown in ("sSFR", "gwh_smooth-or-featured_smooth_fraction", "gwh_gz_undocumented"):
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


def test_scalar_predicates_ranges_and_quality_gates():
    from astropt3.data.scalar_registry import scalar_valid

    good = {"sdss_ZWARNING": 0.0}
    assert scalar_valid("sdss_vdisp", [150.0], good)
    assert not scalar_valid("sdss_vdisp", [0.0], good)  # 0 = no measurement
    assert not scalar_valid("sdss_vdisp", [850.0], good)  # SDSS cap
    assert not scalar_valid("sdss_vdisp", [150.0], {"sdss_ZWARNING": 1.0})
    assert not scalar_valid("sdss_vdisp", [150.0], {})  # unknown quality is not reliable
    assert not scalar_valid("gwh_total_sfr_median", [-99.0], {"gwh_total_sfr_flag": 0.0})
    errors = {f"hsc_{band}_cmodel_magerr": 0.01 for band in "grizy"}
    assert scalar_valid("hsc_cmodel_mag", [21.0] * 5, errors)
    assert not scalar_valid("hsc_cmodel_mag", [21.0] * 5, {**errors, "hsc_y_cmodel_magerr": 0.2})
    assert not scalar_valid("hsc_cmodel_mag", [21.0, 21.0, float("nan"), 21.0, 21.0], errors)
    with pytest.raises(ValueError, match="non-numeric"):
        scalar_valid("ebv", ["dusty"], {})


def test_maps_undefined_at_zero_have_a_positive_valid_range():
    """A log map without a lower bound would emit NaN tokens for x <= 0."""
    from astropt3.data.scalar_registry import SCALAR_TRANSFORMS, VALID_RANGES

    for name in SCALAR_TRANSFORMS:
        edge = scalar_normalize(name, torch.tensor([-0.5, 0.0]))
        if not torch.isfinite(edge).all():
            assert VALID_RANGES[name][0] >= 0.0, name


def test_recipe_galaxy_zoo_spans_match_registry_questions():
    from pathlib import Path

    import yaml

    from astropt3.data.scalar_registry import GZ_QUESTIONS

    recipe = Path(__file__).resolve().parents[1] / "configs/nanotron/astropt3-70m.yaml"
    modalities = yaml.safe_load(recipe.read_text())["model"]["model_config"]["modalities"]
    spans = {m["name"]: m for m in modalities if m["name"].startswith("gwh_gz_")}
    assert set(spans) == {f"gwh_gz_{question}" for question in GZ_QUESTIONS}
    for question, answers in GZ_QUESTIONS.items():
        span = spans[f"gwh_gz_{question}"]
        assert span["record_keys"] == [f"gwh_{question}_{answer}_fraction" for answer in answers]
        assert span["input_size"] == len(answers)


GZ10_LABEL = {
    "name": "gz10_label", "family": "scalar", "source": "gz10", "record_keys": ["gz10_label"],
    "token_ids": [64, 65, 66], "input_size": 10, "patch_size": 1, "pos_type": "index",
    "max_positions": 1, "categories": 10,
}


@pytest.fixture
def categorical_model(tiny_config):
    from astropt3 import AstroPT3Config, AstroPT3Model

    config = AstroPT3Config(**{**tiny_config.to_dict(), "modalities": [*tiny_config.modalities, GZ10_LABEL]})
    torch.manual_seed(0)
    return AstroPT3Model(config).eval()


def test_categorical_scalar_is_one_hot_with_cross_entropy(categorical_model, full_record):
    from astropt3.data.packing import ObjectSequencer
    from astropt3.data.scalar_registry import scalar_valid
    from astropt3.modalities import CategoricalHead
    from astropt3.modeling_astropt3 import left_shift_mask

    model = categorical_model
    assert isinstance(model.decoders["gz10_label"], CategoricalHead)
    assert "gz10_label" in model.scalar_names and "gz10_label" not in model.flows
    sequencer = ObjectSequencer(model.config)
    obj = sequencer.build({**full_record, "gz10_label": 7})
    torch.testing.assert_close(obj.values["gz10_label"], torch.nn.functional.one_hot(torch.tensor([7]), 10).float())
    assert scalar_inverse("gz10_label", obj.values["gz10_label"]).item() == 7
    for bad in (10, -1, 2.5):  # out of range, or not a class index
        assert not scalar_valid("gz10_label", [bad], {})
        assert "gz10_label" not in sequencer.build({**full_record, "gz10_label": bad}).masks
    from astropt3.data.packing import PackedCollator

    batch = PackedCollator(model.config, seq_len=896)([obj])
    with torch.no_grad():
        out = model(**batch)
    mask = batch["modality_masks"]["gz10_label"]
    logits = model.decoders["gz10_label"](out.last_hidden_state[left_shift_mask(mask)])
    expected = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([7]))
    torch.testing.assert_close(out.modality_losses["gz10_label"], expected)
    assert out.predictions["gz10_label"].shape == (1, 10)


def test_categorical_generation_samples_one_hot_classes(categorical_model, full_record):
    from astropt3.data.packing import ObjectSequencer
    from astropt3.generation import generate

    sequencer = ObjectSequencer(categorical_model.config)
    record = {**full_record, "gz10_label": 3}
    order = list(sequencer.build(full_record).masks)
    template = sequencer.build(record, modality_order=[*order, "gz10_label"])
    sampled = generate(categorical_model, template, {"gz10_label"}, n=4,
                       generator=torch.Generator().manual_seed(0))["gz10_label"]
    assert sampled.shape == (4, 1, 10)
    assert torch.equal(sampled.sum(-1), torch.ones(4, 1)) and set(sampled.unique().tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("patch", [{"categories": 10, "family": "spectrum"}, {"categories": 10, "input_size": 9},
                                   {"categories": 1, "input_size": 1}])
def test_categorical_config_is_validated(patch):
    from astropt3.modalities import ModalityConfig

    with pytest.raises(ValueError, match="categorical"):
        ModalityConfig(**{**GZ10_LABEL, "record_keys": tuple(GZ10_LABEL["record_keys"]), **patch})
