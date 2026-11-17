"""Gaussian factor-out: exact likelihood, bottleneck, and inverse sampling."""

from typing import cast

import pytest
import torch

from astropt3 import AstroPT3Config, AstroPT3Model
from astropt3.data.packing import ObjectSeq, PackedCollator
from astropt3.eval.val_loss import evaluate
from astropt3.flows import flow_object_ids
from astropt3.generation import generate, reconstruct
from astropt3.modalities import Encoder, gmm_nll, std_normal_nll
from astropt3.modeling_astropt3 import left_shift_mask


def config_with_split(tiny_config, split):
    mods = [dict(m) for m in tiny_config.modalities]
    for m in mods:
        m["d_ar"] = max(1, m["input_size"] // 4) if split and m["family"] != "scalar" else m["input_size"]
    return AstroPT3Config(**{**tiny_config.to_dict(), "modalities": mods})


def short_template(config):
    ids, masks, values, positions = [1], {}, {}, {}
    for m in config.modalities:
        begin, placeholder, end = m["token_ids"]
        count = 1 if m["family"] == "scalar" else 3
        start = len(ids) + 1
        ids.extend([begin, *([placeholder] * count), end])
        masks[m["name"]] = (start, count)
        values[m["name"]] = torch.randn(count, m["input_size"])
        positions[m["name"]] = (
            torch.arange(count) if m.get("pos_type", "index") == "index"
            else torch.linspace(0, 1, count).unsqueeze(-1)
        )
    ids = torch.tensor(ids)
    masks = {name: (torch.arange(len(ids)) >= start) & (torch.arange(len(ids)) < start + count)
             for name, (start, count) in masks.items()}
    return ObjectSeq(ids, masks, values, positions)


@pytest.mark.parametrize("value", [0, -1, 193, True, 1.5])
def test_invalid_ar_width(tiny_config, value):
    mods = [dict(m) for m in tiny_config.modalities]
    next(m for m in mods if m["name"] == "images")["d_ar"] = value
    with pytest.raises(ValueError, match="d_ar"):
        AstroPT3Config(**{**tiny_config.to_dict(), "modalities": mods})


def test_scalar_width_cannot_change(tiny_config):
    mods = [dict(m) for m in tiny_config.modalities]
    next(m for m in mods if m["name"] == "photometry")["d_ar"] = 1
    with pytest.raises(ValueError, match="scalar.*d_ar"):
        AstroPT3Config(**{**tiny_config.to_dict(), "modalities": mods})


def test_normal_prior_matches_distribution():
    y = torch.randn(7, 13, requires_grad=True)
    expected = -torch.distributions.Normal(0, 1).log_prob(y).sum(-1)
    torch.testing.assert_close(std_normal_nll(y), expected)
    std_normal_nll(y).sum().backward()
    torch.testing.assert_close(y.grad, y)
    assert torch.equal(std_normal_nll(y[:, :0]), torch.zeros(7))


@pytest.mark.parametrize("split", [False, True])
def test_likelihood_components_and_flow_gradient(tiny_config, split):
    config = config_with_split(tiny_config, split)
    model = AstroPT3Model(config).eval()
    template = short_template(config)
    batch = PackedCollator(config, seq_len=len(template))([template])
    out = model(**batch)
    for name, flow in model.flows.items():
        z, ld = flow(batch["modality_values"][name], batch["modality_positions"][name],
                     flow_object_ids(batch["position_ids"], batch["modality_masks"][name]))
        d = model.ar_dims[name]
        params = model.decoders[name](out.last_hidden_state[left_shift_mask(batch["modality_masks"][name])])
        ar = gmm_nll(z[:, :d], *params)
        prior = std_normal_nll(z[:, d:])
        expected = (ar + prior - ld).mean() if split else (ar - ld).mean()
        assert torch.equal(out.modality_losses[name], expected)
        parts = out.loss_components[name]
        torch.testing.assert_close(out.modality_losses[name], parts["ar"] + parts["prior"] - parts["logdet"])
        assert cast(Encoder, model.encoders[name]).c_fc.in_features == d
        assert model.decoders[name].d == d
        assert out.predictions[name].shape == z.shape
        assert torch.count_nonzero(out.predictions[name][:, d:]) == 0
        if split:
            grad = torch.autograd.grad(prior.sum(), z, retain_graph=True)[0]
            assert grad[:, d:].abs().sum() > 0
            assert grad[:, :d].abs().sum() == 0
    out.loss.backward()
    for flow in model.flows.values():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in flow.parameters())
    report = evaluate(model, batches=[batch])
    assert set(report["loss_components"]) == set(model.flows)


def test_default_and_explicit_full_width_identical(tiny_config):
    mods = [{k: v for k, v in m.items() if k != "d_ar"} for m in tiny_config.modalities]
    default = AstroPT3Config(**{**tiny_config.to_dict(), "modalities": mods})
    explicit = config_with_split(tiny_config, False)
    torch.manual_seed(42)
    baseline = AstroPT3Model(default).eval()
    torch.manual_seed(42)
    full = AstroPT3Model(explicit).eval()
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, full.state_dict()[key])
    template = short_template(default)
    batch = PackedCollator(default, seq_len=len(template))([template])
    assert torch.equal(baseline(**batch).loss, full(**batch).loss)


def test_prior_coordinates_cannot_leak_to_trunk(tiny_config):
    config = config_with_split(tiny_config, True)
    model = AstroPT3Model(config).eval()
    template = short_template(config)
    batch = PackedCollator(config, seq_len=len(template))([template])
    with torch.no_grad():
        first = model(**batch, compute_loss=False, values_are_latents=True).last_hidden_state
        for name in model.flows:
            batch["modality_values"][name][:, model.ar_dims[name]:] += 100
        second = model(**batch, compute_loss=False, values_are_latents=True).last_hidden_state
    assert torch.equal(first, second)


def test_split_save_load_generation_and_reconstruction(tiny_config, tmp_path):
    config = config_with_split(tiny_config, True)
    model = AstroPT3Model(config).eval()
    model.save_pretrained(tmp_path)
    loaded = AstroPT3Model.from_pretrained(tmp_path).eval()
    assert loaded.config.modalities == config.modalities
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    template = short_template(config)

    def draw(seed, argmax=False):
        return generate(loaded, template, {"spectra"}, n=32, argmax=argmax,
                        temperature=0, generator=torch.Generator().manual_seed(seed))["spectra"]

    a = draw(1)
    assert torch.equal(a, draw(1))
    assert not torch.equal(a, draw(2))
    assert torch.equal(draw(1, True), draw(2, True))
    z, _ = loaded.flows["spectra"](
        a.flatten(0, 1), template.positions["spectra"].repeat(32, 1),
        torch.arange(32).repeat_interleave(3),
    )
    prior = z[:, loaded.ar_dims["spectra"]:]
    assert abs(prior.mean().item()) < 0.04
    assert abs(prior.std().item() - 1) < 0.04
    point = reconstruct(loaded, template)
    assert all(point[name].shape == values.shape for name, values in template.values.items())
