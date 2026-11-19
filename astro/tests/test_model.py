import pytest
import torch
from legacy_fixture import record_stream

from astropt3 import AstroPT3Config, AstroPT3Model
from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.flows import flow_object_ids


@pytest.fixture(scope="session")
def single_band_config(tiny_config):
    modalities = [
        m for m in tiny_config.modalities
        if m["family"] != "image" or m["name"] == "legacy_north_g"
    ]
    return AstroPT3Config(**{**tiny_config.to_dict(), "modalities": modalities})


@pytest.fixture(scope="session")
def single_band_sequencer(single_band_config):
    return ObjectSequencer(single_band_config)


@pytest.fixture(scope="session")
def single_band_collator(single_band_config):
    return PackedCollator(single_band_config, seq_len=896)


@pytest.fixture(scope="session")
def single_band_model(single_band_config):
    torch.manual_seed(0)
    return AstroPT3Model(single_band_config).eval()


def test_forward_backward(single_band_config, single_band_sequencer, single_band_collator):
    torch.manual_seed(0)
    model = AstroPT3Model(single_band_config)
    model.train()
    batch = single_band_collator([single_band_sequencer.build(r) for r in record_stream(4)])
    out = model(**batch)
    assert torch.isfinite(out.loss)
    assert set(out.modality_losses) == {"legacy_north_g", "spectra", "Z", "ebv", "photometry"}
    assert set(out.family_losses) == {"image", "spectrum", "scalar"}
    assert all(torch.isfinite(v) for v in out.modality_losses.values())
    out.loss.backward()
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, f"params without grad: {missing}"


def test_pad_invariance(single_band_model, single_band_config, single_band_sequencer, full_record):
    """Trailing pad tokens must not change the loss at all."""
    obj = single_band_sequencer.build(full_record)
    tight = PackedCollator(single_band_config, seq_len=len(obj))([obj])
    padded = PackedCollator(single_band_config, seq_len=520)([obj])
    with torch.no_grad():
        loss_tight = single_band_model(**tight).loss
        loss_padded = single_band_model(**padded).loss
    assert torch.allclose(loss_tight, loss_padded, atol=1e-6)


def test_doc_mask_blocks_cross_object_attention(
    single_band_model, single_band_config, single_band_sequencer
):
    """Perturbing object A's pixels must not move object B's hidden states."""
    records = list(record_stream(2))
    objs = [single_band_sequencer.build(r) for r in records]
    collator = PackedCollator(single_band_config, seq_len=len(objs[0]) + len(objs[1]))
    batch = collator(objs)
    assert batch["input_ids"].shape[0] == 1  # both objects share one row

    perturbed = {
        "input_ids": batch["input_ids"],
        "position_ids": batch["position_ids"],
        "modality_masks": batch["modality_masks"],
        "modality_positions": batch["modality_positions"],
        "modality_values": {
            k: v.clone() for k, v in batch["modality_values"].items()
        },
    }
    n_a = objs[0].values["legacy_north_g"].shape[0]
    perturbed["modality_values"]["legacy_north_g"][:n_a] += 10.0

    with torch.no_grad():
        h0 = single_band_model(**batch, compute_loss=False).last_hidden_state
        h1 = single_band_model(**perturbed, compute_loss=False).last_hidden_state

    start_b = len(objs[0])
    assert not torch.allclose(h0[0, :start_b], h1[0, :start_b])  # A moved
    assert torch.equal(h0[0, start_b:], h1[0, start_b:])  # B bit-identical


def test_family_loss_pools_tokens_and_normalizes_patch_dims(
    single_band_model, single_band_sequencer, single_band_collator, full_record
):
    """Families pool their tokens; patch NLLs are per dimension; families 1:1:0.1."""
    without_z = {k: v for k, v in full_record.items() if k != "Z"}
    batch = single_band_collator([
        single_band_sequencer.build(full_record),
        single_band_sequencer.build({**without_z, "object_id": "other"}),
    ])
    with torch.no_grad():
        out = single_band_model(**batch)
    masks, losses = batch["modality_masks"], out.modality_losses
    n = {name: int(masks[name].sum()) for name in masks}
    assert (n["Z"], n["ebv"], n["photometry"]) == (1, 2, 2)
    scalar = (n["Z"] * losses["Z"] + n["ebv"] * losses["ebv"] + n["photometry"] * losses["photometry"]) / 5
    torch.testing.assert_close(out.family_losses["scalar"], scalar)
    torch.testing.assert_close(out.family_losses["image"], losses["legacy_north_g"] / 64)
    torch.testing.assert_close(out.family_losses["spectrum"], losses["spectra"] / 256)
    expected = (out.family_losses["image"] + out.family_losses["spectrum"] + 0.1 * out.family_losses["scalar"]) / 2.1
    torch.testing.assert_close(out.loss, expected)


def test_loss_matches_manual_computation(
    single_band_model, single_band_sequencer, single_band_collator, full_record
):
    """outputs.loss must equal NLL_GMM(z) - logdet recomputed from the
    model's own flow + GMM head, for each patch modality.

    Scalar-free build: the scalar spans' GMM NLL terms are covered by
    test_scalar_modalities' manual-loss check.
    """
    from astropt3.modalities import gmm_nll
    from astropt3.modeling_astropt3 import left_shift_mask

    batch = single_band_collator([single_band_sequencer.build(full_record, include_scalars=False)])
    with torch.no_grad():
        out = single_band_model(**batch)
        manual = []
        for m in ("legacy_north_g", "spectra"):
            mask = batch["modality_masks"][m]
            hidden = out.last_hidden_state[left_shift_mask(mask)]
            z, logdet = single_band_model.flows[m](
                batch["modality_values"][m], batch["modality_positions"][m],
                flow_object_ids(batch["position_ids"], mask),
            )
            logits_pi, mu, log_sigma = single_band_model.decoders[m](hidden)
            nll = gmm_nll(z, logits_pi, mu, log_sigma)
            # pooled per family, per dimension: one patch modality per family here
            manual.append((nll - logdet).mean() / single_band_model.modality_input_sizes[m])
    expected = torch.stack(manual).mean()
    assert torch.allclose(out.loss, expected, atol=1e-6)
