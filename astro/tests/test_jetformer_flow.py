"""Whole-span flow correctness, packing isolation and HF integration."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from legacy_fixture import make_record
from test_model import single_band_config as single_band_config

from astropt3 import AstroPT3Config, AstroPT3Model
from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.flows import FlowTransformerBlock, JetFormerFlow, flow_object_ids
from astropt3.generation import generate, reconstruct

RECIPES = Path(__file__).resolve().parents[1] / "configs/nanotron"


def small_flow(**kwargs):
    return JetFormerFlow(
        dim=4, steps=2, hidden_dim=8, depth=1, num_heads=2,
        pos_type="continuous", **kwargs,
    )


def activate(flow):
    with torch.no_grad():
        for block in flow.blocks:
            block.out_proj.weight.normal_(std=0.08)
            block.out_proj.bias.normal_(std=0.02)
    return flow


def test_transformer_matches_torch_encoder_layer():
    torch.manual_seed(1)
    block = FlowTransformerBlock(8, 2).eval()
    reference = torch.nn.TransformerEncoderLayer(
        8, 2, 32, dropout=0, activation="gelu", batch_first=True, norm_first=True,
    ).eval()
    state = block.state_dict()
    reference.load_state_dict({
        "self_attn.in_proj_weight": state["qkv.weight"],
        "self_attn.in_proj_bias": state["qkv.bias"],
        "self_attn.out_proj.weight": state["attn_out.weight"],
        "self_attn.out_proj.bias": state["attn_out.bias"],
        **{key: value for key, value in state.items() if key.startswith(("norm", "linear"))},
    })
    x = torch.randn(2, 3, 8)
    valid = torch.tensor([[True, True, True], [True, True, False]])
    torch.testing.assert_close(
        block(x, valid[:, None, None, :]),
        reference(x, src_key_padding_mask=~valid), atol=2e-6, rtol=2e-6,
    )


@pytest.mark.parametrize("pos_type", ["index", "continuous"])
def test_roundtrip_and_logdet_with_ragged_objects(pos_type):
    torch.manual_seed(2)
    flow = activate(JetFormerFlow(6, 3, 12, 2, 3, pos_type, 8)).double().eval()
    values = torch.randn(5, 6, dtype=torch.float64)
    ids = torch.tensor([0, 0, 1, 1, 1])
    pos = torch.tensor([0, 1, 0, 1, 2])
    if pos_type == "continuous":
        pos = pos.double().unsqueeze(-1) / 10
    z, ld = flow(values, pos, ids)
    recovered, inverse_ld = flow(z, pos, ids, reverse=True)
    torch.testing.assert_close(recovered, values, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(ld, -inverse_ld, atol=1e-9, rtol=1e-9)


def test_logdet_matches_full_cross_token_jacobian():
    torch.manual_seed(3)
    flow = activate(small_flow()).double().eval()
    values = torch.randn(2, 4, dtype=torch.float64)
    pos = torch.tensor([[0.1], [0.3]], dtype=torch.float64)
    ids = torch.zeros(2, dtype=torch.long)
    _, ld = flow(values, pos, ids)
    jacobian = torch.autograd.functional.jacobian(
        lambda flat: flow(flat.reshape(2, 4), pos, ids)[0].reshape(-1),
        values.reshape(-1),
    )
    _, expected = torch.linalg.slogdet(jacobian)
    torch.testing.assert_close(ld.sum(), expected, atol=1e-9, rtol=1e-9)


def test_cross_patch_context_but_no_cross_object_context_or_padding_effect():
    torch.manual_seed(4)
    flow = activate(small_flow()).double().eval()
    values = torch.randn(5, 4, dtype=torch.float64)
    ids = torch.tensor([0, 0, 1, 1, 1])
    pos = torch.arange(5, dtype=torch.float64).unsqueeze(-1)
    z, ld = flow(values, pos, ids)
    changed = values.clone()
    changed[1] += 3
    altered, altered_ld = flow(changed, pos, ids)
    assert not torch.allclose(z[0], altered[0])
    torch.testing.assert_close(z[2:], altered[2:], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(ld[2:], altered_ld[2:], atol=1e-12, rtol=1e-12)
    separate, separate_ld = flow(values[:2], pos[:2], ids[:2])
    torch.testing.assert_close(z[:2], separate, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(ld[:2], separate_ld, atol=1e-12, rtol=1e-12)


def test_checkpointed_backward_matches_eager():
    torch.manual_seed(5)
    checkpointed = activate(small_flow()).train()
    eager = deepcopy(checkpointed).eval()
    x = torch.randn(5, 4)
    pos = torch.randn(5, 1)
    ids = torch.tensor([0, 0, 1, 1, 1])
    for flow in (checkpointed, eager):
        z, ld = flow(x, pos, ids)
        (z.square().mean() - ld.mean()).backward()
    for left, right in zip(checkpointed.parameters(), eager.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, atol=2e-6, rtol=2e-5)


def test_empty_modality_connects_all_parameters():
    flow = small_flow().train()
    z, ld = flow(torch.empty(0, 4), torch.empty(0, 1), torch.empty(0, dtype=torch.long))
    (z.sum() + ld.sum()).backward()
    assert all(p.grad is not None and not p.grad.any() for p in flow.parameters())


def test_permutations_are_exact_reconstructible_derived_buffers():
    flow = JetFormerFlow(320, steps=2, hidden_dim=8, depth=1, num_heads=2).bfloat16().eval()
    for block in flow.blocks:
        permutation = block.get_buffer("permutation")
        assert permutation.dtype == torch.int64
        torch.testing.assert_close(permutation.sort().values, torch.arange(320))
        torch.testing.assert_close(permutation[block.get_buffer("inverse_permutation")], torch.arange(320))
    assert not any("permutation" in key for key in flow.state_dict())
    torch.manual_seed(12345)
    restored = JetFormerFlow(320, steps=2, hidden_dim=8, depth=1, num_heads=2).bfloat16().eval()
    restored.load_state_dict(flow.state_dict(), strict=True)
    x = torch.randn(3, 320).bfloat16()
    pos = torch.arange(3)
    ids = torch.zeros(3, dtype=torch.long)
    z, ld = flow(x, pos, ids)
    torch.testing.assert_close(z, restored(x, pos, ids)[0], atol=0, rtol=0)
    torch.testing.assert_close(ld, torch.zeros(3), atol=0, rtol=0)
    torch.testing.assert_close(flow(z, pos, ids, reverse=True)[0], x.float(), atol=0, rtol=0)


def test_packed_object_ids_do_not_merge_adjacent_rows():
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3], [0, 1, 2, 3, 0, 0, 0]])
    mask = torch.tensor([[0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    ids = flow_object_ids(positions, mask)
    assert ids.tolist() == [1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError, match="position_ids"):
        flow_object_ids(None, mask)  # pyright: ignore[reportArgumentType] -- exercise invalid input


@pytest.mark.parametrize("field,value", [
    ("jetformer_flow_steps", 0),
    ("jetformer_flow_depth", True), ("jetformer_flow_hidden", 15),
])
def test_reject_invalid_flow_config(single_band_config, field, value):
    kwargs = {**single_band_config.to_dict(), field: value}
    with pytest.raises(ValueError):
        AstroPT3Config(**kwargs)


def test_hf_packing_normalization_backward_and_save_load(tmp_path, single_band_config, tiny_config):
    torch.manual_seed(6)
    cfg = single_band_config
    model = AstroPT3Model(cfg)
    for flow in model.flows.values():
        assert isinstance(flow, JetFormerFlow)
        for block in flow.blocks:
            assert not block.get_parameter("out_proj.weight").any()
            assert not block.get_parameter("out_proj.bias").any()
        activate(flow)
    record = make_record(3, image_only_fraction=0)
    obj = ObjectSequencer(cfg).build(record)
    original = ObjectSequencer(tiny_config).build(record)
    for name in obj.values:
        assert torch.equal(obj.values[name], original.values[name])
        assert torch.equal(obj.positions[name], original.positions[name])
    batch = PackedCollator(cfg, seq_len=2 * len(obj) + 10)([obj, obj])
    model.train()
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    model.eval()
    tight = PackedCollator(cfg, seq_len=len(obj))([obj])
    with torch.no_grad():
        expected = model(**tight).loss
        torch.testing.assert_close(model(**batch).loss, expected, atol=1e-4, rtol=1e-5)
    model.save_pretrained(tmp_path / "model")
    restored = AstroPT3Model.from_pretrained(tmp_path / "model").eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**tight).loss, expected, atol=1e-6, rtol=1e-6)
    assert restored.config.jetformer_flow_depth == 1


def test_generation_uses_latent_prefixes_and_complete_span_inverse(single_band_config):
    torch.manual_seed(7)
    cfg = single_band_config
    model = AstroPT3Model(cfg).eval()
    for flow in model.flows.values():
        activate(flow)
    obj = ObjectSequencer(cfg).build(make_record(3, image_only_fraction=0))
    # Only spectra are generated; images/scalars are conditioned on when preceding them.
    first = generate(model, obj, {"spectra"}, n=2, generator=torch.Generator().manual_seed(5))
    second = generate(model, obj, {"spectra"}, n=2, generator=torch.Generator().manual_seed(5))
    assert first["spectra"].shape == (2, 122, 64)
    assert torch.isfinite(first["spectra"]).all()
    torch.testing.assert_close(first["spectra"], second["spectra"], atol=0, rtol=0)
    reconstructed = reconstruct(model, obj)
    assert reconstructed["legacy_north_g"].shape == (144, 64)
    assert torch.isfinite(reconstructed["spectra"]).all()


def load_recipe(name):
    import yaml

    return yaml.safe_load((RECIPES / name).read_text())


@pytest.mark.parametrize("recipe", sorted(RECIPES.glob("*.yaml")), ids=lambda p: p.stem)
def test_flow_width_follows_trunk_rule(recipe):
    config = load_recipe(recipe.name)["model"]["model_config"]
    assert config["jetformer_flow_hidden"] == min(config["hidden_size"] // 4, 512)
    assert (config["jetformer_flow_steps"], config["jetformer_flow_depth"], config["jetformer_flow_heads"]) == (8, 4, 8)


@pytest.mark.parametrize("name,flow", [
    ("../controls/astropt3-70m-north", 72_531_456),
    ("astropt3-70m", 98_899_456),
    ("astropt3-410m", 386_533_888),
])
def test_recipe_flow_parameter_counts(name, flow):
    config = load_recipe(f"{name}.yaml")["model"]["model_config"]
    config["_attn_implementation"] = "sdpa"  # Same geometry, no CUDA imports for a CPU count gate.
    with torch.device("meta"):
        model = AstroPT3Model(AstroPT3Config(**config))
    assert sum(p.numel() for p in model.flows.parameters()) == flow
