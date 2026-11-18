"""JetFormer flow GPU gates: real trainer init, checkpoint reconstruction, HF parity, DP=2.

Uses tiny conditioners/backbone with all four recipe patch dimensions, including
HSC's 320-wide integer permutations. Batches are local tensors, never LSDB reads.
Run only on an explicitly reserved device: pytest -m gpu tests/test_jetformer_flow_nanotron_gpu.py
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from test_nanotron_gpu import REL_TOL, build_nt_model, make_nt, regroup

pytestmark = pytest.mark.gpu
ASTRO_DIR = Path(__file__).resolve().parents[1]
SEQ_LEN = 128  # fits two objects carrying every recipe modality


@pytest.fixture(scope="module")
def nt():
    return make_nt()


def _trainer(tmp_path, resume=None):
    from nanotron.trainer import DistributedTrainer

    config = yaml.safe_load((ASTRO_DIR / "tests/fixtures/astropt3-test-tiny.yaml").read_text())
    pilot = yaml.safe_load((ASTRO_DIR / "configs/nanotron/astropt3-70m.yaml").read_text())
    config["model"]["model_config"].update(
        modalities=pilot["model"]["model_config"]["modalities"],
        jetformer_flow_steps=2,
        jetformer_flow_hidden=32,
        jetformer_flow_depth=1,
        jetformer_flow_heads=4,
        jetformer_noise_max=0.0,
        jetformer_noise_min=0.0,
    )
    next(
        m for m in config["model"]["model_config"]["modalities"] if m["name"] == "hsc_images"
    )["pre_flow_noise_max"] = 0.5
    config["data_stages"] = pilot["data_stages"]  # parsed only: catalog streams are required, never opened
    config["tokens"].update(sequence_length=SEQ_LEN, micro_batch_size=2)
    config["optimizer"]["learning_rate_scheduler"].update(lr_warmup_steps=0, lr_warmup_style="linear")
    config["checkpoints"]["checkpoints_path"] = str(tmp_path / "checkpoints")
    config["checkpoints"]["resume_checkpoint_path"] = str(resume) if resume is not None else None
    config_path = tmp_path / ("resume.yaml" if resume is not None else "fresh.yaml")
    config_path.write_text(yaml.safe_dump(config))
    # Construct the actual trainer, including generic random init and fp32 optimizer masters.
    # No data loader or external logger is started.
    return DistributedTrainer(str(config_path), benchmark=True)


def _flat_batch(config):
    """Two packed rows, three unequal object spans, shuffled modality ordering."""
    names = config.modality_names()
    batch = {
        "input_ids": torch.zeros((2, SEQ_LEN), dtype=torch.long),
        "position_ids": torch.zeros((2, SEQ_LEN), dtype=torch.long),
    }
    values = {name: [] for name in names}
    positions = {name: [] for name in names}
    for name in names:
        batch[f"{name}_mask"] = torch.zeros((2, SEQ_LEN), dtype=torch.bool)
    generator = torch.Generator().manual_seed(19)
    for row, lengths in enumerate(((3, 2), (1,))):
        cursor = 0
        for object_index, object_length in enumerate(lengths):
            tokens = [config.bos_token_id]
            order = names if object_index == 0 else list(reversed(names))
            for name in order:
                modality = config.modality(name)
                length = 1 if modality["family"] == "scalar" else object_length
                begin, placeholder, end = modality["token_ids"]
                tokens.append(begin)
                start = cursor + len(tokens)
                batch[f"{name}_mask"][row, start : start + length] = True
                tokens.extend([placeholder] * length + [end])
                values[name].append(torch.randn(length, modality["input_size"], generator=generator) * 0.1)
                positions[name].append(
                    torch.arange(length)
                    if modality["pos_type"] == "index"
                    else torch.linspace(0.2, 0.8, length).unsqueeze(-1)
                )
            batch["input_ids"][row, cursor : cursor + len(tokens)] = torch.tensor(tokens)
            batch["position_ids"][row, cursor : cursor + len(tokens)] = torch.arange(len(tokens))
            cursor += len(tokens)
    for name in names:
        batch[f"{name}_values"] = torch.cat(values[name])
        batch[f"{name}_positions"] = torch.cat(positions[name])
    return {key: value.cuda() for key, value in batch.items()}


def _flows(model):
    return model.model.token_position_embeddings.pp_block.flows


def _assert_permutations(flows):
    for flow in flows.values():
        assert set(flow.state_dict()) == set(dict(flow.named_parameters()))
        for stage, block in enumerate(flow.blocks):
            # Checkpoint ABI: immutable CPU randperm(dim), seed 42 + stage, inverse=argsort.
            expected = torch.randperm(
                flow.dim, generator=torch.Generator(device="cpu").manual_seed(42 + stage), device="cpu"
            )
            assert block.permutation.dtype == torch.int64
            assert block.inverse_permutation.dtype == torch.int64
            assert block.permutation.device.type == "cuda"
            assert torch.equal(block.permutation.cpu(), expected)
            assert torch.equal(block.inverse_permutation.cpu(), expected.argsort())
    hsc = flows["hsc_images"].blocks[0].permutation
    assert hsc.numel() == hsc.unique().numel() == 320
    assert hsc.max().item() == 319  # bf16 would round this and corrupt indexing.


def test_full_trainer_initializes_zero_outputs_and_integer_buffers(nt, tmp_path):
    from nanotron.models.astropt3 import modality_flops_per_token

    trainer = _trainer(tmp_path)
    model = trainer.unwrapped_model
    flops = modality_flops_per_token(model.config)
    assert all(value > 0 and value == value for value in flops.values())
    flows = _flows(model)
    _assert_permutations(flows)
    for flow in flows.values():
        for block in flow.blocks:
            assert torch.count_nonzero(block.out_proj.weight) == 0
            assert torch.count_nonzero(block.out_proj.bias) == 0
        for module in flow.modules():
            if isinstance(module, torch.nn.LayerNorm):
                assert torch.equal(module.weight, torch.ones_like(module.weight))
                assert torch.count_nonzero(module.bias) == 0
    assert all(parameter.is_floating_point() for parameter in model.parameters())
    assert not any("permutation" in name for name in trainer.grad_accumulator.parameters)
    for group in trainer.optimizer.param_groups:
        assert all(parameter.is_floating_point() for parameter in group["params"])

    flat = _flat_batch(model.config)
    model.eval()
    with torch.no_grad():
        embedding = model.model.token_position_embeddings.pp_block(**regroup(flat, model.config.modality_names()))
        loss = model(**flat)["loss"]
    assert torch.isfinite(loss)
    assert embedding["input_embeds"].dtype == torch.bfloat16
    for name, flow in flows.items():
        expected = flat[f"{name}_values"].bfloat16().float()
        for block in flow.blocks:
            expected = expected[:, block.permutation]
        assert torch.equal(embedding[f"{name}_z"], expected)
        assert embedding[f"{name}_z"].dtype == torch.float32
        assert torch.count_nonzero(embedding[f"{name}_logdet"]) == 0


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.bool, torch.complex64, torch.float32])
def test_init_context_moves_buffers_without_casting_nonfloat(nt, dtype):
    from nanotron.models import init_on_device_and_dtype

    original = torch.tensor([0, 1], dtype=dtype)
    with init_on_device_and_dtype(device=torch.device("cuda"), dtype=torch.bfloat16):
        module = torch.nn.Module()
        module.register_buffer("value", original, persistent=False)
    expected_dtype = torch.bfloat16 if original.is_floating_point() else dtype
    assert module.value.device.type == "cuda"
    assert module.value.dtype == expected_dtype
    assert torch.equal(module.value.cpu(), original.to(expected_dtype))
    assert not module.state_dict()


def test_full_trainer_checkpoint_reconstructs_exact_latents_and_permutations(nt, tmp_path):
    from nanotron.models.astropt3 import AstroPT3ForTraining

    trainer = _trainer(tmp_path)
    model = trainer.unwrapped_model
    assert isinstance(model, AstroPT3ForTraining)
    assert trainer.config.tokens is not None
    flat = _flat_batch(model.config)
    # The baseline scheduler starts at LR=0; the second update exercises nonzero outputs.
    model.train()
    trainer.optimizer.zero_grad()
    for step in (1, 2):
        model.set_jet_noise_frac(step / trainer.config.tokens.train_steps)
        loss = model(**flat)["loss"]
        assert torch.isfinite(loss)
        trainer.grad_accumulator.backward(loss)
        assert all(
            torch.isfinite(entry["fp32_grad"]).all()
            for entry in trainer.grad_accumulator.fp32_grad_buffers.values()
        )
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        trainer.optimizer.zero_grad()
    assert all(
        torch.count_nonzero(block.out_proj.weight) > 0
        for flow in _flows(model).values()
        for block in flow.blocks
    )
    trainer.iteration_step = trainer.metadata.last_train_step = 2
    trainer.metadata.consumed_train_samples = 4
    trainer.metadata.consumed_tokens_total = 256
    trainer.metadata.current_stage.consumed_train_samples = 4
    trainer.metadata.current_stage.consumed_tokens_per_dataset_folder = {"synthetic": 256}
    model.eval()
    with torch.no_grad():
        before = model.model(**regroup(flat, model.config.modality_names()))
    checkpoint = trainer.save_checkpoint()
    # Construction must reconstruct buffers without reading them from checkpoint tensors.
    resumed = _trainer(tmp_path, resume=checkpoint)
    loaded = resumed.unwrapped_model.eval()
    assert resumed.init_checkpoint_path is not None
    assert resumed.metadata.last_train_step == 2
    _assert_permutations(_flows(loaded))
    original_parameters = dict(model.named_parameters())
    for name, parameter in loaded.named_parameters():
        assert torch.equal(parameter, original_parameters[name]), name
    with torch.no_grad():
        after = loaded.model(**regroup(flat, loaded.config.modality_names()))
    for name in _flows(model):  # scalar modalities have no flow, z or logdet
        for suffix in ("z", "logdet"):
            key = f"{name}_{suffix}"
            assert torch.equal(after[key], before[key]), key
        for old_block, new_block in zip(_flows(model)[name].blocks, _flows(loaded)[name].blocks):
            assert torch.equal(old_block.permutation, new_block.permutation)
            assert torch.equal(old_block.inverse_permutation, new_block.inverse_permutation)


def test_hf_nanotron_parity_and_exact_conversion(nt, tmp_path):
    from astropt3 import AstroPT3Model
    from astropt3.flows import flow_object_ids

    trainer = _trainer(tmp_path)
    model = trainer.unwrapped_model.eval()
    config = model.config
    hf_config = nt.to_hf.get_hf_config(config)
    for field in ("jetformer_flow_depth", "jetformer_flow_heads"):
        assert getattr(hf_config, field) == getattr(config, field)
        assert getattr(nt.from_hf.get_nanotron_config(hf_config), field) == getattr(config, field)
    hf_model = AstroPT3Model(hf_config).cuda().bfloat16().eval()
    for flow in hf_model.flows.values():
        assert all(torch.count_nonzero(block.out_proj.weight) == 0 for block in flow.blocks)
    # Nontrivial conditioners catch missing coupling weights, not just initial permutations.
    with torch.no_grad():
        for flow in hf_model.flows.values():
            for block in flow.blocks:
                block.out_proj.weight.normal_(std=0.01)
                block.out_proj.bias.fill_(0.01)
    nt.from_hf.convert_hf_to_nt(hf_model, model, config)
    mapping = nt.convert_weights.get_weight_mapping(config, nt_to_hf=False)
    assert set(mapping) == set(dict(hf_model.named_parameters()))
    assert set(mapping.values()) == set(dict(model.named_parameters()))
    _assert_permutations(hf_model.flows)
    _assert_permutations(_flows(model))
    flat = _flat_batch(config)
    kwargs = regroup(flat, config.modality_names())
    with torch.no_grad():
        predictions = model.model(**kwargs)
        nt_output = model(**flat)
        hf_output = hf_model(**kwargs)
        for name in config.modality_names():
            if name in hf_model.flows:  # scalar modalities have no flow
                z, logdet = hf_model.flows[name](
                    flat[f"{name}_values"].bfloat16(),
                    flat[f"{name}_positions"],
                    flow_object_ids(flat["position_ids"], flat[f"{name}_mask"]),
                )
                assert torch.equal(z, predictions[f"{name}_z"])
                assert torch.equal(logdet, predictions[f"{name}_logdet"])
            assert nt_output[f"{name}_loss"].item() == pytest.approx(
                hf_output.modality_losses[name].item(), rel=REL_TOL
            )
    assert nt_output["loss"].item() == pytest.approx(hf_output.loss.item(), rel=REL_TOL)
    for family, loss in hf_output.family_losses.items():
        assert nt_output[f"{family}_family_loss"].item() == pytest.approx(loss.item(), rel=REL_TOL)

    hf_back = AstroPT3Model(hf_config).cuda().bfloat16().eval()
    nt.to_hf.convert_nt_to_hf(model, hf_back, config)
    for name, parameter in hf_back.state_dict().items():
        assert torch.equal(parameter, hf_model.state_dict()[name]), name
    hf_back.save_pretrained(tmp_path / "hf")
    hf_loaded = AstroPT3Model.from_pretrained(tmp_path / "hf", dtype=torch.bfloat16).cuda().eval()
    _assert_permutations(hf_loaded.flows)
    nt_back = build_nt_model(nt, nt.from_hf.get_nanotron_config(hf_loaded.config)).eval()
    nt.from_hf.convert_hf_to_nt(hf_loaded, nt_back, config)
    for name, parameter in nt_back.named_parameters():
        assert torch.equal(parameter, dict(model.named_parameters())[name]), name
    with torch.no_grad():
        # Conversion selects fused-kernel defaults; weights are exact, bf16 reductions need tolerance.
        torch.testing.assert_close(nt_back(**flat)["loss"], nt_output["loss"], rtol=1e-5, atol=1e-4)


def test_rejects_context_parallel(nt):
    from test_nanotron_gpu import tiny_nt_config

    groups = {name: SimpleNamespace(size=lambda: 1) for name in ("dp_pg", "tp_pg", "pp_pg")}
    groups["cp_pg"] = SimpleNamespace(size=lambda: 2)
    with pytest.raises(AssertionError, match="CP=1"):
        nt.model_cls(
            config=tiny_nt_config(nt),
            parallel_context=SimpleNamespace(**groups),
            parallel_config=nt.convert_weights.make_parallel_config(),
        )


def test_dp2_flow_gradients_stay_synced(nt):
    import os
    import subprocess
    import sys

    if torch.cuda.device_count() < 2:
        pytest.skip("DP=2 check needs 2 visible GPUs")
    result = subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
            "--rdzv-backend=c10d", "--rdzv-endpoint=localhost:0",
            str(ASTRO_DIR / "scripts" / "dp2_flow_check.py"),
        ],
        cwd=ASTRO_DIR.parent,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
    assert "DP2 FLOW CHECK PASS" in result.stdout
