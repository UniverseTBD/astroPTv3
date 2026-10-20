"""Phase 3 GPU verification (gpu-marked; excluded from the default CPU run).

Covers the plan's Phase 3 gates:

- tiny-config HF<->nanotron conversion + forward-loss parity on a fixed
  synthetic batch (bf16 tolerance: nanotron rotates at absolute row positions
  while HF restarts per object — RoPE is relative so the scores agree, but
  float trajectories differ; plus flash-attn vs sdpa);
- replicated-module gradient check across TP=2 (scripts/tp2_grad_check.py
  under torchrun; skipped with fewer than 2 visible GPUs);
- 50-step nanotron run on synthetic data (loss decreases) whose final
  checkpoint converts, loads via ``AutoModel.from_pretrained`` and reproduces
  the val loss.

Environment: nanotron (editable ../nanotron) + flash-attn + astropt3 on a
CUDA machine — the ``[train]`` extra. flash-attn has no prebuilt wheel for
every torch/cuda combo; a torch 2.8 + cu12 venv with the matching
Dao-AILab release wheel works on A100 (see PLAN Phase 3 notes). Run:

    pytest -m gpu tests/test_nanotron_gpu.py -v
"""

import importlib
import json
import os
import re
import subprocess
import sys
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

pytestmark = pytest.mark.gpu

ASTRO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ASTRO_DIR.parent
NANOTRON_DIR = REPO_ROOT / "nanotron"
TOOLS_DIR = NANOTRON_DIR / "tools" / "astropt3"

MBS = 2
SEQ_LEN = 896
# bf16 forward parity: different-but-equivalent RoPE position conventions and
# flash-attn vs sdpa kernels bound the achievable agreement
REL_TOL = 3e-2
LOSS_RE = re.compile(r"lm_loss: ([0-9.eE+-]+)")


def run_train(config: dict, workdir: Path, name: str, timeout: int = 5400) -> str:
    """torchrun a nanotron config to completion; shared with test_jetformer_gpu.py."""
    config_path = workdir / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            str(NANOTRON_DIR / "run_train.py"),
            "--config-file",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
    return result.stdout


def losses_from(stdout: str) -> list[float]:
    return [float(m) for m in LOSS_RE.findall(stdout)]


def _load_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot load YAML from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load JSON from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return value


def _dist_env(port: int = 29123) -> dict:
    env = {
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }
    for key, value in env.items():
        os.environ.setdefault(key, value)
    return env


def make_nt():
    """Late imports so CPU-only collection of this module stays possible.

    Shared with test_jetformer_gpu.py (both wrap it in a session fixture; the
    ParallelContext is process-global so building it twice is harmless).
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    pytest.importorskip("nanotron")
    pytest.importorskip("flash_attn")
    _dist_env()
    sys.path.insert(0, str(TOOLS_DIR))
    config_module = importlib.import_module("nanotron.config")
    models = importlib.import_module("nanotron.models")
    model_module = importlib.import_module("nanotron.models.astropt3")
    parallel = importlib.import_module("nanotron.parallel")
    trainer = importlib.import_module("nanotron.trainer")
    return SimpleNamespace(
        config_cls=config_module.AstroPT3Config,
        model_cls=model_module.AstroPT3ForTraining,
        build_model=models.build_model,
        mark_tied_parameters=trainer.mark_tied_parameters,
        convert_weights=importlib.import_module("convert_weights"),
        to_hf=importlib.import_module("convert_nanotron_to_hf"),
        from_hf=importlib.import_module("convert_hf_to_nanotron"),
        parallel_context=parallel.ParallelContext(
            data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1
        ),
        serialize=importlib.import_module("nanotron.serialize"),
    )


@pytest.fixture(scope="session")
def nt():
    return make_nt()


def tiny_nt_config(nt, **overrides):
    overrides.setdefault("loss_aggregation", "family")
    return nt.config_cls(
        hidden_size=64,
        num_hidden_layers=4,  # crosses one NoPE layer (idx 3)
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=4096,
        rope_theta=100000.0,
        no_rope_layer=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        tie_word_embeddings=False,
        attention_bias=False,
        _attn_implementation="flash_attention_2",
        _use_qkv_packed=True,
        _use_doc_masking=True,
        log_attn_probs=False,
        **overrides,
    )


def build_nt_model(nt, nt_config):
    parallel_config = nt.convert_weights.make_parallel_config()
    model = nt.build_model(
        model_builder=lambda: nt.model_cls(
            config=nt_config,
            parallel_context=nt.parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=nt.parallel_context,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    nt.mark_tied_parameters(model=model, parallel_context=nt.parallel_context)
    return model


def micro_batch(hf_config, device="cuda", skip=0):
    from astropt3.data.nanotron_loader import PackedMicroBatches

    stream = iter(PackedMicroBatches(hf_config, MBS, SEQ_LEN))
    flat = next(islice(stream, skip, None))
    return {k: v.to(device) for k, v in flat.items()}


def regroup(flat: dict, names) -> dict:
    """Flat nanotron micro-batch -> HF AstroPT3Model kwargs."""
    from astropt3.data.nanotron_loader import regroup_micro_batch

    return regroup_micro_batch(flat, names)


@pytest.fixture(scope="session")
def matched_models(nt):
    """HF tiny model and a nanotron model carrying identical weights."""
    import astropt3  # noqa: F401 -- registers Auto classes
    from astropt3 import AstroPT3Model

    nt_config = tiny_nt_config(nt)
    hf_config = nt.to_hf.get_hf_config(nt_config)
    torch.manual_seed(0)
    hf_model = AstroPT3Model(hf_config)
    nt_model = build_nt_model(nt, nt_config)
    nt.from_hf.convert_hf_to_nt(hf_model, nt_model, nt_config)
    torch.nn.Module.cuda(hf_model)
    torch.nn.Module.to(hf_model, dtype=torch.bfloat16)
    hf_model.eval()
    return hf_model, nt_model, nt_config, hf_config


def test_forward_loss_parity(nt, matched_models):
    hf_model, nt_model, nt_config, hf_config = matched_models
    flat = micro_batch(hf_config)
    names = hf_config.modality_registry().names()
    with torch.no_grad():
        nt_out = nt_model(**flat)
        hf_out = hf_model(**regroup(flat, names))

    nt_loss = nt_out["loss"].item()
    hf_loss = hf_out.loss.item()
    assert nt_loss == pytest.approx(hf_loss, rel=REL_TOL), f"NT={nt_loss}, HF={hf_loss}"
    for name in names:
        nt_mod = nt_out[f"{name}_loss"].item()
        if name not in hf_out.modality_losses:
            # absent from this batch: HF omits the key, nanotron reports 0.0
            assert nt_mod == 0.0, f"{name}: NT={nt_mod} for an absent modality"
            continue
        hf_mod = hf_out.modality_losses[name].item()
        assert nt_mod == pytest.approx(hf_mod, rel=REL_TOL), (
            f"{name}: NT={nt_mod}, HF={hf_mod}"
        )
    for family, hf_family in hf_out.family_losses.items():
        nt_family = nt_out[f"{family}_family_loss"].item()
        assert nt_family == pytest.approx(hf_family.item(), rel=REL_TOL), (
            f"{family}: NT={nt_family}, HF={hf_family.item()}"
        )


def test_conversion_roundtrip_exact(nt, matched_models):
    from astropt3 import AstroPT3Model

    hf_model, nt_model, nt_config, hf_config = matched_models
    hf_back = AstroPT3Model(hf_config)
    torch.nn.Module.cuda(hf_back)
    torch.nn.Module.to(hf_back, dtype=torch.bfloat16)
    nt.to_hf.convert_nt_to_hf(nt_model, hf_back, nt_config)
    original = hf_model.state_dict()
    for key, value in hf_back.state_dict().items():
        assert torch.equal(value, original[key]), f"roundtrip drift in {key}"


def test_tp2_replicated_module_gradients(nt):
    if torch.cuda.device_count() < 2:
        pytest.skip("TP=2 check needs 2 visible GPUs")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_port=29511",
            str(ASTRO_DIR / "scripts" / "tp2_grad_check.py"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "TP2 GRAD CHECK PASS" in result.stdout


def test_50step_synthetic_run_and_checkpoint_conversion(nt, tmp_path_factory):
    """torchrun 50 steps on synthetic data; convert; reload; reproduce loss."""
    workdir = tmp_path_factory.mktemp("nanotron_smoke")
    config = _load_yaml(
        ASTRO_DIR / "tests" / "fixtures" / "astropt3-test-tiny.yaml"
    )
    config["checkpoints"]["checkpoints_path"] = str(workdir / "checkpoints")
    config_path = workdir / "test-tiny.yaml"
    try:
        config_path.write_text(yaml.safe_dump(config))
    except OSError as error:
        raise RuntimeError(f"cannot write {config_path}") from error

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--master_port=29512",
            str(NANOTRON_DIR / "run_train.py"),
            "--config-file",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"

    losses = [float(m) for m in re.findall(r"lm_loss: ([0-9.]+)", result.stdout)]
    assert len(losses) >= 40, f"expected ~50 logged losses, got {len(losses)}"
    assert losses[-1] < 0.8 * losses[0], f"nanotron smoke did not learn: {losses[0]:.4f} -> {losses[-1]:.4f}"

    checkpoint = workdir / "checkpoints" / "50"
    assert (checkpoint / "model_config.json").exists()

    # convert in a subprocess: the converter builds its own ParallelContext
    save_path = workdir / "hf"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--master_port=29513",
            str(TOOLS_DIR / "convert_nanotron_to_hf.py"),
            f"--checkpoint_path={checkpoint}",
            f"--save_path={save_path}",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    # the converted checkpoint loads through the Auto classes...
    from transformers import AutoModel

    import astropt3  # noqa: F401

    hf_model = AutoModel.from_pretrained(save_path)
    torch.nn.Module.cuda(hf_model)
    torch.nn.Module.to(hf_model, dtype=torch.bfloat16)
    hf_model.eval()
    assert hf_model.config.model_type == "astropt3"

    # ...and reproduces the nanotron model's val loss on a held-out batch
    nt_config = nt.config_cls(**_load_json(checkpoint / "model_config.json"))
    nt_model = build_nt_model(nt, nt_config)
    nt.serialize.load_weights(
        model=nt_model, parallel_context=nt.parallel_context, root_folder=checkpoint
    )
    flat = micro_batch(hf_model.config, skip=7)  # not the first training batch
    names = hf_model.config.modality_registry().names()
    with torch.no_grad():
        nt_loss = nt_model(**flat)["loss"].item()
        hf_loss = hf_model(**regroup(flat, names)).loss.item()
    assert hf_loss == pytest.approx(nt_loss, rel=REL_TOL), f"NT={nt_loss}, HF={hf_loss}"
