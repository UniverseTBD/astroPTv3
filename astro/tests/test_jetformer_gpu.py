"""JetFormer-tokeniser GPU verification (gpu-marked; excluded from CPU runs).

Covers the jetformer GPU gates:

1. HF <-> nanotron loss parity on a fixed synthetic batch after conversion,
   eval mode (noise off), both conversion directions;
2. TP=2: flow + GMM-head gradients bit-identical across ranks, with the
   noise curriculum ACTIVE (drawn under the tp_synced random state);
3. 50-step CUDA smoke — the likelihood loss (NLL - logdet, may be negative)
   decreases; the final checkpoint converts to HF and reproduces the loss.

Kill/resume with exact object-stream reproduction is retired with the old
dataset-checkpoint state (ADR 0015): an LSDB stream restarts fresh on every
resume, so there is no stream position to verify a replay-free continuation
against.

Run (reserved GPU node, [train] env):
    pytest -m gpu tests/test_jetformer_gpu.py -v
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from test_nanotron_gpu import (
    REL_TOL,
    build_nt_model,
    losses_from,
    make_nt,
    micro_batch,
    regroup,
    run_train,
    tiny_nt_config,
)

pytestmark = pytest.mark.gpu

ASTRO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ASTRO_DIR.parent
NANOTRON_DIR = REPO_ROOT / "nanotron"
TOOLS_DIR = NANOTRON_DIR / "tools" / "astropt3"
JET_YAML = ASTRO_DIR / "tests" / "fixtures" / "astropt3-test-tiny.yaml"


@pytest.fixture(scope="session")
def nt():
    return make_nt()


def tiny_jet_config(nt):
    return tiny_nt_config(nt, jetformer_flow_hidden=32)


@pytest.fixture(scope="session")
def matched_jet_models(nt):
    """HF tiny jetformer model and a nanotron model carrying identical weights."""
    import astropt3  # noqa: F401 -- registers Auto classes
    from astropt3 import AstroPT3Model

    nt_config = tiny_jet_config(nt)
    hf_config = nt.to_hf.get_hf_config(nt_config)
    assert hf_config.jetformer_flow_hidden == 32
    torch.manual_seed(0)
    hf_model = AstroPT3Model(hf_config)
    nt_model = build_nt_model(nt, nt_config)
    nt.from_hf.convert_hf_to_nt(hf_model, nt_model, nt_config)
    hf_model = hf_model.cuda().to(torch.bfloat16).eval()
    nt_model.eval()  # noise curriculum off: deterministic parity
    return hf_model, nt_model, nt_config, hf_config


def test_forward_loss_parity(nt, matched_jet_models):
    hf_model, nt_model, _nt_config, hf_config = matched_jet_models
    flat = micro_batch(hf_config)
    names = hf_config.modality_registry().names()
    with torch.no_grad():
        nt_out = nt_model(**flat)
        hf_out = hf_model(**regroup(flat, names))

    nt_loss = nt_out["loss"].item()
    hf_loss = hf_out.loss.item()
    assert nt_loss == pytest.approx(hf_loss, rel=REL_TOL), (nt_loss, hf_loss)
    for name in names:
        nt_mod = nt_out[f"{name}_loss"].item()
        if name not in hf_out.modality_losses:
            # absent from this batch: HF omits the key, nanotron reports 0.0
            assert nt_mod == 0.0, (name, nt_mod)
            continue
        hf_mod = hf_out.modality_losses[name].item()
        assert nt_mod == pytest.approx(hf_mod, rel=REL_TOL), (name, nt_mod, hf_mod)


def test_conversion_roundtrip_exact(nt, matched_jet_models):
    from astropt3 import AstroPT3Model

    hf_model, nt_model, nt_config, hf_config = matched_jet_models
    hf_back = AstroPT3Model(hf_config).cuda().to(torch.bfloat16)
    nt.to_hf.convert_nt_to_hf(nt_model, hf_back, nt_config)
    original = hf_model.state_dict()
    back = hf_back.state_dict()
    assert set(back) == set(original)
    for key, value in back.items():
        assert torch.equal(value, original[key]), f"roundtrip drift in {key}"
    # the flow weights actually took part (not just body params)
    assert any(key.startswith("flows.") for key in back), "no flow params converted"
    assert any("decoders." in key and ".proj." in key for key in back), "no GMM heads converted"


def test_tp2_replicated_module_gradients_with_noise(nt):
    if torch.cuda.device_count() < 2:
        pytest.skip("TP=2 check needs 2 visible GPUs")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            str(ASTRO_DIR / "scripts" / "tp2_grad_check.py"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "TP2 GRAD CHECK PASS (" in result.stdout


def test_50step_synthetic_run_and_checkpoint_conversion(nt, tmp_path_factory):
    """torchrun 50 jetformer steps on synthetic data; convert; reproduce loss."""
    workdir = tmp_path_factory.mktemp("jetformer_smoke")
    config = yaml.safe_load(JET_YAML.read_text())
    config["checkpoints"]["checkpoints_path"] = str(workdir / "checkpoints")
    stdout = run_train(config, workdir, "jet-tiny")

    losses = losses_from(stdout)
    assert len(losses) >= 40, f"expected ~50 logged losses, got {len(losses)}"
    # likelihood loss can cross zero: assert an absolute drop, not a ratio
    assert losses[-1] < losses[0] - 10.0, f"jetformer smoke did not learn: {losses[0]:.4f} -> {losses[-1]:.4f}"

    checkpoint = workdir / "checkpoints" / "50"
    assert (checkpoint / "model_config.json").exists()

    save_path = workdir / "hf"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            str(TOOLS_DIR / "convert_nanotron_to_hf.py"),
            f"--checkpoint_path={checkpoint}",
            f"--save_path={save_path}",
        ],
        cwd=REPO_ROOT,
        check=False,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    import json

    from transformers import AutoModel

    import astropt3  # noqa: F401

    hf_model = AutoModel.from_pretrained(save_path).cuda().to(torch.bfloat16).eval()

    with open(checkpoint / "model_config.json") as f:
        nt_config = nt.config_cls(**json.load(f))
    nt_model = build_nt_model(nt, nt_config)
    nt.serialize.load_weights(
        model=nt_model, parallel_context=nt.parallel_context, root_folder=checkpoint
    )
    nt_model.eval()
    flat = micro_batch(hf_model.config, skip=7)  # not the first training batch
    names = hf_model.config.modality_registry().names()
    with torch.no_grad():
        nt_loss = nt_model(**flat)["loss"].item()
        hf_loss = hf_model(**regroup(flat, names)).loss.item()
    assert hf_loss == pytest.approx(nt_loss, rel=REL_TOL), (nt_loss, hf_loss)
