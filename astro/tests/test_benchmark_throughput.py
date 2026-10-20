"""Runner/config checks without importing nanotron's GPU dependencies."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_throughput as bench  # pyright: ignore[reportMissingImports]


@pytest.fixture
def baseline(monkeypatch):
    monkeypatch.delenv("NANOTRON_BENCHMARK", raising=False)
    return yaml.safe_load((bench.REPO / "astro/configs/nanotron/astropt3-test-tiny.yaml").read_text())


def test_overrides_preserve_workload_schedules_and_explicit_initialization(baseline, tmp_path):
    baseline["checkpoints"].update(resume_checkpoint_path=str(tmp_path), load_optimizer=False)
    baseline["profiler"] = {"active": 2}
    baseline["s3_upload"] = {"upload_s3_path": "s3://not-used"}
    baseline["lighteval"] = {"eval_interval": 1}
    before = copy.deepcopy(baseline)
    effective, overrides = bench.prepare_config(baseline, tmp_path / "result", 1, 2, 3)
    assert baseline == before
    for key in ("model", "optimizer", "data_stages", "parallelism", "logging"):
        assert effective[key] == baseline[key]
    assert effective["tokens"]["train_steps"] == 50
    assert effective["checkpoints"]["resume_checkpoint_path"] == str(tmp_path)
    assert effective["checkpoints"]["load_optimizer"] is False
    assert effective["checkpoints"]["checkpoint_interval"] == 0
    assert effective["checkpoints"]["checkpoint_schedule"] is None
    assert not effective["checkpoints"]["save_final_state"]
    assert not effective["checkpoints"]["save_initial_state"]
    assert all(effective[key] is None for key in ("profiler", "lighteval", "s3_upload"))
    assert overrides["trainer.max_steps"] == 5
    assert overrides["trainer.benchmark"] is True


@pytest.mark.parametrize("world,warmup,steps,match", [
    (2, 1, 2, "WORLD_SIZE"), (1, 0, 2, "warmup-steps"),
    (1, 1, 0, "steps"), (1, 1, 50, "horizon"),
])
def test_invalid_launch_is_rejected(baseline, tmp_path, world, warmup, steps, match):
    with pytest.raises(ValueError, match=match):
        bench.prepare_config(baseline, tmp_path, world, warmup, steps)


def test_remote_resume_must_be_staged_locally(baseline, tmp_path):
    baseline["checkpoints"]["resume_checkpoint_path"] = "s3://bucket/checkpoint"
    with pytest.raises(ValueError, match="resume"):
        bench.prepare_config(baseline, tmp_path, 1, 1, 1)


def test_unsupported_parallelism_is_rejected(baseline, tmp_path):
    baseline["parallelism"]["pp"] = 2
    with pytest.raises(ValueError, match="PP=CP=EP=1"):
        bench.prepare_config(baseline, tmp_path, 2, 1, 1)


def test_legacy_early_exit_mode_is_rejected(baseline, tmp_path, monkeypatch):
    monkeypatch.setenv("NANOTRON_BENCHMARK", "1")
    with pytest.raises(ValueError, match="NANOTRON_BENCHMARK"):
        bench.prepare_config(baseline, tmp_path, 1, 1, 1)


def test_output_must_be_new_and_retains_exact_baseline(baseline, tmp_path):
    output = tmp_path / "result"
    text = "# Preserve comments too\n" + yaml.safe_dump(baseline)
    effective, _ = bench.prepare_config(baseline, output, 1, 1, 1)
    bench.create_output(output, text, effective, {"schema_version": 1})
    assert (output / "baseline.yaml").read_text() == text
    assert yaml.safe_load((output / "effective.yaml").read_text()) == effective
    assert json.loads((output / "result.json").read_text())["status"] == "partial"
    with pytest.raises(FileExistsError):
        bench.create_output(output, "replaced", effective, {})
    assert (output / "baseline.yaml").read_text() == text


def test_stale_node_local_output_is_rejected(tmp_path):
    old_manifest = {"schema_version": 1, "run_id": "previous-invocation"}
    (tmp_path / "run.json").write_text(json.dumps(old_manifest))
    with pytest.raises(ValueError, match="run ID"):
        bench.validate_output(tmp_path, {**old_manifest, "run_id": "new-invocation"})
    bench.validate_output(tmp_path, old_manifest)


@pytest.mark.parametrize("fail", [None, "train", "resume"])
def test_runner_reuses_train_and_records_failure(tmp_path, monkeypatch, fail):
    import torch.distributed as dist

    calls = []

    class Trainer:
        def __init__(self, path, *, benchmark):
            calls.append(("setup", benchmark))
            resume = "missing-checkpoint" if fail == "resume" else None
            self.config = SimpleNamespace(tokens=SimpleNamespace(train_steps=100),
                                          checkpoints=SimpleNamespace(resume_checkpoint_path=resume),
                                          save_as_yaml=lambda path: calls.append("resolved"))
            self.init_checkpoint_path = None  # nanotron could not resolve the resume
            self.metadata = SimpleNamespace(last_train_step=20)
            self.parallel_context = SimpleNamespace(dp_pg=None, tp_pg=None)
            self.telemetry_started_at = 10.0

        def train(self, dataloaders, *, max_steps):
            calls.append(("train", max_steps, self.config.tokens.train_steps))
            if fail == "train":
                raise RuntimeError("training failure")

    monkeypatch.setitem(sys.modules, "run_train", SimpleNamespace(
        DistributedTrainer=Trainer, get_dataloader=lambda trainer: {"stage": object()},
    ))
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(bench, "capture_context", lambda *args: {})
    state = {"rank": 0}
    if fail:
        with pytest.raises((RuntimeError, ValueError), match="failure|random initialization"):
            bench.run_training(tmp_path, 2, 3, state, 0.0)
    else:
        bench.run_training(tmp_path, 2, 3, state, 0.0)
    saved = json.loads((tmp_path / "rank0.json").read_text())
    assert saved["status"] == ("failed" if fail else "complete")
    assert (("train", 5, 100) in calls) is (fail != "resume")
    if not fail:
        assert saved["start_step"] == 20
        assert saved["startup_s"] == 10.0


@pytest.mark.parametrize("name,vram,peak", [
    ("NVIDIA H200 NVL", 150_000_000_000, 835.5),
    ("NVIDIA H200 SXM", 150_000_000_000, 989.5),
    ("NVIDIA H100 NVL", 100_000_000_000, 835.5),
    ("NVIDIA H100 80GB HBM3", 85_000_000_000, 989.5),
    ("NVIDIA A100-SXM4-80GB", 85_000_000_000, 312.0),
    ("NVIDIA A100 80GB PCIe", 85_000_000_000, 312.0),
    ("NVIDIA H200", 150_000_000_000, None),  # ambiguous variant
    ("NVIDIA H200 NVL MIG 1g.18gb", 18_000_000_000, None),
    ("NVIDIA H200 NVL", 18_000_000_000, None),  # slice with a full-device name
    ("unknown GPU", 150_000_000_000, None),
])
def test_dense_bf16_peak_uses_exact_gpu_variant(name, vram, peak):
    result = bench.gpu_peak(name, vram)
    assert result["peak_tflops_per_gpu"] == peak
    assert bool(result["peak_source"]) is (peak is not None)


def test_network_link_speed_excludes_virtual_down_and_unknown_nics(tmp_path, monkeypatch):
    stats = {name: SimpleNamespace(isup=up, speed=speed) for name, up, speed in [
        ("eth0", True, 10000), ("eth1", True, 10000), ("slow", True, 1000),
        ("down", False, 100000), ("unknown", True, -1),
        ("veth0", True, 400000), ("lo", True, 0),
    ]}
    for name in ("eth0", "eth1", "slow", "down", "unknown"):
        (tmp_path / name / "device").mkdir(parents=True)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(net_if_stats=lambda: stats))
    result = bench.network_context(tmp_path)
    assert result["max_active_link_speed_mbps"] == 10000
    assert result["fastest_active_interfaces"] == ["eth0", "eth1"]
    assert {nic["name"] for nic in result["interfaces"]} == {"eth0", "eth1", "slow", "down", "unknown"}
    assert next(n for n in result["interfaces"] if n["name"] == "unknown")["link_speed_mbps"] is None


@pytest.mark.parametrize("problem", ["missing_sysfs", "no_links", "os_error"])
def test_network_speed_unavailable_is_not_zero(tmp_path, monkeypatch, problem):
    def stats():
        if problem == "os_error":
            raise OSError("not supported")
        return {}
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(net_if_stats=stats))
    result = bench.network_context(tmp_path / "absent" if problem == "missing_sysfs" else tmp_path)
    assert result["max_active_link_speed_mbps"] is None
    assert result["unavailable_reason"]


def test_flop_capture_undoes_world_size_and_records_model_costs(monkeypatch):
    model_config = object()
    def modality_costs(config):
        assert config is model_config
        return {"images": 123.0}
    def flops(seconds, sequence, batch):
        assert (seconds, sequence, batch) == (1.0, 8, 1)
        return 2.0, 2.0
    monkeypatch.setitem(sys.modules, "nanotron.models.astropt3", SimpleNamespace(
        modality_flops_per_token=modality_costs))
    trainer = SimpleNamespace(
        parallel_context=SimpleNamespace(world_pg=SimpleNamespace(size=lambda: 4)),
        sequence_length=8, model_config=model_config,
        config=SimpleNamespace(model=SimpleNamespace(dtype="torch.bfloat16")),
        unwrapped_model=SimpleNamespace(get_flops_per_sec=flops),
    )
    assert bench.capture_flop_estimate(trainer) == {
        "dtype": "torch.bfloat16", "backbone_flops_per_token": 1e12,
        "modality_forward_flops_per_token": {"images": 123.0},
    }


def test_capture_context_uses_rank_current_device_and_records_new_metrics(monkeypatch):
    import astropt3.data as data_package

    device = SimpleNamespace(name="NVIDIA H200 NVL", total_memory=150_000_000_000)
    requested = []
    def properties(index):
        requested.append(index)
        return device
    torch = SimpleNamespace(cuda=SimpleNamespace(
        current_device=lambda: 1, get_device_properties=properties),
        version=SimpleNamespace(cuda="12.8"))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=1024)))
    monkeypatch.setattr(data_package, "nanotron_loader", SimpleNamespace(
        LEGACY_CATALOG="legacy", DESI_CATALOG="desi", _CROSSMATCH_RADIUS_ARCSEC=1,
        _CROSSMATCH_COUNT_FRACTION_THRESHOLD=0, _PARTITIONS_PER_CHUNK=4), raising=False)
    monkeypatch.setattr(bench, "revision", lambda _: {})
    monkeypatch.setattr(bench, "capture_flop_estimate", lambda _: {"cost": 42})
    monkeypatch.setattr(bench, "network_context", lambda: {"max_active_link_speed_mbps": 10000})
    trainer = SimpleNamespace(init_checkpoint_path=None, config=SimpleNamespace(
        model=SimpleNamespace(init_method=object())))
    loader = SimpleNamespace(num_workers=0, dataset=SimpleNamespace(
        columns=[], desi_columns=[], config=SimpleNamespace(modalities=[])))
    result = bench.capture_context(torch, trainer, loader)
    assert requested == [1]
    assert result["gpu"]["peak_tflops_per_gpu"] == 835.5
    assert result["flops_estimate"] == {"cost": 42}
    assert result["network"]["max_active_link_speed_mbps"] == 10000
