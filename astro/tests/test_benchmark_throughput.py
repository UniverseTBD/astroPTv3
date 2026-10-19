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
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "model_config.json").write_text("{}")
    baseline["checkpoints"].update(resume_checkpoint_path=str(resume), load_optimizer=False)
    baseline["profiler"] = {"active": 2}
    baseline["s3_upload"] = {"upload_s3_path": "s3://not-used"}
    baseline["lighteval"] = {"eval_interval": 1}
    before = copy.deepcopy(baseline)
    effective, overrides = bench.prepare_config(baseline, tmp_path / "result", 1, 2, 3)
    assert baseline == before
    for key in ("model", "optimizer", "data_stages", "parallelism", "logging"):
        assert effective[key] == baseline[key]
    assert effective["tokens"]["train_steps"] == 50
    assert effective["checkpoints"]["resume_checkpoint_path"] == str(resume)
    assert effective["checkpoints"]["load_optimizer"] is False
    assert effective["checkpoints"]["checkpoint_interval"] == 0
    assert effective["checkpoints"]["checkpoint_schedule"] is None
    assert not effective["checkpoints"]["save_final_state"]
    assert not effective["checkpoints"]["save_initial_state"]
    assert all(effective[key] is None for key in ("profiler", "lighteval", "s3_upload"))
    assert overrides["trainer.max_steps"] == 5
    assert overrides["trainer.benchmark"] is True


@pytest.mark.parametrize("world,warmup,steps,match", [
    (2, 1, 2, "WORLD_SIZE"), (1, -1, 2, "warmup-steps"),
    (1, 1, 0, "steps"), (1, 1, 50, "horizon"),
])
def test_invalid_launch_is_rejected(baseline, tmp_path, world, warmup, steps, match):
    with pytest.raises(ValueError, match=match):
        bench.prepare_config(baseline, tmp_path, world, warmup, steps)


def test_missing_and_remote_resume_do_not_fall_back_to_random_init(baseline, tmp_path):
    for path in (str(tmp_path / "missing"), "s3://bucket/checkpoint"):
        baseline["checkpoints"]["resume_checkpoint_path"] = path
        with pytest.raises(ValueError, match="resume"):
            bench.prepare_config(baseline, tmp_path, 1, 0, 1)


@pytest.mark.parametrize("field,value", [("dp", 0), ("tp", True), ("pp", 2)])
def test_invalid_topology_is_rejected(baseline, tmp_path, field, value):
    baseline["parallelism"][field] = value
    with pytest.raises(ValueError):
        bench.prepare_config(baseline, tmp_path, 1, 0, 1)


def test_legacy_early_exit_mode_is_rejected(baseline, tmp_path, monkeypatch):
    monkeypatch.setenv("NANOTRON_BENCHMARK", "1")
    with pytest.raises(ValueError, match="NANOTRON_BENCHMARK"):
        bench.prepare_config(baseline, tmp_path, 1, 0, 1)


def test_output_must_be_new_and_retains_exact_baseline(baseline, tmp_path):
    output = tmp_path / "result"
    text = "# Preserve comments too\n" + yaml.safe_dump(baseline)
    effective, _ = bench.prepare_config(baseline, output, 1, 0, 1)
    bench.create_output(output, text, effective, {"schema_version": 1})
    assert (output / "baseline.yaml").read_text() == text
    assert yaml.safe_load((output / "effective.yaml").read_text()) == effective
    assert json.loads((output / "result.json").read_text())["status"] == "partial"
    with pytest.raises(FileExistsError):
        bench.create_output(output, "replaced", effective, {})
    assert (output / "baseline.yaml").read_text() == text


def test_stale_node_local_output_is_rejected_before_any_writes(tmp_path):
    old_manifest = {"schema_version": 1, "run_id": "previous-invocation"}
    (tmp_path / "run.json").write_text(json.dumps(old_manifest))
    (tmp_path / "rank1.json").write_text("old rank metadata")
    (tmp_path / "trainer.rank1.log").write_text("old logs")
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="run ID"):
        bench.validate_output(tmp_path, {**old_manifest, "run_id": "new-invocation"})
    assert all(path.read_bytes() == value for path, value in before.items())
    bench.validate_output(tmp_path, old_manifest)


@pytest.mark.parametrize("fail", [None, "train", "shutdown"])
def test_runner_reuses_train_and_propagates_failure(tmp_path, monkeypatch, fail):
    import torch.distributed as dist

    calls = []

    class Trainer:
        def __init__(self, path, *, benchmark):
            calls.append(("setup", path, benchmark))
            self.config = SimpleNamespace(tokens=SimpleNamespace(train_steps=100),
                                          checkpoints=SimpleNamespace(resume_checkpoint_path=None),
                                          save_as_yaml=lambda path: calls.append("resolved"))
            self.init_checkpoint_path = None
            self.metadata = SimpleNamespace(last_train_step=20)
            self.parallel_context = SimpleNamespace(dp_pg=None, tp_pg=None)
            self.telemetry_started_at = 10.0

        def train(self, dataloaders, *, max_steps):
            calls.append(("train", max_steps, self.config.tokens.train_steps))
            if fail == "train":
                raise RuntimeError("training failure")

    def close():
        calls.append("closed")
        if fail == "shutdown":
            raise RuntimeError("shutdown failure")

    loader = SimpleNamespace(prime=lambda: 3.0, close=close)
    monkeypatch.setitem(sys.modules, "run_train", SimpleNamespace(
        DistributedTrainer=Trainer, get_dataloader=lambda trainer: {"stage": loader},
    ))
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(bench, "capture_context", lambda *args: {})
    state = {"rank": 0, "status": "running"}
    if fail:
        with pytest.raises(RuntimeError, match="failure"):
            bench.run_training(tmp_path, 2, 3, state, 0.0)
    else:
        bench.run_training(tmp_path, 2, 3, state, 0.0)
    assert ("train", 5, 100) in calls
    assert calls[-1] == "closed"
    saved = json.loads((tmp_path / "rank0.json").read_text())
    assert saved["start_step"] == 20
    assert saved["first_batch_s"] == 3.0
    assert saved["status"] == ("failed" if fail else "complete")
