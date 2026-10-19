"""Execute the actual nanotron loop with CPU doubles (no flash-attn import)."""

import ast
import datetime
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from astropt3.data import telemetry


@pytest.fixture
def trainer(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[2] / "nanotron/src/nanotron/trainer.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DistributedTrainer")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {"train", "_should_checkpoint", "pre_training"}]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *methods], type_ignores=[])
    ast.fix_missing_locations(module)
    ticks = iter([100.0, 101.0, 103.0, 110.0])
    calls = {"logs": [], "noise": [], "timer_ends": [], "trackers": []}
    timer = SimpleNamespace(start=lambda: None, end=lambda: calls["timer_ends"].append(True))
    groups = SimpleNamespace(world_pg=SimpleNamespace(size=lambda: 4), dp_pg="dp", tp_pg="tp")
    namespace = {
        "os": __import__("os"), "time": SimpleNamespace(perf_counter=lambda: next(ticks)),
        "gc": SimpleNamespace(collect=lambda: None),
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None),
                                 profiler=SimpleNamespace(profile=type("UnusedProfiler", (), {}))),
        "dist": SimpleNamespace(barrier=lambda *args: None,
                                get_rank=lambda group: {"dp": 1, "tp": 0}.get(group, 2)
                                if isinstance(group, str) else 2),
        "DistributedDataParallel": type("UnusedDDP", (), {}),
        "nanotron_timer": lambda *args, **kwargs: timer,
        "get_profiler": lambda **kwargs: nullcontext(),
        "log_rank": lambda *args, **kwargs: None, "logger": logging.getLogger(__name__),
        "logging": logging, "datetime": datetime,
        "wandb": SimpleNamespace(init=lambda **kwargs: calls["trackers"].append(True)),
    }
    exec(compile(module, str(source), "exec"), namespace)  # noqa: S102 -- execute repository code with CPU doubles
    Trainer = type("Trainer", (), {name: namespace[name] for name in ("train", "_should_checkpoint", "pre_training")})
    instance: Any = Trainer()
    instance.benchmark = True
    instance.config = SimpleNamespace(
        tokens=SimpleNamespace(train_steps=100),
        checkpoints=SimpleNamespace(save_initial_state=False, save_final_state=False, checkpoint_interval=0),
        parallelism=SimpleNamespace(pp_engine=SimpleNamespace()),
        logging=SimpleNamespace(iteration_step_info_interval=3),
        general=SimpleNamespace(ignore_sanity_checks=True),
    )
    stage = SimpleNamespace(consumed_train_samples=0, sequence_length=4)
    instance.metadata = SimpleNamespace(last_train_step=7, consumed_train_samples=0,
                                        consumed_tokens_total=0, current_stage=stage)
    instance.model = SimpleNamespace(named_modules=list, set_jet_noise_frac=calls["noise"].append,
                                     get_mfu_report=lambda **kwargs: {})  # unknown GPU peak
    instance.parallel_context = groups
    instance.n_micro_batches_per_batch = instance.micro_batch_size = 1
    instance.global_batch_size = 2
    instance.sequence_length = 4
    instance.current_base_dl = instance.current_dataloader = None
    instance.init_checkpoint_path = None
    instance._update_dataloader_based_on_training_stages = lambda loaders: None
    instance.train_step_logs = lambda **kwargs: calls["logs"].append(instance.iteration_step)
    instance.save_checkpoint = lambda: pytest.fail("benchmark wrote a checkpoint")
    instance.post_training = lambda: None
    instance.pre_training = lambda **kwargs: None

    def step(**kwargs):
        telemetry.observe_micro_batch({"input_ids": torch.tensor([[1, 3, 0, 0]])}, wait_s=0.5)
        return None, None, None, None

    instance.training_step = step
    monkeypatch.setenv(telemetry.TELEMETRY_DIR_ENV, str(tmp_path))
    telemetry.drain_step()
    yield instance, calls, tmp_path, Trainer
    telemetry.drain_step()


def test_loop_bounds_resume_without_changing_schedules_and_logs_every_step(trainer):
    import json

    instance, calls, directory, _ = trainer
    # Exercise unknown-GPU MFU too, outside benchmark suppression.
    instance.benchmark = False
    instance.train({}, max_steps=3)
    assert instance.config.tokens.train_steps == 100
    assert instance.metadata.last_train_step == 10
    assert calls["noise"] == [0.08, 0.09, 0.10]
    assert calls["logs"] == [10]  # sparse console logs, not sparse counters
    assert len(calls["timer_ends"]) == 3
    records = [json.loads(line) for line in (directory / "steps.rank2.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [8, 9, 10]
    assert [record["step_seconds"] for record in records] == [1.0, 2.0, 7.0]
    assert [record["elapsed_s"] for record in records] == [1.0, 3.0, 10.0]
    assert all(record["micro_batches"] == 1 and record["tokens_nonpad"] == 2 for record in records)
    assert all(record["world_size"] == 4 and record["dp_rank"] == 1 for record in records)


@pytest.mark.parametrize("limit", [0, -1, 94, True, 1.5])
def test_invalid_stop_limit_fails_before_any_side_effect(trainer, limit):
    instance, calls, directory, _ = trainer
    instance.config.checkpoints.save_initial_state = True
    with pytest.raises(ValueError, match="max_steps"):
        instance.train({}, max_steps=limit)
    assert not calls["noise"]
    assert not list(directory.glob("steps.*"))


def test_benchmark_disables_tracker_initialization(trainer):
    instance, calls, _, cls = trainer
    instance.parallel_context.cp_pg = SimpleNamespace(size=lambda: 1)
    instance.parallel_context.dp_pg = SimpleNamespace(size=lambda: 1)
    cls.pre_training(instance)
    assert not calls["trackers"]
