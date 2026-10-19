"""Bounded real-training throughput run; launch with the normal torchrun arguments.

No training loop lives here. YAML owns the workload and schedule; nanotron
owns setup and training. Results must live on storage shared by every rank.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def prepare_config(baseline: dict, output: Path, world_size: int,
                   warmup_steps: int, steps: int) -> tuple[dict, dict]:
    """Validate without CUDA/nanotron imports; never change a workload knob."""
    if warmup_steps < 0 or steps <= 0:
        raise ValueError("warmup-steps must be >= 0 and steps must be > 0")
    if os.environ.get("NANOTRON_BENCHMARK", "0") != "0":
        raise ValueError("Unset NANOTRON_BENCHMARK (its legacy mode exits at step four)")
    parallel = baseline["parallelism"]
    if any(parallel.get(key, 1) != 1 for key in
           ("pp", "context_parallel_size", "expert_parallel_size")):
        raise ValueError("AstroPT benchmarking requires PP=CP=EP=1")
    if any(type(parallel.get(key, 1)) is not int or parallel.get(key, 1) < 1 for key in ("dp", "tp")):
        raise ValueError("YAML dp and tp must be positive integers")
    if parallel.get("dp", 1) * parallel.get("tp", 1) != world_size:
        raise ValueError("launch WORLD_SIZE must equal YAML dp * tp")
    if not baseline["model"]["model_config"].get("is_astropt3_config"):
        raise ValueError("benchmark requires an AstroPT3 model")
    stages = baseline.get("data_stages", [])
    if len(stages) != 1 or not stages[0]["data"]["dataset"].get("is_astropt3_streaming"):
        raise ValueError("benchmark requires one AstroPT3 streaming data stage")
    if warmup_steps + steps > baseline["tokens"]["train_steps"]:
        raise ValueError("warmup + measured steps exceed the unchanged YAML horizon")
    # Remote resume currently depends on S3 upload settings in nanotron.
    checkpoints = baseline["checkpoints"]
    resume = checkpoints.get("resume_checkpoint_path")
    init_path = baseline["model"]["init_method"].get("path")
    for path in (resume, init_path):
        if path and "://" in str(path):
            raise ValueError("stage remote initialization/resume locally before benchmarking")
    if resume:
        path = Path(resume)
        if not ((path / "latest.txt").is_file() or (path / "model_config.json").is_file()):
            raise ValueError(f"explicit resume checkpoint not found: {path}")

    effective = copy.deepcopy(baseline)
    overrides = {
        "checkpoints.checkpoints_path": str(output / "disabled-checkpoints"),
        "checkpoints.checkpoint_interval": 0,
        "checkpoints.checkpoint_schedule": None,
        "checkpoints.save_initial_state": False,
        "checkpoints.save_final_state": False,
        "profiler": None,
        "lighteval": None,
        "s3_upload": None,
        "tokens.limit_val_batches": 0,
        "tokens.limit_test_batches": 0,
        "general.benchmark_csv_path": None,
    }
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        section = effective
        for key in keys[:-1]:
            section = section.setdefault(key, {})
        section[keys[-1]] = value
    overrides["trainer.benchmark"] = True  # disables external tracker calls
    overrides["trainer.max_steps"] = warmup_steps + steps
    overrides["env.ASTROPT3_TELEMETRY_DIR"] = str(output)
    return effective, overrides


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def create_output(output: Path, baseline_text: str, effective: dict, manifest: dict) -> None:
    """Rank zero only. Never attach to an earlier run, even an empty one."""
    output.mkdir(parents=True, exist_ok=False)
    (output / "baseline.yaml").write_text(baseline_text)
    (output / "effective.yaml").write_text(yaml.safe_dump(effective, sort_keys=False))
    write_json(output / "run.json", manifest)
    write_json(output / "result.json", {
        "schema_version": 1, "status": "partial", "diagnostics": ["run has not completed"],
    })


def validate_output(output: Path, manifest: dict) -> None:
    """Check rank zero's fresh nonce before touching any rank-local files."""
    try:
        stored = json.loads((output / "run.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("cannot read rank zero's manifest from shared output storage") from error
    if stored != manifest:
        raise ValueError("ranks disagree on config/run ID or output is not shared storage")


@contextmanager
def local_log(path: Path):
    """Capture native libraries and worker output too, not just Python prints."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        with path.open("w") as handle:
            os.dup2(handle.fileno(), 1)
            os.dup2(handle.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, original in zip((1, 2), saved, strict=True):
            os.dup2(original, fd)
            os.close(original)


def revision(path: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    try:
        status = git("status", "--porcelain")
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def capture_context(torch, trainer, loader) -> dict:
    import psutil

    from astropt3.data import nanotron_loader as data

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    packages = {}
    for name in ("torch", "flash-attn", "nanotron", "astropt3", "transformers", "lsdb",
                 "hats", "dask", "distributed", "huggingface-hub", "fsspec", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    try:
        lsdb_source = json.loads(distribution("lsdb").read_text("direct_url.json") or "null")
    except (PackageNotFoundError, json.JSONDecodeError):
        lsdb_source = None
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    dataset = loader.dataset
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = "unknown"
    return {
        "host": socket.gethostname(), "local_rank": int(os.environ["LOCAL_RANK"]),
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
        "gpu": {"name": device.name, "vram_bytes": device.total_memory,
                "uuid": str(getattr(device, "uuid", "unknown"))},
        "cpu": {"model": cpu_model, "logical_count": os.cpu_count(),
                "affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
        "ram_bytes": psutil.virtual_memory().total,
        "software": {"python": platform.python_version(), "platform": platform.platform(),
                     "cuda_runtime": torch.version.cuda, "driver": driver,
                     "packages": packages, "lsdb_source": lsdb_source},
        "code": {"astropt3": revision(REPO), "nanotron": revision(REPO / "nanotron"),
                 "uv_lock_sha256": hashlib.sha256((REPO / "astro/uv.lock").read_bytes()).hexdigest()},
        "num_loading_workers": loader.num_workers,
        "loader": {
            "legacy_catalog": data.LEGACY_CATALOG, "legacy_columns": dataset.columns,
            "desi_catalog": data.DESI_CATALOG if dataset.desi_columns else None,
            "desi_columns": dataset.desi_columns, "modalities": dataset.config.modalities,
            "crossmatch_radius_arcsec": data._CROSSMATCH_RADIUS_ARCSEC,
            "count_fraction_threshold": data._CROSSMATCH_COUNT_FRACTION_THRESHOLD,
            "partitions_per_chunk": data._PARTITIONS_PER_CHUNK,
            "dask_processes": False, "dask_workers_per_reader": 1,
            "dask_threads_per_reader": data._PARTITIONS_PER_CHUNK,
            "prefetch_factor": loader.prefetch_factor, "pin_memory": loader.pin_memory,
            "max_net_retries": data._MAX_NET_RETRIES, "max_net_retry_wait_s": data._MAX_NET_RETRY_WAIT,
        },
        "initialization": {
            "configured": trainer.config.model.init_method.__class__.__name__,
            "resolved_resume_checkpoint": str(trainer.init_checkpoint_path) if trainer.init_checkpoint_path else None,
            "load_optimizer": trainer.config.checkpoints.load_optimizer,
            "load_lr_scheduler": trainer.config.checkpoints.load_lr_scheduler,
            "stream_position": "fresh cursorless stream (including on resume)",
        },
    }


def run_training(output: Path, warmup_steps: int, steps: int, state: dict, started: float) -> None:
    """Reuse nanotron setup and train; exceptions remain failures for torchrun."""
    import torch
    import torch.distributed as dist

    sys.path.insert(0, str(REPO / "nanotron"))
    from run_train import (  # pyright: ignore[reportMissingImports] -- checkout path above
        DistributedTrainer,
        get_dataloader,
    )

    loader = trainer = None
    try:
        trainer = DistributedTrainer(str(output / "effective.yaml"), benchmark=True)
        if trainer.config.checkpoints.resume_checkpoint_path and trainer.init_checkpoint_path is None:
            raise ValueError("explicit resume could not be resolved; refusing random initialization")
        state.update({
            "start_step": trainer.metadata.last_train_step,
            "dp_rank": dist.get_rank(trainer.parallel_context.dp_pg),
            "tp_rank": dist.get_rank(trainer.parallel_context.tp_pg),
        })
        if state["start_step"] + warmup_steps + steps > trainer.config.tokens.train_steps:
            raise ValueError("not enough remaining YAML horizon after resume")
        dataloaders = get_dataloader(trainer)
        loader = next(iter(dataloaders.values()))
        state.update(capture_context(torch, trainer, loader))
        if state["rank"] == 0:
            trainer.config.save_as_yaml(output / "resolved.yaml")
        write_json(output / f"rank{state['rank']}.json", state)
        state["first_batch_s"] = loader.prime()
        state["first_batch_ready_s"] = time.perf_counter() - started
        trainer.train(dataloaders, max_steps=warmup_steps + steps)
        state["startup_s"] = trainer.telemetry_started_at - started
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if loader is not None:
                loader.close()
        except BaseException as error:
            state.update(status="failed", error=f"loader shutdown: {type(error).__name__}: {error}")
            raise
        finally:
            state["whole_run_s"] = time.perf_counter() - started
            write_json(output / f"rank{state['rank']}.json", state)


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory on shared output storage")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--bandwidth-cap", default="unknown", help="external cap description; does not apply a cap")
    parser.add_argument("--cache-state", default="unknown", help="known cache state; never flushes caches")
    args = parser.parse_args(argv)
    if not all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        parser.error("launch with torchrun, including for one GPU")
    output = args.output.resolve()
    baseline_text = args.config_file.read_text()
    baseline = yaml.safe_load(baseline_text)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    effective, overrides = prepare_config(baseline, output, world, args.warmup_steps, args.steps)
    manifest = {
        "schema_version": 1, "world_size": world,
        "parallelism": {key: baseline["parallelism"].get(key, 1) for key in
                        ("dp", "tp", "pp", "context_parallel_size", "expert_parallel_size")},
        "warmup_steps": args.warmup_steps, "steps": args.steps,
        "baseline_config": baseline, "overrides": overrides,
        "config_sha256": hashlib.sha256(baseline_text.encode()).hexdigest(),
        "bandwidth_cap": args.bandwidth_cap, "cache_state": args.cache_state,
        "timing_policy": "CUDA synchronize + world barrier at every step boundary; continuous rank-local wall clock",
    }
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    publication: list[str | None] = [None, None]
    if rank == 0:
        publication[1] = uuid.uuid4().hex
        manifest["run_id"] = publication[1]
        try:
            create_output(output, baseline_text, effective, manifest)
        except Exception as error:
            publication[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(publication, src=0)
    if publication[0]:
        dist.destroy_process_group()
        raise RuntimeError(publication[0])
    manifest["run_id"] = publication[1]
    try:
        validate_output(output, manifest)
    except BaseException:
        dist.destroy_process_group()
        raise
    os.environ["ASTROPT3_TELEMETRY_DIR"] = str(output)
    state = {"rank": rank, "status": "running"}
    write_json(output / f"rank{rank}.json", state)
    print(f"[benchmark rank {rank}] local logs: {output / f'trainer.rank{rank}.log'}", flush=True)
    try:
        with local_log(output / f"trainer.rank{rank}.log"):
            try:
                run_training(output, args.warmup_steps, args.steps, state, started)
            except BaseException as error:
                state.update(status="failed", error=f"{type(error).__name__}: {error}")
                write_json(output / f"rank{rank}.json", state)
                traceback.print_exc()
                raise
        dist.barrier()
        if rank == 0:
            from bench_report import build_report, render

            report = build_report(output, None, "throughput", 60.0)
            write_json(output / "result.json", report)
            print(render(report))
            if report["status"] != "complete":
                raise RuntimeError("benchmark inputs incomplete; see result.json")
    except BaseException:
        if rank == 0:
            from bench_report import build_report

            write_json(output / "result.json", build_report(output, None, "throughput", 60.0))
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
