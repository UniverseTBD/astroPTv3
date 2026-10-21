"""Bounded real-training throughput run; launch with the normal torchrun arguments.

No training loop lives here. YAML owns the workload and schedule; nanotron
owns setup and training. Results must live on storage shared by every rank.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def prepare_config(baseline: dict, output: Path, world_size: int,
                   warmup_steps: int, steps: int) -> tuple[dict, dict]:
    """Validate without CUDA/nanotron imports; never change a workload knob."""
    if warmup_steps < 1 or steps < 1:
        # Warmup absorbs the first batch (stream open + first fetch).
        raise ValueError("warmup-steps and steps must both be >= 1")
    if os.environ.get("NANOTRON_BENCHMARK", "0") != "0":
        raise ValueError("Unset NANOTRON_BENCHMARK (its legacy mode exits at step four)")
    parallel = baseline["parallelism"]
    if any(parallel.get(key, 1) != 1 for key in ("pp", "context_parallel_size", "expert_parallel_size")):
        raise ValueError("AstroPT benchmarking requires PP=CP=EP=1")
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
    for path in (baseline["checkpoints"].get("resume_checkpoint_path"),
                 baseline["model"]["init_method"].get("path")):
        if path and "://" in str(path):
            raise ValueError("stage remote initialization/resume locally before benchmarking")

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
    """Check rank zero's fresh run ID before touching any rank-local files."""
    try:
        stored = json.loads((output / "run.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError("cannot read rank zero's manifest from shared output storage") from error
    if stored != manifest:
        raise ValueError("ranks disagree on config/run ID or output is not shared storage")


def revision(path: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


# Dense BF16 TFLOP/s (half NVIDIA's sparse datasheet figures) and the smallest
# memory a full device reports: MIG/vGPU slices can keep the full-device name.
DENSE_BF16_PEAKS = {
    "H200 NVL": (835.5, 141e9), "H200 SXM": (989.5, 141e9), "H100 NVL": (835.5, 94e9),
    "H100 80GB HBM3": (989.5, 80e9), "A100-SXM4-80GB": (312.0, 80e9), "A100 80GB PCIe": (312.0, 80e9),
}


def gpu_peak(name: str, vram_bytes: int) -> float | None:
    """Exact variants only, never a family-name guess."""
    peak, min_vram = DENSE_BF16_PEAKS.get(name.removeprefix("NVIDIA "), (None, 0.0))
    return peak if vram_bytes >= min_vram else None


def nic_link_speeds(sys_class_net: Path = Path("/sys/class/net")) -> dict:
    """Negotiated Mbit/s of each up physical NIC (None if unknown); not route capacity."""
    import psutil

    return {name: nic.speed if nic.speed > 0 else None
            for name, nic in sorted(psutil.net_if_stats().items())
            if nic.isup and (sys_class_net / name / "device").exists()}  # skips virtual NICs


def capture_flop_estimate(trainer) -> dict:
    """Persist the model's existing arithmetic so offline reports need no CUDA."""
    from nanotron.models.astropt3 import modality_flops_per_token

    world = trainer.parallel_context.world_pg.size()
    sequence = trainer.sequence_length
    tflops, _ = trainer.unwrapped_model.get_flops_per_sec(1.0, sequence, 1)
    return {
        "dtype": str(trainer.config.model.dtype),
        "backbone_flops_per_token": tflops * world * 1e12 / sequence,
        "modality_forward_flops_per_token": modality_flops_per_token(trainer.model_config),
    }


def capture_context(torch, trainer, loader) -> dict:
    import psutil

    from astropt3.data import nanotron_loader as data

    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    cpuinfo = Path("/proc/cpuinfo")
    cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                      if line.startswith("model name")), None) if cpuinfo.exists() else None
    packages = {}
    for name in ("torch", "flash-attn", "nanotron", "lsdb", "hats", "huggingface-hub"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    dataset = loader.dataset
    return {
        "host": socket.gethostname(),
        "gpu": {"name": device.name, "vram_bytes": device.total_memory,
                "peak_tflops_per_gpu": gpu_peak(device.name, device.total_memory)},
        "flops_estimate": capture_flop_estimate(trainer),
        "nic_link_mbps": nic_link_speeds(),
        "cpu": {"model": cpu_model or platform.processor(), "count": os.cpu_count()},
        "ram_bytes": psutil.virtual_memory().total,
        "software": {"python": platform.python_version(), "cuda": torch.version.cuda, **packages},
        "code": {"astropt3": revision(REPO), "nanotron": revision(REPO / "nanotron")},
        "num_loading_workers": loader.num_workers,
        "loader": {
            "legacy_catalog": data.LEGACY_CATALOG, "legacy_columns": dataset.columns,
            "desi_catalog": data.DESI_CATALOG if dataset.desi_columns else None,
            "desi_columns": dataset.desi_columns, "modalities": dataset.config.modalities,
            "crossmatch_radius_arcsec": data._CROSSMATCH_RADIUS_ARCSEC,
            "count_fraction_threshold": data._CROSSMATCH_COUNT_FRACTION_THRESHOLD,
            "partitions_per_chunk": data._PARTITIONS_PER_CHUNK,
        },
        "init_method": trainer.config.model.init_method.__class__.__name__,
        "resumed_from": str(trainer.init_checkpoint_path) if trainer.init_checkpoint_path else None,
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

    try:
        trainer = DistributedTrainer(str(output / "effective.yaml"), benchmark=True)
        if trainer.config.checkpoints.resume_checkpoint_path and trainer.init_checkpoint_path is None:
            raise ValueError("explicit resume could not be resolved; refusing random initialization")
        dataloaders = get_dataloader(trainer)
        state.update(
            start_step=trainer.metadata.last_train_step,
            dp_rank=dist.get_rank(trainer.parallel_context.dp_pg),
            tp_rank=dist.get_rank(trainer.parallel_context.tp_pg),
            **capture_context(torch, trainer, next(iter(dataloaders.values()))),
        )
        if state["rank"] == 0:
            trainer.config.save_as_yaml(output / "resolved.yaml")
        trainer.train(dataloaders, max_steps=warmup_steps + steps)
        state.update(status="complete", startup_s=trainer.telemetry_started_at - started,
                     whole_run_s=time.perf_counter() - started, ended_at=time.time())
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
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
        "bandwidth_cap": args.bandwidth_cap, "cache_state": args.cache_state,
    }
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl")
    # Rank zero creates the directory; every rank then checks it sees that same fresh run.
    publication: list[str | None] = [None, uuid.uuid4().hex]
    if rank == 0:
        manifest["run_id"] = publication[1]
        try:
            create_output(output, baseline_text, effective, manifest)
        except Exception as error:
            publication[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(publication, src=0)
    manifest["run_id"] = publication[1]
    report = None
    try:
        if publication[0]:
            raise RuntimeError(publication[0])
        validate_output(output, manifest)
        os.environ["ASTROPT3_TELEMETRY_DIR"] = str(output)
        run_training(output, args.warmup_steps, args.steps, {"rank": rank}, started)
        dist.barrier()
    finally:
        if rank == 0 and not publication[0]:
            from bench_report import build_report, render

            report = build_report(output, None, "throughput", 60.0)
            write_json(output / "result.json", report)
            print(render(report))
        dist.destroy_process_group()
    if report is not None and report["status"] != "complete":
        raise SystemExit("benchmark inputs incomplete; see result.json")


if __name__ == "__main__":
    main()
