"""Train YAML-configured MMU modalities on one GPU.

--config-file selects a complete Nanotron YAML; the default is the four-modality,
echo-4 recipe. --output must be a new directory. --max-steps bounds a startup
check without changing the configured schedule. --resume-from restores model,
optimizer, scheduler and RNG, not a data cursor; use the same model recipe.
--echo-factor overrides the checkpoint setting on resume, then the YAML default.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

from astropt3.data import nanotron_loader as loader
from astropt3.data import telemetry
from astropt3.data.catalog_stream import CatalogStream
from astropt3.data.echo import validate_echo_factor

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "astro/configs/nanotron/astropt3-70m-four-modal.yaml"


def validate_resume(resume_from: Path, model_config: dict) -> int:
    step = (resume_from / "latest.txt").read_text().strip()
    # pi-lens-ignore: unchecked-throwing-call-python -- main reports invalid resume before creating outputs
    saved = json.loads((resume_from / step / "model_config.json").read_text())
    expected_registry = loader.hf_config_from_modalities(model_config["modalities"]).modality_registry()
    saved_registry = loader.hf_config_from_modalities(saved["modalities"]).modality_registry()
    if expected_registry.modalities != saved_registry.modalities:
        raise ValueError("Checkpoint modalities differ from the selected YAML; use the matching recipe")
    for key, value in model_config.items():
        if key not in {"modalities", "is_astropt3_config"} and key in saved and saved[key] != value:
            raise ValueError(f"Checkpoint model setting {key!r} differs from the selected YAML")
    saved_config = yaml.safe_load((resume_from / step / "config.yaml").read_text())
    factor = saved_config["data_stages"][0]["data"]["dataset"].get("echo_factor", 1)
    validate_echo_factor(factor)
    return factor


def prepare_config(
    baseline: dict, output: Path, resume_from: Path | None = None,
    echo_factor: int | None = None,
) -> dict:
    """Validate the recipe; override only run paths, echoing, and output services."""
    config = copy.deepcopy(baseline)
    if len(config["data_stages"]) != 1:
        raise ValueError("this runner requires exactly one data stage")
    for key in ("dp", "tp", "pp", "context_parallel_size", "expert_parallel_size"):
        if config["parallelism"].get(key, 1) != 1:
            raise ValueError("this runner requires single-GPU parallelism; use nanotron/run_train.py otherwise")
    target = config["tokens"]["train_steps"]
    if type(target) is not int or target < 1:
        raise ValueError("train_steps must be a positive integer")
    dataset = config["data_stages"][0]["data"]["dataset"]
    factor = dataset.get("echo_factor", 1) if echo_factor is None else echo_factor
    validate_echo_factor(factor)
    if factor > 1 and config["tokens"]["batch_accumulation_per_replica"] != 1:
        raise ValueError("echo-factor > 1 requires batch_accumulation_per_replica=1")
    if "crossmatch_desi" in dataset:
        raise ValueError("crossmatch_desi is removed; use an explicit catalog_stream")
    model = config["model"]["model_config"]
    hf_config = loader.hf_config_from_modalities(model["modalities"], vocab_size=model.get("vocab_size", 64))
    dataset["catalog_stream"] = CatalogStream(dataset.get("catalog_stream"), hf_config).spec
    dataset["echo_factor"] = factor
    config["general"]["run"] = output.name
    config["checkpoints"].update(
        checkpoints_path=str(output / "checkpoints"),
        resume_checkpoint_path=str(resume_from) if resume_from else None,
    )
    config.update(profiler=None, lighteval=None, s3_upload=None)
    return config


def wandb_config(config: dict, max_steps: int | None) -> dict:
    """Publish training settings only; catalog locations and provenance stay local."""
    training = {
        key: copy.deepcopy(config[key])
        for key in ("model", "optimizer", "parallelism", "tokens", "data_stages")
    }
    for stage in training["data_stages"]:
        stage["data"]["dataset"].pop("catalog_stream", None)
    if "path" in training["model"].get("init_method", {}):
        training["model"].pop("init_method")
    training["seed"] = config["general"]["seed"]
    training["checkpoints"] = {
        key: config["checkpoints"][key]
        for key in ("checkpoint_schedule", "checkpoint_interval", "save_initial_state", "save_final_state")
    }
    dataset = config["data_stages"][0]["data"]["dataset"]
    stream = dataset["catalog_stream"]
    return {
        "training": training,
        "stream": {
            key: copy.deepcopy(stream[key])
            for key in ("how", "min_surveys", "radius_arcsec", "count_fraction_threshold", "partitions_per_chunk")
        },
        "max_steps": max_steps,
        "echo_factor": dataset["echo_factor"],
        "echo_order": "draw_shuffled_batches_and_modality_spans",
        "throughput_policy": "synchronized end-to-end wall time including loader/checkpoint stalls; first 10 steps warmup",
    }


class Throughput:
    """Aggregate configured modality counts over wall time, never average rates."""

    def __init__(self, modality_names, train_steps, start_step=0):
        self.modality_names = tuple(modality_names)
        self.train_steps = train_steps
        self.warmup_end = start_step + 10
        self.window = deque(maxlen=50)
        self.counts = dict.fromkeys(
            ("tokens_nonpad", "rows", "fresh_tokens_nonpad", "fresh_rows"), 0
        )
        self.seconds = 0.0

    def observe(self, step, record):
        seconds = record["step_seconds"]
        if seconds <= 0 or not math.isfinite(seconds):
            raise ValueError("Step duration must be finite and positive")
        metrics = {
            "train/step": step,
            "throughput/warmup": step <= self.warmup_end,
            "throughput/step_seconds": seconds,
            "throughput/nonpad_tokens_per_s": record["tokens_nonpad"] / seconds,
            "throughput/rows_per_s": record["rows"] / seconds,
            "data/loader_wait_s": record["loader_wait_s"],
            "data/stall_fraction": record["loader_wait_s"] / seconds,
            "data/packing_fraction": record["utilisation_packing"],
            "data/echo_index": record["echo_index"],
            "data/echo_factor": record["echo_factor"],
        }
        for prefix in ("fresh", "replayed"):
            for key in ("micro_batches", "tokens_nonpad", "rows"):
                metrics[f"data/{prefix}_{key}"] = record[f"{prefix}_{key}"]
        total = sum(record["loss_tokens"].values())
        for name in self.modality_names:
            count = record["loss_tokens"].get(name, 0)
            metrics[f"data/{name}_loss_tokens"] = count
            metrics[f"data/{name}_token_fraction"] = count / total if total else 0.0
            for prefix in ("fresh", "replayed"):
                metrics[f"data/{prefix}_{name}_loss_tokens"] = record[f"{prefix}_loss_tokens"].get(name, 0)
        if step > self.warmup_end:
            self.window.append(record)
            for key in self.counts:
                self.counts[key] += record[key]
            self.seconds += seconds
            window_s = sum(item["step_seconds"] for item in self.window)
            metrics.update({
                "throughput/rolling_nonpad_tokens_per_s": sum(item["tokens_nonpad"] for item in self.window) / window_s,
                "throughput/cumulative_nonpad_tokens_per_s": self.counts["tokens_nonpad"] / self.seconds,
                "throughput/cumulative_rows_per_s": self.counts["rows"] / self.seconds,
                "throughput/cumulative_fresh_nonpad_tokens_per_s": self.counts["fresh_tokens_nonpad"] / self.seconds,
                "throughput/cumulative_fresh_rows_per_s": self.counts["fresh_rows"] / self.seconds,
                "throughput/eta_hours": max(0, self.train_steps - step) * window_s / len(self.window) / 3600,
            })
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--echo-factor", type=int, help="override dataset.echo_factor")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 1:
        parser.error("launch with torchrun --nproc_per_node=1")
    if os.environ.get("NANOTRON_BENCHMARK", "0") != "0":
        parser.error("unset NANOTRON_BENCHMARK; it exits training early")
    output = args.output.resolve()
    try:
        baseline = yaml.safe_load(args.config_file.read_text())
        resume_factor = validate_resume(args.resume_from, baseline["model"]["model_config"]) if args.resume_from else None
        config = prepare_config(
            baseline, output, args.resume_from,
            args.echo_factor if args.echo_factor is not None else resume_factor,
        )
        target_steps = config["tokens"]["train_steps"]
        if args.max_steps is not None and not 1 <= args.max_steps <= target_steps:
            raise ValueError(f"max-steps must be in 1..{target_steps}")
    except (OSError, ValueError, KeyError, TypeError, NotImplementedError, yaml.YAMLError) as error:
        parser.error(f"Invalid configuration or resume: {error}")
    output.mkdir(parents=True, exist_ok=False)
    effective = output / "effective.yaml"
    effective.write_text(yaml.safe_dump(config, sort_keys=False))
    os.environ["ASTROPT3_TELEMETRY_DIR"] = str(output / "telemetry")
    torch.set_num_threads(4)

    import torch.distributed as dist
    import wandb

    sys.path.insert(0, str(REPO / "nanotron"))
    from benchmark_throughput import revision
    from nanotron.random import RandomState, RandomStates
    from nanotron.serialize.random import load_random_states
    from run_train import (  # pyright: ignore[reportMissingImports] -- nanotron added above
        DistributedTrainer,
        get_dataloader,
    )

    class LoggedTrainer(DistributedTrainer):
        step_metrics = None

        def save_checkpoint(self):
            stage = self.metadata.current_stage
            stage.consumed_tokens_per_dataset_folder = {
                "catalog-stream": stage.consumed_train_samples * self.sequence_length
            }
            return super().save_checkpoint()

        def train_step_logs(self, outputs, loss_avg, z_loss_avg, tbi_logs):
            super().train_step_logs(outputs, loss_avg, z_loss_avg, tbi_logs)
            self.step_metrics = {
                "train/loss": loss_avg.item(),
                "train/lr": self.lr_scheduler.get_last_lr()[0],
                "train/grad_norm": self.grad_norm_unclipped.item(),
            }
            for key in outputs[0]:
                if key.endswith("_loss") and isinstance(outputs[0][key], torch.Tensor):
                    self.step_metrics[f"train/{key}"] = torch.stack([item[key] for item in outputs]).mean().item()
            if not all(math.isfinite(value) for value in self.step_metrics.values()):
                raise ValueError("Non-finite loss or gradient; refusing to continue or save this step")

    import lsdb

    dataset = config["data_stages"][0]["data"]["dataset"]
    state = {
        "status": "starting", "target_steps": target_steps,
        "echo_factor": dataset["echo_factor"],
        "echo_order": "draw_shuffled_batches_and_modality_spans",
        "stream": dataset["catalog_stream"],
        "provenance": {
            "astropt3": revision(REPO), "nanotron": revision(REPO / "nanotron"),
            "lsdb": revision(Path(lsdb.__file__).resolve().parents[2]),
            "lsdb_source": lsdb.__file__,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    try:
        # Disable only Nanotron's built-in tracker; retain its normal training loop.
        trainer = LoggedTrainer(str(effective), benchmark=True)
        if args.resume_from and trainer.init_checkpoint_path is None:
            raise ValueError("Explicit resume did not load; refusing random initialization")
        if trainer.init_checkpoint_path is not None:
            with torch.serialization.safe_globals([
                RandomState, RandomStates, np.ndarray, np.dtype, type(np.dtype(np.uint32)),
                np._core.multiarray._reconstruct,  # pyright: ignore[reportAttributeAccessIssue] -- checkpoint's NumPy pickle type
            ]):
                restored = load_random_states(trainer.parallel_context, trainer.init_checkpoint_path)
            for name, random_state in restored.items():
                trainer.random_states[name] = random_state
            assert trainer.random_states == restored
            state["rng_restored"] = True
        trainer.config.save_as_yaml(output / "resolved.yaml")
        dataloaders = get_dataloader(trainer)
        throughput = Throughput(
            [modality["name"] for modality in config["model"]["model_config"]["modalities"]],
            target_steps, trainer.metadata.last_train_step,
        )
        state.update(start_step=trainer.metadata.last_train_step, parameters=trainer.num_params)
        with wandb.init(
            entity="smith42", project=config["general"]["project"], mode="online",
            name=output.name, job_type="startup-check" if args.max_steps else "training",
            config=wandb_config(config, args.max_steps), dir=str(output), save_code=False,
            settings=wandb.Settings(
                disable_git=True, disable_code=True, disable_job_creation=True, console="off",
                ignore_globs=["requirements.txt"], x_disable_meta=True,
                x_disable_stats=True, x_disable_machine_info=True,
            ),
        ) as run:
            if not run.url:
                raise RuntimeError("W&B did not establish an online run")
            run.define_metric("train/step")
            run.define_metric("*", step_metric="train/step")
            state.update(status="running", wandb_url=run.url, wandb_id=run.id)
            (output / "run.json").write_text(json.dumps(state, indent=2) + "\n")
            print(f"WANDB_RUN_URL={run.url}", flush=True)
            original_write = telemetry.write_step

            def write_step(step, record, rank=0):
                original_write(step, record, rank)
                metrics = throughput.observe(step, record)
                metrics.update(trainer.step_metrics or {})
                metrics["gpu/allocated_bytes"] = torch.cuda.memory_allocated()
                metrics["gpu/peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                run.log(metrics, step=step)
                state["completed_steps"] = step
                if step % 10 == 0:
                    (output / "progress.json").write_text(json.dumps({**state, "metrics": metrics}, indent=2) + "\n")

            telemetry.write_step = write_step
            trainer.train(dataloaders, max_steps=args.max_steps)
            state.update(status="startup-check-passed" if args.max_steps else "finished",
                         completed_steps=trainer.metadata.last_train_step)
            if not args.max_steps and state["completed_steps"] != target_steps:
                raise RuntimeError(f"Training ended before the requested {target_steps} steps")
            run.summary.update({key: state[key] for key in ("status", "start_step", "completed_steps", "parameters")})
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "run.json").write_text(json.dumps(state, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
