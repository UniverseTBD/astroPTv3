"""Train the 70M preset on live Legacy North x SDSS x DESI x HSC Wide, on one GPU.

Launch with the GPU environment's torchrun, this script, and --output pointing
at a new directory. Use --max-steps 3 for startup validation; the actual run
has a 30,000-step schedule. --resume-from accepts an existing checkpoint root
and still requires a new output directory. No production loader is modified.
All four surveys are always included. Outer joins retain unpaired objects in
regions covered by at least three surveys, within Legacy's anchor coverage. Right-only records remain separate:
without a Legacy match, SDSS/DESI/HSC are not crossmatched with each other.
Adding HSC requires a fresh model, not a three-survey checkpoint resume.
Historical HSC/JWST checkpoints using the reversed zero-point normalization
require their original preprocessing or retraining.
--echo-factor N uses each CPU batch for N updates, reshuffling modality spans
on echoes (default 1). Supply it again on resume; pending echoes are discarded.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from dask.distributed import Client, LocalCluster
from lsdb.loaders.hats.read_hats import open_catalog
from lsdb.streams.catalog_streams import CrossMatchStream

from astropt3.data import nanotron_loader as loader
from astropt3.data import telemetry
from astropt3.data.packing import span_order
from astropt3.tokenization import BOS_ID, PAD_ID

REPO = Path(__file__).resolve().parents[2]
CATALOGS = {
    "legacy": "UniverseTBD/mmu_ssl_legacysurvey_north@f634744d3c44dd4fde0dee3172d4887c5e3c31c0",
    "sdss": "UniverseTBD/mmu_sdss_sdss@da175ca9bb931f4301ab950431922ba9f99089ba",
    "desi": "UniverseTBD/mmu_desi_edr_sv3@9fd88ba48233cb9857701ce802a7eade2d4c4a88",
    "hsc": "hugging-science/mmu_hsc_pdr3_wide_21@7de6b92fd99097207a00b9c42f2d2862d62662bf",
}
NESTED = {
    "image": ("band", "flux", "psf_fwhm"),
    "spectrum_sdss": ("flux", "lambda", "mask"),
    "spectrum_desi": ("flux", "lambda", "mask"),
    "image_hsc": ("band", "flux", "scale"),
}
STREAM = {
    "catalogs": CATALOGS,
    "how": "outer",
    "min_surveys": 3,
    "radius_arcsec": 1.0,
    "count_fraction_threshold": 0.0,
    "partitions_per_chunk": 4,
}


def validate_resume(resume_from: Path) -> None:
    step = (resume_from / "latest.txt").read_text().strip()
    # pi-lens-ignore: unchecked-throwing-call-python -- main reports invalid resume before creating outputs
    saved = json.loads((resume_from / step / "model_config.json").read_text())
    has_hsc = any(modality["name"] == "hsc_images" for modality in saved["modalities"])
    if not has_hsc:
        raise ValueError(
            "Checkpoint has no HSC Wide modality; start a fresh four-survey "
            "run without --resume-from"
        )


def prepare_config(
    baseline: dict,
    output: Path,
    resume_from: Path | None = None,
    echo_factor: int = 1,
) -> dict:
    """Keep the established per-GPU recipe, adding modalities and the run horizon."""
    validate_echo_factor(echo_factor)
    if echo_factor > 1 and baseline["tokens"]["batch_accumulation_per_replica"] != 1:
        raise ValueError("echo-factor > 1 requires batch_accumulation_per_replica=1")
    config = copy.deepcopy(baseline)
    config["general"].update(project="astropt3", run=output.name)
    config["parallelism"].update(dp=1, tp=1)
    model = config["model"]["model_config"]
    model["loss_aggregation"] = "family"
    model["modalities"].append(
        {
            "name": "sdss_spectra",
            "family": "spectrum",
            "source": "sdss",
            "record_keys": ["spectrum_sdss"],
            "token_ids": [17, 18, 19],
            "input_size": 256,
            "patch_size": 256,
            "pos_type": "continuous",
            "pos_input_size": 1,
            "max_positions": 1024,
            "loss_weight": 1.0,
        }
    )
    model["modalities"].append(
        {
            "name": "hsc_images",
            "family": "image",
            "source": "hsc",
            "record_keys": ["hsc_image"],
            "token_ids": [20, 21, 22],
            "input_size": 320,
            "patch_size": 8,
            "pos_type": "index",
            "pos_input_size": 1,
            "max_positions": 144,
            "loss_weight": 1.0,
        }
    )
    config["tokens"]["train_steps"] = 30_000
    config["data_stages"][0]["data"]["num_loading_workers"] = 4
    config["checkpoints"].update(
        checkpoints_path=str(output / "checkpoints"),
        resume_checkpoint_path=str(resume_from) if resume_from else None,
        checkpoint_schedule="pythia",
        checkpoint_interval=1000,
        save_initial_state=False,
        save_final_state=True,
    )
    config.update(profiler=None, lighteval=None, s3_upload=None)
    return config


def wandb_config(config: dict, max_steps: int | None, echo_factor: int = 1) -> dict:
    """Publish training settings only; filesystem paths and provenance stay local."""
    training = {
        key: copy.deepcopy(config[key])
        for key in ("model", "optimizer", "parallelism", "tokens", "data_stages")
    }
    training["seed"] = config["general"]["seed"]
    training["checkpoints"] = {
        key: config["checkpoints"][key]
        for key in (
            "checkpoint_schedule",
            "checkpoint_interval",
            "save_initial_state",
            "save_final_state",
        )
    }
    return {
        "training": training,
        "stream": copy.deepcopy(STREAM),
        "max_steps": max_steps,
        "echo_factor": echo_factor,
        "echo_order": "seeded_uniform_modality_spans",
        "throughput_policy": "synchronized end-to-end wall time including loader/checkpoint stalls; first 10 steps warmup",
    }


def decode_row(mapped):
    row = loader._row_from_map_rows(mapped, NESTED)
    for source, column, id_column in (
        ("legacy", "image", "object_id"),
        ("sdss", "spectrum_sdss", "object_id_sdss"),
        ("desi", "spectrum_desi", "object_id_desi"),
        ("hsc", "image_hsc", "object_id_hsc"),
    ):
        if row[column] is not None:
            object_id = row.get(id_column)
            if bool(pd.isna(object_id)):
                raise ValueError(f"{source} row has no {id_column}")
            # Right-only source IDs must not collide across catalogs.
            object_id = (
                str(object_id) if source == "legacy" else f"{source}:{object_id}"
            )
            break
    else:
        raise ValueError("crossmatched row has no image or spectrum")
    record: dict = {"object_id": object_id}
    if row["image"] is not None:
        record = loader.decode_legacy_row({**record, "image": row["image"]})
    if row["image_hsc"] is not None:
        image = row["image_hsc"]
        flux = loader._stack_nested(image.get("flux")).astype(np.float32, copy=False)
        bands = [str(band) for band in image.get("band", ())]
        if flux.shape != (5, 160, 160) or not np.isfinite(flux).all():
            raise ValueError("HSC Wide flux must be a finite (5, 160, 160) cube")
        if bands != [f"hsc-{band}" for band in "grizy"]:
            raise ValueError("HSC Wide bands must be ordered hsc-g/r/i/z/y")
        scale = np.asarray(image.get("scale"), dtype=np.float32)
        if scale.shape != (5,) or not np.allclose(scale, 0.168, rtol=1e-6, atol=0):
            raise ValueError(
                "HSC Wide pixel scale must be 0.168 arcsec in all five bands"
            )
        record["hsc_image"] = {"flux": flux, "band": bands}
    for column, key in (
        ("spectrum_sdss", "spectrum_sdss"),
        ("spectrum_desi", "spectrum"),
    ):
        if row[column] is not None:
            spectrum = loader._decode_spectrum(row[column])
            if key == "spectrum_sdss":
                # SDSS nonpositive wavelengths are padding; retain NaNs for strict validation.
                keep = ~(spectrum["lambda"] <= 0)
                spectrum = {name: values[keep] for name, values in spectrum.items()}
            record[key] = spectrum
    return {"record": record}


class ThreeCatalogBatches(loader.PackedMicroBatches):
    """Reuse the sequencer, packer, and flat batch contract; change only live records."""

    def _open_records(self, worker):
        worker_id = worker.id if worker else 0
        failures = generation = draw = 0
        settings = STREAM
        repos = CATALOGS
        telemetry._REPO_TO_SOURCE.update(
            {repo.split("@")[0]: name for name, repo in repos.items()}
        )
        while True:
            try:
                with LocalCluster(
                    processes=False,
                    n_workers=1,
                    threads_per_worker=4,
                    dashboard_address=None,  # pyright: ignore[reportArgumentType] -- documented disable value
                    protocol="inproc",
                    memory_limit="64GiB",
                ) as cluster, Client(cluster) as client:
                    catalogs = {}
                    for name, repo in repos.items():
                        payload = "image" if name in ("legacy", "hsc") else "spectrum"
                        catalogs[name] = open_catalog(
                            f"hf://datasets/{repo}",
                            columns=["object_id", "ra", "dec", payload],
                        )
                    stream: Any = CrossMatchStream(
                        catalogs["legacy"],
                        *[
                            {
                                "other": catalog,
                                "how": settings["how"],
                                "suffixes": ("", f"_{name}"),
                                "radius_arcsec": settings["radius_arcsec"],
                            }
                            for name, catalog in catalogs.items()
                            if name != "legacy"
                        ],
                        min_surveys=settings["min_surveys"],  # pyright: ignore[reportCallIssue] -- launcher's LSDB worktree API
                        count_fraction_threshold=settings["count_fraction_threshold"],
                        client=client,
                        partitions_per_chunk=settings["partitions_per_chunk"],
                        seed=loader.consumer_seed(
                            self.seed, self.rank, worker_id, generation
                        ),
                    )
                    if not stream.npartitions:
                        raise ValueError(
                            "The selected catalog footprint contains no partitions"
                        )
                    print(
                        f"[catalog-stream] worker={worker_id} pixels={stream.npartitions} "
                        f"generation={generation} catalogs={len(repos)} min_surveys=3",
                        flush=True,
                    )
                    for frame in stream:
                        failures = 0
                        self._epoch = draw
                        draw += 1
                        if frame.empty:
                            continue
                        columns = (
                            ["object_id"]
                            + [
                                f"object_id_{name}"
                                for name in repos
                                if name != "legacy"
                            ]
                            + [
                                f"{name}.{field}"
                                for name, fields in NESTED.items()
                                for field in fields
                            ]
                        )
                        decoded = frame.map_rows(
                            decode_row, columns=columns, infer_nesting=False
                        )
                        yield from decoded["record"]
                raise RuntimeError("Infinite crossmatch stream ended unexpectedly")
            except Exception as error:
                if not loader._retryable(error):
                    raise
                failures += 1
                if failures > loader._MAX_NET_RETRIES:
                    raise
                wait = min(5 * 2 ** (failures - 1), loader._MAX_NET_RETRY_WAIT)
                print(
                    f"[catalog-stream] worker={worker_id} retry={failures} "
                    f"error={type(error).__name__} wait_s={wait}",
                    flush=True,
                )
                time.sleep(wait)
                generation += 1


def validate_echo_factor(factor):
    if type(factor) is not int or factor < 1:
        raise ValueError("echo-factor must be a positive integer")


class EchoLoader:
    """Cache one CPU batch; tensors are read-only, each yielded mapping is owned.

    Only modality spans move, never objects or patches within a modality, so
    row-major modality values/positions and document position_ids stay aligned.
    """

    _loader: Any = None

    def __init__(self, batches, config, echo_factor=1, seed=0):
        validate_echo_factor(echo_factor)
        self._loader = batches
        self.echo_factor = echo_factor
        self.seed = seed
        self.echo_index = 0
        registry = config.modality_registry()
        self.token_ids = {
            name: registry.get_config(name).token_ids for name in registry.names()
        }

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __len__(self):
        return len(self._loader) * self.echo_factor

    def __iter__(self):
        for group, batch in enumerate(self._loader):
            for index in range(self.echo_factor):
                self.echo_index = index
                yield self._reshuffle(batch, group, index) if index else dict(batch)

    def _reshuffle(self, batch, group, index):
        original = batch["input_ids"]
        if original.device.type != "cpu":
            raise ValueError("EchoLoader requires CPU batches before device transfer")
        ids = torch.empty_like(original, pin_memory=original.is_pinned())
        ids.copy_(original)
        by_begin = {
            begin: (name, placeholder, end)
            for name, (begin, placeholder, end) in self.token_ids.items()
        }
        for row_index, row in enumerate(original.tolist()):
            cursor = 0
            while cursor < len(row) and row[cursor] != PAD_ID:
                if row[cursor] != BOS_ID:
                    raise ValueError("Echo batch must contain whole packed objects")
                start = cursor + 1
                cursor = start
                spans = {}
                while cursor < len(row) and row[cursor] not in (BOS_ID, PAD_ID):
                    if row[cursor] not in by_begin:
                        raise ValueError("Invalid modality delimiter in echo batch")
                    name, placeholder, end_id = by_begin[row[cursor]]
                    end = row.index(end_id, cursor + 1)
                    if name in spans or any(
                        token != placeholder for token in row[cursor + 1:end]
                    ):
                        raise ValueError(
                            "Echo batch requires one contiguous span per modality per object"
                        )
                    spans[name] = row[cursor:end + 1]
                    cursor = end + 1
                if len(spans) > 1:
                    # Packed batches no longer carry object IDs; use batch-local identity.
                    order = span_order(
                        sorted(spans), f"echo:{group}:{row_index}:{start}:{index}", self.seed
                    )
                    ids[row_index, start:cursor] = torch.tensor(
                        [token for name in order for token in spans[name]], dtype=ids.dtype
                    )
        echoed = {**batch, "input_ids": ids}
        for name, (_, placeholder, _) in self.token_ids.items():
            mask = batch[f"{name}_mask"]
            echoed[f"{name}_mask"] = torch.empty_like(mask, pin_memory=mask.is_pinned())
            torch.eq(ids, placeholder, out=echoed[f"{name}_mask"])
        return echoed


def instrument_echo(batches, *, config, echo_factor, seed):
    """Replay after worker prefetch, before the existing timing/device wrappers."""
    return telemetry.instrument(EchoLoader(batches, config, echo_factor, seed))


def echo_record(record, echo_index, echo_factor):
    """Fresh means newly delivered, not unique objects, remote reads, or bytes."""
    result = {**record, "echo_index": echo_index, "echo_factor": echo_factor}
    for prefix, include in (("fresh", echo_index == 0), ("replayed", echo_index != 0)):
        for key in ("micro_batches", "tokens_nonpad", "rows"):
            result[f"{prefix}_{key}"] = record[key] if include else 0
        for key in ("loss_tokens", "target_values"):
            result[f"{prefix}_{key}"] = {
                name: count if include else 0 for name, count in record[key].items()
            }
    return result


class Throughput:
    """Aggregate counts over elapsed wall time; never average per-step rates."""

    def __init__(self, start_step=0):
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
        }
        metrics["data/echo_index"] = record["echo_index"]
        metrics["data/echo_factor"] = record["echo_factor"]
        for prefix in ("fresh", "replayed"):
            for key in ("micro_batches", "tokens_nonpad", "rows"):
                metrics[f"data/{prefix}_{key}"] = record[f"{prefix}_{key}"]
        total = sum(record["loss_tokens"].values())
        for name in ("images", "spectra", "sdss_spectra", "hsc_images"):
            count = record["loss_tokens"].get(name, 0)
            metrics[f"data/{name}_loss_tokens"] = count
            metrics[f"data/{name}_token_fraction"] = count / total if total else 0.0
            for prefix in ("fresh", "replayed"):
                metrics[f"data/{prefix}_{name}_loss_tokens"] = record[
                    f"{prefix}_loss_tokens"
                ].get(name, 0)
        if step > self.warmup_end:
            self.window.append(record)
            for key in self.counts:
                self.counts[key] += record[key]
            self.seconds += seconds
            window_s = sum(item["step_seconds"] for item in self.window)
            metrics.update(
                {
                    "throughput/rolling_nonpad_tokens_per_s": sum(
                        item["tokens_nonpad"] for item in self.window
                    )
                    / window_s,
                    "throughput/cumulative_nonpad_tokens_per_s": self.counts["tokens_nonpad"] / self.seconds,
                    "throughput/cumulative_rows_per_s": self.counts["rows"] / self.seconds,
                    "throughput/cumulative_fresh_nonpad_tokens_per_s": self.counts["fresh_tokens_nonpad"] / self.seconds,
                    "throughput/cumulative_fresh_rows_per_s": self.counts["fresh_rows"] / self.seconds,
                    "throughput/eta_hours": (30_000 - step)
                    * window_s
                    / len(self.window)
                    / 3600,
                }
            )
        return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--echo-factor", type=int, default=1)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 1:
        parser.error("launch with torchrun --nproc_per_node=1")
    if args.max_steps is not None and not 1 <= args.max_steps <= 30_000:
        parser.error("max-steps must be in 1..30000")
    if os.environ.get("NANOTRON_BENCHMARK", "0") != "0":
        parser.error("unset NANOTRON_BENCHMARK; it exits training early")
    if args.resume_from:
        try:
            validate_resume(args.resume_from)
        except (OSError, ValueError, KeyError) as error:
            parser.error(f"Invalid resume: {error}")
    output = args.output.resolve()
    baseline = yaml.safe_load(
        (REPO / "astro/configs/nanotron/astropt3-70m-dp1.yaml").read_text()
    )
    try:
        config = prepare_config(baseline, output, args.resume_from, args.echo_factor)
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=False)
    effective = output / "effective.yaml"
    effective.write_text(yaml.safe_dump(config, sort_keys=False))
    os.environ["ASTROPT3_TELEMETRY_DIR"] = str(output / "telemetry")
    torch.set_num_threads(4)
    loader.PackedMicroBatches = ThreeCatalogBatches
    loader.instrument = functools.partial(
        instrument_echo,
        config=loader.hf_config_from_modalities(config["model"]["model_config"]["modalities"]),
        echo_factor=args.echo_factor,
        seed=config["general"]["seed"],
    )
    loader.build_astropt3_dataloader = functools.partial(
        loader.build_astropt3_dataloader, multiprocessing_context="spawn"
    )

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
                "four-catalog-stream": stage.consumed_train_samples
                * self.sequence_length
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
                    self.step_metrics[f"train/{key}"] = (
                        torch.stack([item[key] for item in outputs]).mean().item()
                    )
            if not all(math.isfinite(value) for value in self.step_metrics.values()):
                raise ValueError(
                    "Non-finite loss or gradient; refusing to continue or save this step"
                )

    import lsdb

    provenance = {
        "astropt3": revision(REPO),
        "nanotron": revision(REPO / "nanotron"),
        "lsdb": revision(Path(lsdb.__file__).resolve().parents[2]),
        "lsdb_source": lsdb.__file__,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    state = {
        "status": "starting",
        "target_steps": 30_000,
        "echo_factor": args.echo_factor,
        "echo_order": "seeded_uniform_modality_spans",
        "stream": STREAM,
        "provenance": provenance,
    }
    try:
        # This flag only disables nanotron's built-in W&B/MFU logging; the normal
        # training loop, optimizer, schedule, and checkpoint policy remain active.
        trainer = LoggedTrainer(str(effective), benchmark=True)
        if args.resume_from and trainer.init_checkpoint_path is None:
            raise ValueError(
                "Explicit resume did not load; refusing random initialization"
            )
        if trainer.init_checkpoint_path is not None:
            # Allow only the concrete RNG container types, not arbitrary checkpoint pickle code.
            with torch.serialization.safe_globals(
                [
                    RandomState,
                    RandomStates,
                    np.ndarray,
                    np.dtype,
                    type(np.dtype(np.uint32)),
                    np._core.multiarray._reconstruct,  # pyright: ignore[reportAttributeAccessIssue] -- checkpoint's NumPy pickle type
                ]
            ):
                restored = load_random_states(
                    trainer.parallel_context, trainer.init_checkpoint_path
                )
            for name, random_state in restored.items():
                trainer.random_states[name] = random_state
            assert trainer.random_states == restored
            state["rng_restored"] = True
        trainer.config.save_as_yaml(output / "resolved.yaml")
        dataloaders = get_dataloader(trainer)
        throughput = Throughput(trainer.metadata.last_train_step)
        state.update(
            start_step=trainer.metadata.last_train_step, parameters=trainer.num_params
        )
        with wandb.init(
            entity="smith42",
            project="astropt3",
            mode="online",
            name=output.name,
            job_type="startup-check" if args.max_steps else "training",
            config=wandb_config(config, args.max_steps, args.echo_factor),
            dir=str(output),
            save_code=False,
            settings=wandb.Settings(
                disable_git=True,
                disable_code=True,
                disable_job_creation=True,
                console="off",
                ignore_globs=["requirements.txt"],
                x_disable_meta=True,
                x_disable_stats=True,
                x_disable_machine_info=True,
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
                record = echo_record(record, trainer.current_base_dl.echo_index, args.echo_factor)
                original_write(step, record, rank)
                metrics = throughput.observe(step, record)
                metrics.update(trainer.step_metrics or {})
                metrics["gpu/allocated_bytes"] = torch.cuda.memory_allocated()
                metrics["gpu/peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                run.log(metrics, step=step)
                state["completed_steps"] = step
                if step % 10 == 0:
                    (output / "progress.json").write_text(
                        json.dumps({**state, "metrics": metrics}, indent=2) + "\n"
                    )

            telemetry.write_step = write_step
            trainer.train(dataloaders, max_steps=args.max_steps)
            state.update(
                status="startup-check-passed" if args.max_steps else "finished",
                completed_steps=trainer.metadata.last_train_step,
            )
            if not args.max_steps and state["completed_steps"] != 30_000:
                raise RuntimeError("Training ended before the requested 30,000 steps")
            run.summary.update(
                {
                    key: state[key]
                    for key in ("status", "start_step", "completed_steps", "parameters")
                }
            )
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (output / "run.json").write_text(json.dumps(state, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
