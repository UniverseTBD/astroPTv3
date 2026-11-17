"""Small fixed-draw HF A/B, NOT a full five-catalog training replication.

Run in the training venv; pin CUDA_VISIBLE_DEVICES externally (or hide CUDA
for CPU). Source patches have any pixel/channel >4 robust sigma from that
object's channel median in sinh(tokens)*image_norm_divisor units. Sigma is
1.4826*MAD with an epsilon floor: an approximate source/sky split, not the
issue study's segmentation. Reported NLL is nats/token, not bits/dimension;
scalars are likelihoods, not redshift accuracy or representation probes.
"""
import argparse
import copy
import importlib
import json
import os
import re
import signal
import sys
import time
from contextlib import closing
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from astropt3 import AstroPT3Config, AstroPT3Model
from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.flows import flow_object_ids
from astropt3.modalities import gmm_nll, std_normal_nll
from astropt3.modeling_astropt3 import left_shift_mask


def source_mask(values, modality, divisor):
    """One object's image tokens -> source-token mask, including band-major data."""
    bands = len(modality.band_order) if modality.channel_tokenization == "per_band" else 1
    pixels = values.float().sinh() * divisor
    if bands > 1:
        pixels = pixels.reshape(bands, -1, modality.patch_size**2).permute(1, 2, 0)
    else:
        pixels = pixels.reshape(len(values), modality.patch_size**2, -1)
    flat = pixels.flatten(0, 1)
    median = flat.median(dim=0).values
    sigma = 1.4826 * (flat - median).abs().median(dim=0).values
    mask = ((pixels - median).abs() > 4 * sigma.clamp_min(torch.finfo(pixels.dtype).eps)).any(dim=(1, 2))
    return mask.repeat(bands)


def accumulate(stats, name, ar, prior, logdet):
    """Accumulate sums, never averages of unequal-size batches/modalities."""
    if not ar.numel():
        return
    row = stats.setdefault(name, {"tokens": 0, "ar": 0.0, "prior": 0.0, "logdet": 0.0})
    row["tokens"] += ar.numel()
    for key, values in (("ar", ar), ("prior", prior), ("logdet", logdet)):
        row[key] += values.double().sum().item()


def summarize(stats):
    result = {}
    for name, row in stats.items():
        means = {key: row[key] / row["tokens"] for key in ("ar", "prior", "logdet")}
        result[name] = {"tokens": row["tokens"], **means, "nll": means["ar"] + means["prior"] - means["logdet"]}
    return result


def collect(stream, sequencer, count, seed, max_draws=100):
    objects, seen, visited = [], set(), 0
    with closing(stream.draws(seed=seed, rank=0, worker_id=0)) as draws:
        for _, records in islice(draws, max_draws):
            for record in records:
                visited += 1
                if visited > count * 100:
                    raise RuntimeError(f"Collection exhausted record budget: {len(objects)}/{count} unique objects")
                obj = sequencer.build(record, epoch=0)
                if not obj.object_id:
                    raise ValueError("Cannot deduplicate an empty object_id")
                if obj.object_id not in seen:
                    seen.add(obj.object_id)
                    objects.append(obj)
                    if len(objects) == count:
                        return objects
    raise RuntimeError(f"Collection exhausted {max_draws} draws: {len(objects)}/{count} unique objects")


def move(batch, device):
    return {key: move(value, device) if isinstance(value, dict) else value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    stats = {}
    registry = model.config.modality_registry()
    for cpu_batch in batches:
        batch = move(cpu_batch, device)
        hidden = model(**batch, compute_loss=False).last_hidden_state
        for name, values in batch["modality_values"].items():
            mask = batch["modality_masks"][name]
            ids = flow_object_ids(batch["position_ids"], mask)
            z, logdet = values, values.new_zeros(len(values))
            if name not in model.scalar_names:
                z, logdet = model.flows[name](values, batch["modality_positions"][name], ids)
            d_ar = model.ar_dims[name]
            ar = gmm_nll(z[:, :d_ar], *model.decoders[name](hidden[left_shift_mask(mask)]))
            prior = std_normal_nll(z[:, d_ar:])
            accumulate(stats, name, ar, prior, logdet)
            accumulate(stats, "total", ar, prior, logdet)
            modality = registry.get_config(name)
            if modality.family == "image":
                source = torch.zeros_like(ar, dtype=torch.bool)
                for object_id in ids.unique():
                    selected = ids == object_id
                    source[selected] = source_mask(values[selected], modality, model.config.image_norm_divisor)
                for region, selected in (("source", source), ("sky", ~source)):
                    accumulate(stats, f"{name}/{region}", ar[selected], prior[selected], logdet[selected])
                    accumulate(stats, f"images/{region}/total", ar[selected], prior[selected], logdet[selected])
    return summarize(stats)


def arm_config(config, fraction):
    config = copy.deepcopy(config)
    for modality in config.modalities:
        if modality["family"] != "scalar":
            modality["d_ar"] = max(1, int(modality["input_size"] * fraction))
    return config


def train_arm(config, baseline, train, evaluation, recipe, args, device) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    model = AstroPT3Model(config)
    own = model.state_dict()
    shared = {key: value for key, value in baseline.items() if own[key].shape == value.shape}
    model.load_state_dict(shared, strict=False)
    del own
    model.to(device).train()
    opt = recipe["optimizer"]
    adam = opt["optimizer_factory"]
    excluded = opt.get("weight_decay_exclude_named_params", [])
    def decay(name, param):
        return param.ndim >= 2 and not any(re.search(pattern, name) for pattern in excluded)
    groups = [{"params": [p for n, p in model.named_parameters() if decay(n, p) == use_decay],
               "weight_decay": opt["weight_decay"] if use_decay else 0.0} for use_decay in (True, False)]
    optimizer = torch.optim.AdamW(groups, lr=opt["learning_rate_scheduler"]["learning_rate"],
                                 betas=(adam["adam_beta1"], adam["adam_beta2"]), eps=adam["adam_eps"])
    torch.manual_seed(args.seed)  # identical full-width input-noise draws across arms
    warmup = min(5, args.steps - 1)
    curves, times = [], []
    for step in range(args.steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
            if step == warmup:
                torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        batch = move(train[step % len(train)], device)
        model.set_jet_noise_frac((step + 1) / recipe["tokens"]["train_steps"])
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        if not torch.isfinite(out.loss):
            raise RuntimeError(f"Nonfinite loss at step {step + 1}")
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.get("clip_grad", 1.0), error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        if step >= warmup:
            times.append(duration)
        curves.append({"step": step + 1, "loss": out.loss.item()})
        del out, batch
    result: dict[str, Any] = {"d_ar": model.ar_dims, "parameters": sum(p.numel() for p in model.parameters()),
                    "shared_state_tensors": len(shared), "warmup_steps": warmup, "timed_steps": len(times),
                    "mean_step_seconds": sum(times) / len(times), "training_curve": curves,
                    "peak_train_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
    optimizer.zero_grad(set_to_none=True)
    result["eval"] = evaluate(model, evaluation, device)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, default=Path(__file__).resolve().parents[1] / "configs/nanotron/astropt3-70m.yaml")
    parser.add_argument("--output", type=Path, required=True)
    for name, default in (("steps", 100), ("train-objects", 64), ("eval-objects", 32), ("batch-objects", 4), ("seed", 0)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.4, 0.2])
    args = parser.parse_args()
    if min(args.steps, args.train_objects, args.eval_objects, args.batch_objects) < 1 or not all(0 < f <= 1 for f in args.fractions):
        parser.error("counts must be positive and fractions in (0, 1]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        parser.error("CUDA requires an explicit CUDA_VISIBLE_DEVICES pin")
    from astropt3.data import catalog_stream

    # Use the release converter, including nanotron dataclass defaults, not HF defaults.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nanotron/tools/astropt3"))
    get_config_mapping = importlib.import_module("convert_weights").get_config_mapping
    recipe = yaml.safe_load(args.config_file.read_text())
    nt = importlib.import_module("nanotron.config").AstroPT3Config(**recipe["model"]["model_config"])
    config = AstroPT3Config(**{hf: getattr(nt, name) for name, hf in get_config_mapping().items()})
    config._attn_implementation = "sdpa"
    spec = recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "recipe.yaml").write_text(args.config_file.read_text())
    config.to_json_file(args.output / "hf_config.json")
    catalog_stream._DEQUANTIZE = np.random.default_rng(args.seed)

    def deadline(_signum, _frame):
        raise RuntimeError("Catalog collection exceeded 900 seconds")  # not a retryable transport timeout

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        objects = collect(catalog_stream.CatalogStream(spec, config), ObjectSequencer(config),
                          args.train_objects + args.eval_objects, args.seed)
    except Exception as error:
        (args.output / "result.json").write_text(json.dumps({"status": "collection_failed", "error": str(error)}))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    collate = PackedCollator(config, seq_len=recipe["tokens"]["sequence_length"])
    train, evaluation = [
        [collate(split[i:i + args.batch_objects]) for i in range(0, len(split), args.batch_objects)]
        for split in (objects[:args.train_objects], objects[args.train_objects:])]
    torch.save({"objects": [vars(o) for o in objects], "train_batches": train, "eval_batches": evaluation}, args.output / "data.pt")
    report = {"status": "running", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch_version": torch.__version__, "train_ids": [o.object_id for o in objects[:args.train_objects]],
              "eval_ids": [o.object_id for o in objects[args.train_objects:]], "arms": [],
              "differences": "HF FP32/SDPA, single device, fresh AdamW at constant recipe peak LR (no LR schedule); "
              "small fixed batches repeated in order, fixed span order, no distributed/echo pipeline; "
              "noise uses original recipe step horizon. Small controlled draw, not full five-catalog replication.",
              "metrics": __doc__, "collection_limits": {"draws": 100, "records": 100 * len(objects), "seconds": 900}}
    torch.manual_seed(args.seed)
    baseline = AstroPT3Model(arm_config(config, 1.0)).state_dict()
    for fraction in args.fractions:
        result = train_arm(arm_config(config, fraction), baseline, train, evaluation, recipe, args, device)
        report["arms"].append({"fraction": fraction, **result})
        (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"fraction={fraction:g} params={result['parameters']:,} step={result['mean_step_seconds']:.3f}s "
              f"peak_bytes={result['peak_train_memory_bytes']} eval_nll={result['eval']['total']['nll']:.4f}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["status"] = "complete"
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
