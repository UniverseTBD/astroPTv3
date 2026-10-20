"""ADR 0014 §3 benchmark report: join the byte log, step log and object audit.

Usage:
    python scripts/bench_report.py --telemetry <dir> [--object-log <prefix>]
                                   [--arm B0] [--out report.json]

Reads what the instruments wrote during a benchmark window:

- ``bytes.rank*.jsonl``   — one line per returned payload (§3)
- ``steps.rank*.jsonl``   — one line per step, with the decomposed MFU (§2a)
- ``{object_log}.dp*``    — one line per emitted sequence (the no-replay audit)

(older runs wrote ``bytes.dp*``/``steps.dp*``; both are read) and reports the
four metrics of §2 plus the timing distribution. A directory written by
``benchmark_throughput.py`` (it has ``run.json``) gets the schema-v1 report
instead. Two rules from the ADR are enforced here rather than left to the reader:

- MFU is only ever printed decomposed (§11 refuses the headline number);
- ``E_AR`` is broken into primary and replay tokens, because replay raises it
  without adding information and the split is what makes that visible.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
TIB = 1024**4


def _read_jsonl(paths):
    for path in paths:
        text = Path(path).read_text()
        # Only newline-terminated lines are records: a loader worker may still
        # be appending its byte log while the benchmark reports.
        for line in text.split("\n")[:-1]:
            if line.strip():
                yield json.loads(line)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; no numpy dependency for a reporting script."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


def summarise_bytes(directory: Path) -> dict:
    return _summarise_byte_records(_read_jsonl(sorted(directory.glob("bytes.*.jsonl"))))


def _summarise_byte_records(records) -> dict:
    by_source: Counter = Counter()
    fetches: Counter = Counter()
    wait = 0.0
    for entry in records:
        by_source[entry["source"]] += entry["bytes"]
        fetches[entry["source"]] += 1
        wait += entry.get("wait_s", 0.0)
    total = sum(by_source.values())
    return {
        "total_bytes": total,
        "total_mib": total / MIB,
        "fetch_wait_s": wait,
        "by_source": {
            source: {
                "bytes": count,
                "mib": count / MIB,
                "share": count / total if total else 0.0,
                "fetches": fetches[source],
            }
            for source, count in by_source.most_common()
        },
    }


def _step_seconds(record: dict) -> float:
    """Wall time of one step, preferring the value the trainer recorded."""
    if "step_seconds" in record:
        return record["step_seconds"]
    # older records: recover it from wait / stall_share (exact whenever the
    # step stalled at all; a step with no stall contributes ~0 either way)
    stall = record.get("stall_share") or 0.0
    return record["loader_wait_s"] / stall if stall else record.get("loader_wait_s", 0.0)


def summarise_steps(directory: Path, slow_threshold: float) -> dict:
    paths = sorted(directory.glob("steps.*.jsonl"))
    return _summarise_step_series([list(_read_jsonl([path])) for path in paths], slow_threshold)


def _summarise_step_series(series: list[list[dict]], slow_threshold: float) -> dict:
    """Aggregate one record list per rank file (world rank; DP rank in older logs)."""
    records = [record for entries in series for record in entries]
    if not records:
        return {}
    n_ranks = max(record.get("world_size", 0) for record in records) or len(series)
    # Older logs lack tp_rank: each file is then one DP rank.
    dp_ranks = sum(any(record.get("tp_rank", 0) == 0 for record in entries) for entries in series)

    # Every DP rank writes its own series for the SAME steps — they run
    # concurrently, so concatenating them would report 2x the steps. Group by
    # step number: sum TP0 token counts (TP replicas share the same data),
    # take the slowest world rank's wall time (the step ends when the last
    # rank does), and take the flops estimate once (it is already global).
    by_step: dict[int, dict] = {}
    for record in records:
        step = record.get("step", 0)
        merged = by_step.setdefault(
            step,
            {"step": step, "seconds": 0.0, "wait": 0.0, "flops": 0.0,
             "nonpad": 0, "total": 0, "rows": 0, "loss_tokens": Counter(),
             "target_values": 0, "ranks": 0},
        )
        merged["ranks"] += 1
        merged["seconds"] = max(merged["seconds"], _step_seconds(record))
        merged["wait"] = max(merged["wait"], record.get("loader_wait_s", 0.0))
        # `model_flops` is already the GLOBAL per-step estimate and every rank
        # writes the same number, so take it rather than summing. Older
        # records carry only the ratio: invert it (mfu's denominator includes
        # world_size) so a pre-`model_flops` window still reports a real MFU.
        flops = record.get("model_flops")
        if flops is None:
            flops = (
                record.get("mfu", 0.0)
                * _step_seconds(record)
                * record.get("peak_tflops_per_gpu", 0.0)
                * 1e12
                * n_ranks
            )
        merged["flops"] = max(merged["flops"], flops)
        if record.get("tp_rank", 0) == 0:
            merged["nonpad"] += record.get("tokens_nonpad", 0)
            merged["total"] += record.get("tokens_total", 0)
            merged["rows"] += record.get("rows", 0)
            merged["loss_tokens"].update(record.get("loss_tokens", {}))
            merged["target_values"] += sum(record.get("target_values", {}).values())

    steps = [by_step[k] for k in sorted(by_step)]
    durations = [s["seconds"] for s in steps]
    total_wall = sum(durations) or 1e-9
    total_wait = sum(s["wait"] for s in steps)
    loss_tokens: Counter = Counter()
    for step in steps:
        loss_tokens.update(step["loss_tokens"])
    total_loss_tokens = sum(loss_tokens.values())
    nonpad = sum(s["nonpad"] for s in steps)
    tokens_total = sum(s["total"] for s in steps)
    rows = sum(s["rows"] for s in steps)

    # TIME-WEIGHTED, not a mean of per-step ratios. This corpus is bimodal —
    # most steps stall for nothing, a few for a minute — so averaging ratios
    # reports a stall share several times lower than the one the run actually
    # paid, and buries the very behaviour being measured.
    peak = next(
        (r["peak_tflops_per_gpu"] for r in records if "peak_tflops_per_gpu" in r), 0.0
    )
    gpus = n_ranks
    total_flops = sum(s["flops"] for s in steps)
    denominator = peak * 1e12 * gpus
    busy = max(total_wall - total_wait, 1e-9)
    packing = nonpad / tokens_total if tokens_total else 0.0
    # `model_flops` already refuses padding credit, so the packing factor is
    # baked into it. MFU_busy must therefore be DEFLATED by packing to be the
    # full-occupancy busy number §2a means, or the printed identity
    # MFU = MFU_busy x (1 - stall) x packing double-counts it. With this,
    # the three factors multiply back to MFU exactly.
    busy_flops = total_flops / packing if packing else 0.0

    return {
        "n_steps": len(steps),
        "dp_ranks": dp_ranks,
        "world_size": gpus,
        "step_seconds": {
            "mean": total_wall / len(steps),
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
            "slow_steps": sum(1 for d in durations if d > slow_threshold),
            "slow_threshold_s": slow_threshold,
            "total_s": total_wall,
        },
        # §11: never a headline MFU on its own — the factors always travel
        # with it, or a pipeline effect gets read as a model-shape effect
        "mfu": {
            "mfu": total_flops / (total_wall * denominator) if denominator else 0.0,
            "mfu_busy": busy_flops / (busy * denominator) if denominator else 0.0,
            "stall_share": total_wait / total_wall,
            "utilisation_packing": packing,
            "flops_per_token": (
                total_flops / (3 * total_loss_tokens) if total_loss_tokens else 0.0
            ),
            "peak_tflops_per_gpu": peak,
        },
        "loader_wait_s": total_wait,
        "tokens_nonpad": nonpad,
        "tokens_total": tokens_total,
        "rows": rows,
        # end-to-end, wall-clock throughput — the number a fair loader-backend
        # comparison wants (galaxies actually delivered per second of the run,
        # not per second of stall — see telemetry.drain_step's rows_per_s for
        # the loader-only figure this is time-weighted against).
        "rows_per_s": rows / total_wall,
        "target_values": sum(s["target_values"] for s in steps),
        "loss_tokens": dict(loss_tokens.most_common()),
        # §4: realised modality composition. Short windows swing 160x on this
        # corpus, so an A/B that does not report it is not comparing like
        # with like.
        "composition": {
            name: count / total_loss_tokens
            for name, count in loss_tokens.most_common()
        }
        if total_loss_tokens
        else {},
    }


def summarise_objects(prefix: Path | None) -> dict:
    """Split emitted sequences into primary and replay, from the audit log."""
    if prefix is None:
        return {}
    logs = sorted(prefix.parent.glob(f"{prefix.name}.dp*"))
    if not logs:
        return {}
    primary, replicas, seen = 0, 0, set()
    bases = set()
    duplicates = 0
    for path in logs:
        with open(path) as handle:
            for line in handle:
                object_id = line.strip()
                if not object_id:
                    continue
                if object_id in seen:
                    duplicates += 1
                seen.add(object_id)
                base, _, suffix = object_id.partition("#")
                bases.add(base)
                if suffix:
                    replicas += 1
                else:
                    primary += 1
    return {
        "sequences": primary + replicas,
        "primary": primary,
        "replay": replicas,
        "distinct_base_objects": len(bases),
        # the exactly-once gate: replica 0 keeps the base id, replicas are
        # suffixed, so every logged line must be unique
        "duplicate_lines": duplicates,
        "replay_factor": (primary + replicas) / len(bases) if bases else 0.0,
    }


def _load(path: Path, jsonl: bool = False) -> Any:
    try:
        return list(_read_jsonl([path])) if jsonl else json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def _benchmark_mfu(metas: list[dict], window: list[list[dict]], seconds: float,
                   packing: float, stall_share: float) -> dict:
    """Estimated MFU: measured TP0 tokens priced with the model's own FLOP counts,
    over the rated dense BF16 peak of every participating GPU (docs/training.md §8)."""
    peaks = [meta["gpu"]["peak_tflops_per_gpu"] for meta in metas]
    if None in peaks:
        return {"mfu": None, "unavailable_reason": "GPU variant not in the dense BF16 peak table"}
    if any("bfloat16" not in meta["flops_estimate"]["dtype"] for meta in metas):
        return {"mfu": None, "unavailable_reason": "peak table is BF16-only"}
    flops = 0.0
    for meta, records in zip(metas, window, strict=True):
        costs = meta["flops_estimate"]
        for record in records if meta["tp_rank"] == 0 else []:
            flops += costs["backbone_flops_per_token"] * record["tokens_nonpad"]
            # forward + backward = 3x the forward cost of each modality head
            flops += 3 * sum(count * costs["modality_forward_flops_per_token"][name]
                             for name, count in record["loss_tokens"].items())
    mfu = flops / (seconds * sum(peaks) * 1e12)
    return {
        "mfu": mfu,
        "mfu_busy": mfu / ((1 - stall_share) * packing) if packing and stall_share < 1 else None,
        "model_flops": flops,
        "model_tflops_per_gpu": flops / seconds / len(metas) / 1e12,
    }


def build_benchmark_report(directory: Path, slow: float = 60.0) -> dict:
    """Schema-v1 report: ``complete`` only if every world rank ran the exact window."""
    report = {"schema_version": 1, "status": "partial", "telemetry_dir": str(directory),
              "diagnostics": [], "warnings": []}
    problems = report["diagnostics"]
    try:
        run = report["run"] = _load(directory / "run.json")
        world, warmup, measured = run["world_size"], run["warmup_steps"], run["steps"]
        metas = [_load(directory / f"rank{rank}.json") for rank in range(world)]
        series = [_load(directory / f"steps.rank{rank}.jsonl", jsonl=True) for rank in range(world)]
    except (ValueError, KeyError) as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
        return report
    report["ranks"] = {str(rank): meta for rank, meta in enumerate(metas)}
    for rank, (meta, records) in enumerate(zip(metas, series, strict=True)):
        if meta.get("status") != "complete":
            problems.append(f"rank {rank}: status is {meta.get('status')!r}, not 'complete'")
        start = meta.get("start_step", 0)
        if [record.get("step") for record in records] != list(range(start + 1, start + warmup + measured + 1)):
            problems.append(f"steps.rank{rank}.jsonl: expected exactly steps {start + 1}-{start + warmup + measured}")
        for consumer in [f"w{worker}" for worker in range(meta.get("num_loading_workers", 0))] or ["main"]:
            if not (directory / f"bytes.rank{rank}.{consumer}.jsonl").is_file():
                problems.append(f"bytes.rank{rank}.{consumer}.jsonl: missing (byte probe never installed)")
    if problems:
        return report

    try:
        # Every step ends with a CUDA sync and a world barrier, so the summed
        # slowest-rank step times are the window's synchronized wall time.
        window = _summarise_step_series([records[warmup:] for records in series], slow)
        seconds = window["step_seconds"]["total_s"]
        stalls = window.pop("mfu")  # legacy per-step MFU is not the benchmark's GPU-aware estimate
        mfu = _benchmark_mfu(metas, [records[warmup:] for records in series], seconds,
                             stalls["utilisation_packing"], stalls["stall_share"])
        # Payload: everything returned (t + wait_s) between script entry and the end
        # of training on each rank, on that rank's own host clock, so a report
        # rebuilt after in-flight prefetch lands gives the same total.
        ended_at = [meta["ended_at"] for meta in metas]
        wire = _summarise_byte_records(
            record for record in _read_jsonl(sorted(directory.glob("bytes.rank*.jsonl")))
            if record["t"] + record["wait_s"] <= ended_at[record["rank"]]
        )
        whole_run_s = max(meta["whole_run_s"] for meta in metas)
        wire["bytes_per_s"] = wire["total_bytes"] / whole_run_s
        metrics = {
            "useful_tokens_per_s": window["tokens_nonpad"] / seconds,
            "useful_tokens_per_s_per_gpu": window["tokens_nonpad"] / seconds / world,
            "loss_tokens_per_s": {name: count / seconds for name, count in window["loss_tokens"].items()},
            "packing": stalls["utilisation_packing"],
            "stall_share": stalls["stall_share"],
            "mfu": mfu,
            "timing": {
                "startup_s": max(meta["startup_s"] for meta in metas),
                "first_step_loader_wait_s": max(records[0]["loader_wait_s"] for records in series),
                "warmup_s": max(records[warmup - 1]["elapsed_s"] for records in series) if warmup else 0.0,
                "measured_s": seconds,
                "whole_run_s": whole_run_s,
            },
            "window": window,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        problems.append(f"malformed input: {type(exc).__name__}: {exc}")
        return report
    report.update(status="complete", metrics=metrics, wire=wire)
    if measured < 200 or seconds < 60:
        report["warnings"].append("short window (<200 steps or <60 s): limited evidence, "
                                  "not a stable hardware comparison")
    return report


def _render_benchmark(report: dict) -> str:
    lines = [f"# Throughput benchmark (schema v{report['schema_version']}): {report['status'].upper()}"]
    if report["status"] != "complete":
        return "\n".join(lines + [f"  {problem}" for problem in report["diagnostics"]])
    metrics, wire = report["metrics"], report["wire"]
    timing, window = metrics["timing"], metrics["window"]
    steps = window["step_seconds"]
    gpus = ", ".join(sorted({meta["gpu"]["name"] for meta in report["ranks"].values()}))
    lines += [
        f"Useful tokens/s: {metrics['useful_tokens_per_s']:,.1f} "
        f"({metrics['useful_tokens_per_s_per_gpu']:,.1f}/GPU over {report['run']['world_size']} x {gpus})",
        f"Packing {metrics['packing']:.1%}; stall share {metrics['stall_share']:.1%}; "
        f"{window['rows_per_s']:,.1f} rows/s",
        f"Timing: startup {timing['startup_s']:.1f}s; first-step loader wait "
        f"{timing['first_step_loader_wait_s']:.1f}s; warmup {timing['warmup_s']:.1f}s; "
        f"measured {window['n_steps']} steps / {timing['measured_s']:.1f}s",
        f"Step seconds: mean {steps['mean']:.2f}  p50 {steps['p50']:.2f}  p95 {steps['p95']:.2f}  "
        f"p99 {steps['p99']:.2f}  slow(>{steps['slow_threshold_s']:.0f}s) {steps['slow_steps']}",
        f"Payload: {wire['total_mib']:,.1f} MiB, {wire['bytes_per_s'] / MIB:,.2f} MiB/s over the whole run",
        "Composition: " + ", ".join(f"{name} {share:.1%}" for name, share in window["composition"].items()),
    ]
    mfu = metrics["mfu"]
    if mfu["mfu"] is None:
        lines.append(f"MFU: unavailable ({mfu['unavailable_reason']})")
    else:
        busy = f"{mfu['mfu_busy']:.2%}" if mfu["mfu_busy"] is not None else "unavailable"
        lines.append(f"MFU estimate: {mfu['mfu']:.2%} = busy {busy} x non-stalled "
                     f"{1 - metrics['stall_share']:.3f} x packing {metrics['packing']:.3f}; "
                     f"{mfu['model_tflops_per_gpu']:.2f} TFLOP/s/GPU")
    hosts = {meta["host"]: [speed for speed in meta["nic_link_mbps"].values() if speed]
             for meta in report["ranks"].values()}
    for host, speeds in hosts.items():
        lines.append(f"Fastest NIC link [{host}]: "
                     + (f"{max(speeds) / 1000:g} Gbit/s" if speeds else "unknown"))
    return "\n".join(lines + [f"WARNING: {warning}" for warning in report["warnings"]])


def build_report(directory: Path, object_log: Path | None, arm: str, slow: float):
    if (directory / "run.json").exists() or any(directory.glob("rank*.json")):
        return build_benchmark_report(directory, slow)
    wire = summarise_bytes(directory)
    steps = summarise_steps(directory, slow)
    objects = summarise_objects(object_log)

    mib = wire["total_mib"] or 1.0
    loss_tokens = sum(steps.get("loss_tokens", {}).values())
    # §2: E_values counts OBSERVED target dimensions. A replica re-presents
    # the same values under a different factorisation, so the raw per-batch
    # count double-counts it and E_values would rise with ar_replicas — which
    # is precisely the gaming §2 says it must be immune to. Deflate by the
    # measured replay factor so only distinct observations are counted.
    primary_fraction = (
        objects["primary"] / objects["sequences"] if objects.get("sequences") else 1.0
    )
    report = {
        "arm": arm,
        "telemetry_dir": str(directory),
        "wire": wire,
        "steps": steps,
        "objects": objects,
        # E_values must NOT move under replay or per-band; E_AR should, and
        # the primary split is what shows how much of it is new information.
        "E_values_per_mib": steps.get("target_values", 0) * primary_fraction / mib,
        "E_values_presented_per_mib": steps.get("target_values", 0) / mib,
        "E_AR_per_mib": loss_tokens / mib,
        "E_AR_primary_per_mib": loss_tokens * primary_fraction / mib,
        "tib_downloaded": wire["total_bytes"] / TIB,
    }
    return report


def render(report: dict) -> str:
    if "schema_version" in report:
        return _render_benchmark(report)
    lines = [f"# ADR 0014 §3 benchmark — arm {report['arm']}", ""]
    wire = report["wire"]
    lines.append(f"Wire: {wire['total_mib']:.1f} MiB over "
                 f"{sum(s['fetches'] for s in wire['by_source'].values())} fetches")
    for source, stats in wire["by_source"].items():
        lines.append(
            f"  {source:<18} {stats['mib']:9.1f} MiB  {stats['share']:6.1%}  "
            f"{stats['fetches']:5d} fetches"
        )
    steps = report.get("steps") or {}
    if steps:
        timing = steps["step_seconds"]
        lines += [
            "",
            f"Steps: {steps['n_steps']}  mean {timing['mean']:.2f}s  "
            f"p50 {timing['p50']:.2f}s  p95 {timing['p95']:.2f}s  "
            f"p99 {timing['p99']:.2f}s  "
            f"slow(>{timing['slow_threshold_s']:.0f}s) {timing['slow_steps']}",
            "",
            f"Throughput: {steps['rows']:,} rows (galaxies) over "
            f"{timing['total_s']:.1f}s = {steps['rows_per_s']:,.1f} rows/s",
            "",
            "MFU (decomposed — never quote the headline alone, §11):",
        ]
        mfu = steps["mfu"]
        identity = (
            mfu["mfu_busy"]
            * (1 - mfu["stall_share"])
            * mfu["utilisation_packing"]
        )
        lines += [
            f"  MFU               {mfu['mfu']:.3%}",
            f"  = MFU_busy        {mfu['mfu_busy']:.3%}",
            f"  x (1 - stall)     {1 - mfu['stall_share']:.3f}   "
            f"(stall_share {mfu['stall_share']:.1%})",
            f"  x packing         {mfu['utilisation_packing']:.3f}",
            f"  = {identity:.3%}   (identity check)",
            f"  FLOPs/token       {mfu['flops_per_token']:.3e}   "
            f"(peak {mfu['peak_tflops_per_gpu']:.0f} TFLOP/s/GPU)",
        ]
        if steps.get("composition"):
            lines += ["", "Composition (share of loss-bearing tokens):"]
            for name, share in list(steps["composition"].items())[:10]:
                lines.append(f"  {name:<28} {share:6.2%}")
    objects = report.get("objects") or {}
    if objects:
        lines += [
            "",
            f"Sequences: {objects['sequences']} "
            f"({objects['primary']} primary + {objects['replay']} replay) "
            f"over {objects['distinct_base_objects']} distinct objects "
            f"= {objects['replay_factor']:.2f}x",
            f"Exactly-once audit: {objects['duplicate_lines']} duplicate lines "
            f"({'PASS' if not objects['duplicate_lines'] else 'FAIL'})",
        ]
    lines += [
        "",
        f"E_values  {report['E_values_per_mib']:,.0f} distinct target values / MiB"
        f"  ({report['E_values_presented_per_mib']:,.0f} presented)",
        f"E_AR      {report['E_AR_per_mib']:,.0f} loss-bearing tokens / MiB "
        f"({report['E_AR_primary_per_mib']:,.0f} primary)",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", required=True, help="$ASTROPT3_TELEMETRY_DIR")
    parser.add_argument("--object-log", default=None, help="object_id_log prefix")
    parser.add_argument("--arm", default="unnamed")
    parser.add_argument(
        "--slow-seconds",
        type=float,
        default=60.0,
        help="a step slower than this counts as a stall (§3 slow-step count)",
    )
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    report = build_report(
        Path(args.telemetry),
        Path(args.object_log) if args.object_log else None,
        args.arm,
        args.slow_seconds,
    )
    print(render(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 1 if report.get("status") == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
