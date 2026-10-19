"""ADR 0014 §3 benchmark report: join the byte log, step log and object audit.

Usage:
    python scripts/bench_report.py --telemetry <dir> [--object-log <prefix>]
                                   [--arm B0] [--out report.json]

Reads what the instruments wrote during a benchmark window:

- ``bytes.dp*.w*.jsonl``  — one line per ``_fetch_range`` payload (§3)
- ``steps.dp*.jsonl``     — one line per step, with the decomposed MFU (§2a)
- ``{object_log}.dp*``    — one line per emitted sequence (the no-replay audit)

A directory containing ``run.json`` instead selects the schema-v1 benchmark
report: all world ranks must complete the exact warmup and measurement window.
Token counters count presentations, not unique corpus coverage; payload bytes
cover the whole run (including prefetch), not measured-window consumption.

Legacy logs report the four metrics of §2 plus the timing distribution. Two rules
from the ADR are enforced here rather than left to the reader:

- MFU is only ever printed decomposed (§11 refuses the headline number);
- ``E_AR`` is broken into primary and replay tokens, because replay raises it
  without adding information and the split is what makes that visible.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

MIB = 1024 * 1024
TIB = 1024**4


def _read_jsonl(paths):
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; no numpy dependency for a reporting script."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


def _byte_paths(directory: Path) -> list[Path]:
    return sorted({
        path
        for pattern in ("bytes.dp*.w*.jsonl", "bytes.dp*.main.jsonl",
                        "bytes.rank*.w*.jsonl", "bytes.rank*.main.jsonl")
        for path in directory.glob(pattern)
    })


def summarise_bytes(directory: Path) -> dict:
    return _summarise_byte_records(_read_jsonl(_byte_paths(directory)))


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
    rank_files = sorted({path for pattern in ("steps.dp*.jsonl", "steps.rank*.jsonl")
                         for path in directory.glob(pattern)})
    series = [list(_read_jsonl([path])) for path in rank_files]
    records = [record for entries in series for record in entries]
    if not records:
        return {}
    n_ranks = max(record.get("world_size", 0) for record in records) or len(rank_files)
    logical = [record for record in records if record.get("tp_rank", 0) == 0]
    dp_ranks = (
        len({record["dp_rank"] for record in logical})
        if all("dp_rank" in record for record in logical)
        else sum(any(record.get("tp_rank", 0) == 0 for record in entries) for entries in series)
    )

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


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _benchmark_records(path: Path, diagnostics: list[str], *, jsonl=False):
    """Retain valid records, but never turn a damaged log into a successful run."""
    try:
        text = path.read_text()
    except (OSError, UnicodeError) as exc:
        diagnostics.append(f"{path.name}: {exc}")
        return []
    records = []
    for line, content in enumerate(text.splitlines() if jsonl else [text], 1):
        if jsonl and not content.strip():
            continue
        label = f"{path.name}:{line}"
        try:
            record = json.loads(content, parse_constant=_reject_constant)
            if not isinstance(record, dict):
                raise ValueError("expected a JSON object")
            records.append((label, record))
        except ValueError as exc:
            diagnostics.append(f"{label}: {exc}")
    return records


def _number(record, key, *, integer=False, positive=False):
    value = record.get(key)
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{key} must be a {'positive' if positive else 'nonnegative'} "
                         f"{'integer' if integer else 'finite number'}")
    return value


def _distribution(values, slow):
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "total_s": sum(values),
        "slow_steps": sum(value > slow for value in values),
        "slow_threshold_s": slow,
    }


def build_benchmark_report(directory: Path, slow: float = 60.0) -> dict:
    diagnostics: list[str] = []
    report = {
        "schema_version": 1,
        "status": "partial",
        "telemetry_dir": str(directory),
        "run": {},
        "ranks": {},
        "diagnostics": diagnostics,
        "warnings": [],
        "metrics": {},
        "wire": {},
        "count_semantics": "Token, target-value and row counts are presentations, "
                           "not unique-object counts or corpus coverage. Useful tokens "
                           "are nonpadding tokens (including structural tokens).",
    }
    runs = _benchmark_records(directory / "run.json", diagnostics)
    if not runs:
        return report
    run = report["run"] = runs[0][1]
    try:
        if _number(run, "schema_version", integer=True) != 1:
            raise ValueError("unsupported schema_version (expected 1)")
        world = _number(run, "world_size", integer=True, positive=True)
        warmup = _number(run, "warmup_steps", integer=True)
        measured = _number(run, "steps", integer=True, positive=True)
        topology = run.get("parallelism")
        if not isinstance(topology, dict):
            raise ValueError("parallelism must be an object")
        for key in ("dp", "tp", "pp", "context_parallel_size", "expert_parallel_size"):
            _number(topology, key, integer=True, positive=True)
        if any(topology[key] != 1 for key in ("pp", "context_parallel_size", "expert_parallel_size")):
            raise ValueError("schema v1 requires pp=context_parallel_size=expert_parallel_size=1")
        if topology["dp"] * topology["tp"] != world:
            raise ValueError("world_size must equal dp * tp")
        for key in ("baseline_config", "overrides"):
            if not isinstance(run.get(key), dict):
                raise ValueError(f"{key} must be an object")
    except ValueError as exc:
        diagnostics.append(f"run.json: {exc}")
        return report

    for pattern, expected in (
        ("rank*.json", {f"rank{rank}.json" for rank in range(world)}),
        ("steps.rank*.jsonl", {f"steps.rank{rank}.jsonl" for rank in range(world)}),
    ):
        actual = {path.name for path in directory.glob(pattern)}
        if actual - expected:
            diagnostics.append(f"Unexpected files: {sorted(actual - expected)}")
    ranks, all_steps, identities, starts = {}, {}, set(), set()
    expected_payloads, expected_probes = {}, set()
    report["observed_step_records"] = {}
    for rank in range(world):
        metadata = _benchmark_records(directory / f"rank{rank}.json", diagnostics)
        records = _benchmark_records(directory / f"steps.rank{rank}.jsonl", diagnostics, jsonl=True)
        report["observed_step_records"][str(rank)] = len(records)
        if not metadata:
            continue
        label, meta = metadata[0]
        report["ranks"][str(rank)] = meta
        try:
            if _number(meta, "rank", integer=True) != rank:
                raise ValueError(f"rank identity does not match filename (expected {rank})")
            dp = _number(meta, "dp_rank", integer=True)
            tp = _number(meta, "tp_rank", integer=True)
            if dp >= topology["dp"] or tp >= topology["tp"]:
                raise ValueError("dp_rank/tp_rank outside configured topology")
            if (dp, tp) in identities:
                raise ValueError(f"duplicate logical rank identity {(dp, tp)}")
            identities.add((dp, tp))
            if meta.get("status") != "complete":
                raise ValueError(f"rank status is {meta.get('status')!r}, not 'complete'")
            start = _number(meta, "start_step", integer=True)
            starts.add(start)
            for key in ("startup_s", "first_batch_s", "whole_run_s"):
                _number(meta, key, positive=key == "whole_run_s")
            workers = _number(meta, "num_loading_workers", integer=True)
            ranks[rank] = meta
        except ValueError as exc:
            diagnostics.append(f"{label}: {exc}")
            continue
        for worker in range(workers) if workers else [None]:
            consumer = "main" if worker is None else f"w{worker}"
            stem = f"rank{rank}.{consumer}"
            expected_probes.add(f"probe.{stem}.json")
            expected_payloads[f"bytes.{stem}.jsonl"] = (rank, dp, worker)
            if not (directory / f"bytes.{stem}.jsonl").is_file():
                diagnostics.append(f"bytes.{stem}.jsonl: missing payload log (empty is valid)")
            for label, probe in _benchmark_records(directory / f"probe.{stem}.json", diagnostics):
                try:
                    for key, expected in (("rank", rank), ("dp_rank", dp)):
                        if _number(probe, key, integer=True) != expected:
                            raise ValueError(f"{key} identity mismatch (expected {expected})")
                    if "worker" not in probe or probe["worker"] != worker:
                        raise ValueError(f"worker identity mismatch (expected {worker})")
                    if worker is not None:
                        _number(probe, "worker", integer=True)
                    _number(probe, "pid", integer=True, positive=True)
                    instruments = probe.get("instruments")
                    if not isinstance(instruments, list) or not all(
                        name in instruments for name in ("_fetch_range", "stream.read")
                    ):
                        raise ValueError("instruments must include _fetch_range and stream.read")
                except ValueError as exc:
                    diagnostics.append(f"{label}: {exc}")
        steps = all_steps[rank] = {}
        previous_elapsed, previous_step = 0.0, start
        for label, record in records:
            try:
                for key, expected in (("rank", rank), ("dp_rank", dp), ("tp_rank", tp), ("world_size", world)):
                    if _number(record, key, integer=True) != expected:
                        raise ValueError(f"{key} identity mismatch (expected {expected})")
                step = _number(record, "step", integer=True)
                if step in steps:
                    raise ValueError(f"duplicate step {step}")
                elapsed = _number(record, "elapsed_s", positive=True)
                _number(record, "step_seconds", positive=True)
                _number(record, "loader_wait_s")
                if elapsed <= previous_elapsed:
                    raise ValueError("elapsed_s must increase in step-log order")
                if step <= previous_step:
                    raise ValueError("steps must increase in step-log order")
                for key in ("tokens_nonpad", "tokens_total", "rows", "micro_batches"):
                    _number(record, key, integer=True, positive=key == "micro_batches")
                if record["tokens_nonpad"] > record["tokens_total"]:
                    raise ValueError("tokens_nonpad exceeds tokens_total")
                for key in ("loss_tokens", "target_values"):
                    if not isinstance(record.get(key), dict):
                        raise ValueError(f"{key} must be an object of modality counts")
                    for modality in record[key]:
                        _number(record[key], modality, integer=True)
                steps[step] = record
                previous_elapsed, previous_step = elapsed, step
            except ValueError as exc:
                diagnostics.append(f"{label}: {exc}")
        expected_steps = set(range(start + 1, start + warmup + measured + 1))
        if set(steps) != expected_steps:
            diagnostics.append(f"rank {rank}: missing steps {sorted(expected_steps - steps.keys())}; "
                               f"unexpected steps {sorted(steps.keys() - expected_steps)}")
    expected_identities = {(dp, tp) for dp in range(topology["dp"]) for tp in range(topology["tp"])}
    if identities != expected_identities:
        diagnostics.append("logical dp_rank/tp_rank identities do not cover the configured topology")
    if len(starts) > 1:
        diagnostics.append("start_step differs across ranks")

    for pattern, expected in (("probe.rank*.json", expected_probes),
                              ("bytes.rank*.jsonl", set(expected_payloads))):
        unexpected = {path.name for path in directory.glob(pattern)} - expected
        if unexpected:
            diagnostics.append(f"Unexpected instrumentation files: {sorted(unexpected)}")
    payloads = []
    for path in _byte_paths(directory):
        for label, record in _benchmark_records(path, diagnostics, jsonl=True):
            try:
                if not isinstance(record.get("source"), str) or not record["source"]:
                    raise ValueError("source must be a nonempty string")
                _number(record, "bytes", integer=True)
                _number(record, "t")
                _number(record, "wait_s")
                if path.name in expected_payloads:
                    for key, expected in zip(("rank", "dp_rank", "worker"), expected_payloads[path.name], strict=True):
                        if key in record and (record[key] != expected or type(record[key]) is not type(expected)):
                            raise ValueError(f"{key} identity mismatch (expected {expected})")
                payloads.append(record)
            except ValueError as exc:
                diagnostics.append(f"{label}: {exc}")
    wire = report["wire"] = _summarise_byte_records(payloads)
    wire["whole_run_bytes_per_s"] = None
    wire["interval"] = (
        "All returned payload bytes from script entry through fetcher shutdown (loader close) / "
        "max(rank whole_run_s), using local durations, not cross-node clocks. "
        "Includes prefetch and repeated returns; not measured-window consumption or link capacity."
    )
    if diagnostics:
        return report

    start = next(iter(starts))
    step_ids = range(start + warmup + 1, start + warmup + measured + 1)
    warmup_times = {
        rank: records[start + warmup]["elapsed_s"] if warmup else 0.0
        for rank, records in all_steps.items()
    }
    elapsed = max(records[start + warmup + measured]["elapsed_s"] - warmup_times[rank]
                  for rank, records in all_steps.items())
    durations = [max(records[step]["step_seconds"] for records in all_steps.values()) for step in step_ids]
    waits = [max(records[step]["loader_wait_s"] for records in all_steps.values()) for step in step_ids]
    logical = [records[step] for rank, records in all_steps.items()
               if ranks[rank]["tp_rank"] == 0 for step in step_ids]
    counts: dict = {key: sum(record[key] for record in logical)
                    for key in ("tokens_nonpad", "tokens_total", "rows", "micro_batches")}
    loss_tokens, target_values = Counter(), Counter()
    for record in logical:
        loss_tokens.update(record["loss_tokens"])
        target_values.update(record["target_values"])
    counts.update(loss_tokens=dict(loss_tokens), target_values=dict(target_values))
    loss_total = sum(loss_tokens.values())
    timing = {
        "startup_s": max(meta["startup_s"] for meta in ranks.values()),
        "first_batch_s": max(meta["first_batch_s"] for meta in ranks.values()),
        "warmup_s": max(warmup_times.values()),
        "measured_s": elapsed,
        "whole_run_s": max(meta["whole_run_s"] for meta in ranks.values()),
        "warmup_steps": warmup,
        "measured_steps": measured,
        "interval": "max across all world ranks of last.elapsed_s - warmup_end.elapsed_s "
                    "(warmup_end=0 with no warmup); absolute steps are offset by start_step.",
    }
    report["metrics"] = {
        "timing": timing,
        "counts": counts,
        "throughput": {
            "useful_tokens_per_s": counts["tokens_nonpad"] / elapsed,
            "useful_tokens_per_s_per_gpu": counts["tokens_nonpad"] / elapsed / world,
            "loss_tokens_per_s": loss_total / elapsed,
            "rows_per_s": counts["rows"] / elapsed,
        },
        "modalities": {
            name: {
                "loss_tokens": loss_tokens[name],
                "loss_tokens_per_s": loss_tokens[name] / elapsed,
                "share": loss_tokens[name] / loss_total if loss_total else 0.0,
                "target_values": target_values[name],
                "target_values_per_s": target_values[name] / elapsed,
            }
            for name in sorted(loss_tokens.keys() | target_values.keys())
        },
        "packing": counts["tokens_nonpad"] / counts["tokens_total"] if counts["tokens_total"] else 0.0,
        "step_seconds": _distribution(durations, slow),
        "waits": {
            "loader_wait_s": sum(waits),
            "stall_share": sum(waits) / sum(durations),
            "interval": "sum(per-step max-rank loader_wait_s) / sum(per-step max-rank step_seconds); "
                        "time-weighted, not the mean of per-step ratios or a critical-path decomposition.",
        },
    }
    wire["whole_run_bytes_per_s"] = wire["total_bytes"] / timing["whole_run_s"]
    if measured < 200 or elapsed < 60:
        report["warnings"].append(
            "Short measured window (<200 steps OR <60s heuristic); "
            "startup, stalls and modality mix may dominate."
        )
    report["status"] = "complete"
    return report


def _render_benchmark(report: dict) -> str:
    lines = [f"# Benchmark v{report['schema_version']} — {report['status'].upper()}"]
    if report["status"] != "complete":
        lines.append("No headline throughput: incomplete or invalid benchmark input.")
        lines.extend(f"  ERROR: {message}" for message in report["diagnostics"])
        return "\n".join(lines)
    metrics, wire = report["metrics"], report["wire"]
    rate, timing, steps = metrics["throughput"], metrics["timing"], metrics["step_seconds"]
    lines += [
        f"Useful tokens/s: {rate['useful_tokens_per_s']:,.1f} global; "
        f"{rate['useful_tokens_per_s_per_gpu']:,.1f}/GPU ({report['run']['world_size']} GPUs)",
        f"Packing: {metrics['packing']:.1%}; row presentations/s: {rate['rows_per_s']:,.1f}",
        f"Timing (max across ranks): startup {timing['startup_s']:.2f}s; "
        f"first batch {timing['first_batch_s']:.2f}s; warmup {timing['warmup_steps']} steps / "
        f"{timing['warmup_s']:.2f}s; measured {timing['measured_steps']} steps / {timing['measured_s']:.2f}s",
        f"Steps (max rank per step): mean {steps['mean']:.2f}s; p50 {steps['p50']:.2f}s; "
        f"p95 {steps['p95']:.2f}s; p99 {steps['p99']:.2f}s; max {steps['max']:.2f}s; "
        f"slow(>{steps['slow_threshold_s']:g}s) {steps['slow_steps']}",
        f"Loader waits (per-step max ranks): {metrics['waits']['loader_wait_s']:.2f}s; "
        f"time-weighted stall share {metrics['waits']['stall_share']:.1%}",
        "Modalities (loss-token presentations/s; share of loss tokens; target values/s):",
    ]
    for name, stats in metrics["modalities"].items():
        lines.append(f"  {name}: {stats['loss_tokens_per_s']:,.1f}/s; {stats['share']:.1%}; "
                     f"{stats['target_values_per_s']:,.1f} values/s")
    lines += [
        f"Payload: {wire['total_bytes']:,} bytes ({wire['total_mib']:.2f} MiB); "
        f"{wire['whole_run_bytes_per_s'] / MIB:.2f} MiB/s over whole run {timing['whole_run_s']:.2f}s; "
        f"summed fetch wait {wire['fetch_wait_s']:.2f}s",
        wire["interval"],
        report["count_semantics"],
    ]
    lines.extend(f"WARNING: {message}" for message in report["warnings"])
    return "\n".join(lines)


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

    directory = Path(args.telemetry)
    if args.out:
        raw_paths = {p.resolve() for pattern in ("run.json", "rank*.json", "probe.*.json", "*.jsonl")
                     for p in directory.glob(pattern)}
        if Path(args.out).resolve() in raw_paths:
            parser.error("--out must not overwrite raw benchmark inputs")
    report = build_report(
        directory,
        Path(args.object_log) if args.object_log else None,
        args.arm,
        args.slow_seconds,
    )
    print(render(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))
    return 1 if report.get("status") == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
