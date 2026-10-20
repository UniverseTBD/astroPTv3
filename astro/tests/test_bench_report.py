"""ADR 0014 §3 report aggregation — the arithmetic that decides the arms."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_report  # pyright: ignore[reportMissingImports] -- sys.path above


def _write(directory: Path, ranks=2, steps=None):
    steps = steps or [
        # (seconds, wait) — bimodal on purpose: this corpus stalls rarely and
        # hugely, which is what a mean of ratios gets wrong
        (1.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (97.0, 96.0),
    ]
    for rank in range(ranks):
        lines = []
        for i, (seconds, wait) in enumerate(steps):
            lines.append(
                json.dumps(
                    {
                        "step": i,
                        "step_seconds": seconds,
                        "loader_wait_s": wait,
                        # global estimate: every rank writes the SAME number
                        "model_flops": 1e14,
                        "tokens_total": 1000,
                        "tokens_nonpad": 900,
                        "rows": 50,
                        "utilisation_packing": 0.9,
                        "loss_tokens": {"images": 100},
                        "target_values": {"images": 19200},
                        "peak_tflops_per_gpu": 312.0,
                    }
                )
            )
        (directory / f"steps.dp{rank}.jsonl").write_text("\n".join(lines) + "\n")


def test_ranks_are_grouped_not_concatenated(tmp_path):
    """Two DP ranks run the SAME steps concurrently, not twice as many."""
    _write(tmp_path, ranks=2)
    steps = bench_report.summarise_steps(tmp_path, slow_threshold=60.0)
    assert steps["n_steps"] == 4
    assert steps["dp_ranks"] == 2
    # token counts sum across ranks (each packs its own data)
    assert steps["tokens_total"] == 4 * 1000 * 2


def test_rows_per_s_is_end_to_end_wall_clock_throughput(tmp_path):
    """rows/s uses total wall time, not loader-wait time (that's telemetry's own metric)."""
    _write(tmp_path, ranks=2)
    steps = bench_report.summarise_steps(tmp_path, slow_threshold=60.0)
    # rows sum across ranks (each packs its own data), like token counts
    assert steps["rows"] == 4 * 50 * 2
    total_wall = 1.0 + 1.0 + 1.0 + 97.0
    assert steps["rows_per_s"] == pytest.approx(steps["rows"] / total_wall)


def test_stall_share_is_time_weighted_not_a_mean_of_ratios(tmp_path):
    """The bug this guards: 3 free steps + 1 huge stall.

    Mean of per-step ratios = (0 + 0 + 0 + 0.99)/4 = 25%.
    The share actually paid = 96 / 100 = 96%. Only the second is the truth.
    """
    _write(tmp_path)
    mfu = bench_report.summarise_steps(tmp_path, slow_threshold=60.0)["mfu"]
    assert mfu["stall_share"] == pytest.approx(96.0 / 100.0)
    assert mfu["stall_share"] > 0.9  # never the 25% a ratio mean would give


def test_the_mfu_decomposition_identity_holds(tmp_path):
    """MFU = MFU_busy x (1 - stall) x packing, exactly (ADR 0014 §2a).

    It only holds if MFU_busy is deflated by packing: `model_flops` already
    refuses padding credit, so leaving it in double-counts the factor.
    """
    _write(tmp_path)
    mfu = bench_report.summarise_steps(tmp_path, slow_threshold=60.0)["mfu"]
    identity = mfu["mfu_busy"] * (1 - mfu["stall_share"]) * mfu["utilisation_packing"]
    assert identity == pytest.approx(mfu["mfu"], rel=1e-9)
    assert mfu["utilisation_packing"] == pytest.approx(0.9)


def test_flops_are_taken_once_across_ranks_not_summed(tmp_path):
    """`model_flops` is already global; summing it would double the MFU."""
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    _write(one, ranks=1)
    _write(two, ranks=2)
    # same flops, twice the GPUs in the denominator -> half the MFU
    a = bench_report.summarise_steps(one, slow_threshold=60.0)["mfu"]["mfu"]
    b = bench_report.summarise_steps(two, slow_threshold=60.0)["mfu"]["mfu"]
    assert b == pytest.approx(a / 2)


def test_e_values_does_not_move_under_replay(tmp_path):
    """§2: replay re-presents the same observations; E_values must be immune.

    E_AR is expected to double — that split is the point.
    """
    _write(tmp_path)
    (tmp_path / "bytes.dp0.w0.jsonl").write_text(
        json.dumps(
            {"t": 0, "wait_s": 0.0, "dp": 0, "worker": 0, "source": "legacy",
             "path": "p", "start": 0, "end": 1, "bytes": bench_report.MIB}
        )
        + "\n"
    )
    log = tmp_path / "objects.log"
    # 2 base objects, each with one replica: replay factor 2.0
    (tmp_path / "objects.log.dp0").write_text("a\nb\na#1\nb#1\n")

    plain = bench_report.build_report(tmp_path, None, "no-audit", 60.0)
    replayed = bench_report.build_report(tmp_path, log, "replay2", 60.0)

    # presented values are identical; distinct values halve once the audit
    # reveals that half the sequences were replicas
    assert replayed["E_values_presented_per_mib"] == plain["E_values_presented_per_mib"]
    assert replayed["E_values_per_mib"] == pytest.approx(
        plain["E_values_per_mib"] / 2
    )
    assert replayed["E_AR_per_mib"] == pytest.approx(2 * replayed["E_AR_primary_per_mib"])
    assert replayed["objects"]["duplicate_lines"] == 0


@pytest.mark.parametrize("tp", [1, 2])
def test_world_rank_logs_count_tp_replicas_once_and_divide_by_world_size(tmp_path, tp):
    _write(tmp_path, ranks=2 * tp)
    for rank in range(2 * tp):
        old = tmp_path / f"steps.dp{rank}.jsonl"
        records = [json.loads(line) for line in old.read_text().splitlines()]
        for record in records:
            record.update(rank=rank, dp_rank=rank // tp, tp_rank=rank % tp, world_size=2 * tp)
            if rank % tp:
                record["tokens_nonpad"] *= 1000
                record["rows"] *= 1000
        _jsonl(tmp_path / f"steps.rank{rank}.jsonl", records)
        old.unlink()
    report = _report(tmp_path)
    assert "schema_version" not in report
    steps = report["steps"]
    assert steps["n_steps"] == 4
    assert steps["dp_ranks"] == 2
    assert steps["world_size"] == 2 * tp
    assert steps["tokens_nonpad"] == 7200
    assert steps["rows"] == 400
    assert steps["mfu"]["mfu"] == pytest.approx(4e14 / (100 * 312e12 * 2 * tp))


def _json(path, value):
    path.write_text(json.dumps(value))


def _jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _benchmark(directory, *, dp=1, tp=1, warmup=1, start=12, workers=0, durations=None):
    durations = durations or [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (97.0, 96.0)]
    world = dp * tp
    _json(directory / "run.json", {
        "schema_version": 1, "world_size": world, "parallelism": {"dp": dp, "tp": tp},
        "warmup_steps": warmup, "steps": len(durations),
        "baseline_config": {"model": "70M"}, "overrides": {},
    })
    for rank in range(world):
        dp_rank, tp_rank = divmod(rank, tp)
        _json(directory / f"rank{rank}.json", {
            "rank": rank, "dp_rank": dp_rank, "tp_rank": tp_rank, "status": "complete",
            "start_step": start, "startup_s": 5.0 + rank, "ended_at": 2000.0,
            "whole_run_s": 10.0 + 100 * warmup + sum(d for d, _ in durations) + rank,
            "num_loading_workers": workers,
        })
        for consumer in [f"w{worker}" for worker in range(workers)] or ["main"]:
            (directory / f"bytes.rank{rank}.{consumer}.jsonl").touch()
        records, elapsed = [], 0.0
        for index, (seconds, wait) in enumerate([(100.0, 90.0)] * warmup + durations, 1):
            elapsed += seconds
            # Warmup and TP-replica counters must never enter useful throughput.
            multiplier = 1000 if index <= warmup or tp_rank else 1
            records.append({
                "rank": rank, "dp_rank": dp_rank, "tp_rank": tp_rank,
                "world_size": world, "step": start + index,
                "elapsed_s": elapsed, "step_seconds": seconds, "loader_wait_s": wait,
                "tokens_nonpad": 900 * multiplier, "tokens_total": 1000 * multiplier,
                "rows": 10 * multiplier, "micro_batches": 2,
                "loss_tokens": {"image": 80 * multiplier, "spectrum": 20 * multiplier},
                "target_values": {"image": 15360 * multiplier, "spectrum": 5120 * multiplier},
            })
        _jsonl(directory / f"steps.rank{rank}.jsonl", records)


def _report(directory):
    return bench_report.build_report(directory, None, "ignored-for-v1", 60.0)


def test_benchmark_excludes_warmup_and_offsets_resumed_steps(tmp_path):
    _benchmark(tmp_path, warmup=2, start=100)
    report = _report(tmp_path)
    assert report["status"] == "complete", report["diagnostics"]
    assert report["run"]["baseline_config"] == {"model": "70M"}
    metrics = report["metrics"]
    assert metrics["timing"]["warmup_s"] == 200
    assert metrics["timing"]["first_step_loader_wait_s"] == 90
    assert metrics["timing"]["measured_s"] == 100
    assert metrics["window"]["tokens_nonpad"] == 3600
    assert metrics["window"]["rows"] == 40
    assert metrics["useful_tokens_per_s"] == 36
    assert metrics["packing"] == 0.9
    assert metrics["loss_tokens_per_s"]["image"] == 3.2
    assert metrics["window"]["composition"]["image"] == 0.8
    text = bench_report.render(report)
    for phrase in ("Useful tokens/s", "first-step loader wait", "p99", "stall share", "Payload"):
        assert phrase in text
    assert "short window" in text


def test_benchmark_rates_and_stalls_are_weighted(tmp_path):
    _benchmark(tmp_path)
    metrics = _report(tmp_path)["metrics"]
    assert metrics["useful_tokens_per_s"] == pytest.approx(3600 / 100)
    assert metrics["stall_share"] == pytest.approx(96 / 100)
    assert metrics["window"]["step_seconds"]["mean"] == 25
    assert metrics["window"]["step_seconds"]["p95"] == 97
    assert metrics["window"]["step_seconds"]["slow_steps"] == 1


def test_dp2_tp2_counts_only_logical_data_but_times_all_ranks(tmp_path):
    _benchmark(tmp_path, dp=2, tp=2)
    path = tmp_path / "steps.rank3.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]["step_seconds"] += 20  # a slow TP replica holds the whole step
    _jsonl(path, records)
    report = _report(tmp_path)
    assert report["status"] == "complete", report["diagnostics"]
    metrics = report["metrics"]
    assert metrics["timing"]["measured_s"] == 120
    assert metrics["window"]["tokens_nonpad"] == 7200
    assert metrics["window"]["loss_tokens"] == {"image": 640, "spectrum": 160}
    assert metrics["useful_tokens_per_s"] == 60
    assert metrics["useful_tokens_per_s_per_gpu"] == 15


@pytest.mark.parametrize("workers", [0, 2])
def test_payload_counts_every_return_until_training_ends(tmp_path, workers):
    _benchmark(tmp_path, dp=2, workers=workers)
    payload = {"source": "legacy", "bytes": bench_report.MIB, "t": 1234.0, "wait_s": 1.5}
    consumer = "w1" if workers else "main"
    # The same range fetched twice costs bytes twice.
    _jsonl(tmp_path / f"bytes.rank0.{consumer}.jsonl", [{**payload, "rank": 0}] * 2)
    late = tmp_path / f"bytes.rank1.{consumer}.jsonl"
    # Started before training ended (2000) but returned after it: excluded.
    _jsonl(late, [{**payload, "rank": 1}, {**payload, "rank": 1, "t": 1999.0}])
    with open(late, "a") as log:
        log.write('{"source":')  # a worker still appending when the report runs
    report = _report(tmp_path)
    assert report["status"] == "complete", report["diagnostics"]
    assert report["wire"]["total_bytes"] == 3 * bench_report.MIB
    assert report["wire"]["by_source"]["legacy"]["fetches"] == 3
    assert report["wire"]["fetch_wait_s"] == 4.5
    assert report["wire"]["bytes_per_s"] == pytest.approx(3 * bench_report.MIB / 211)


@pytest.mark.parametrize("missing", ["run.json", "rank1.json", "steps.rank1.jsonl", "bytes.rank1.main.jsonl"])
def test_missing_input_is_partial(tmp_path, missing):
    _benchmark(tmp_path, dp=2)
    (tmp_path / missing).unlink()
    report = _report(tmp_path)
    assert report["status"] == "partial"
    assert "metrics" not in report
    assert any(missing in message for message in report["diagnostics"])


@pytest.mark.parametrize("status", ["failed", "running", None])
def test_noncomplete_rank_is_partial(tmp_path, status):
    _benchmark(tmp_path)
    path = tmp_path / "rank0.json"
    _json(path, {**json.loads(path.read_text()), "status": status})
    report = _report(tmp_path)
    assert report["status"] == "partial"
    assert any("status" in message for message in report["diagnostics"])


@pytest.mark.parametrize("problem", ["missing_warmup", "missing_measured", "duplicate", "extra", "torn", "corrupt"])
def test_invalid_step_series_is_partial(tmp_path, problem):
    _benchmark(tmp_path)
    path = tmp_path / "steps.rank0.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if problem == "missing_warmup":
        records.pop(0)
    elif problem == "missing_measured":
        records.pop()
    elif problem == "duplicate":
        records.append(records[-1])
    elif problem == "extra":
        records.append({**records[-1], "step": records[-1]["step"] + 1})
    text = "".join(json.dumps(record) + "\n" for record in records)
    if problem == "torn":  # crashed while writing the final step
        text = text[:-20]
    elif problem == "corrupt":
        text = '{"step":\n' + text
    path.write_text(text)
    report = _report(tmp_path)
    assert report["status"] == "partial"
    assert "metrics" not in report
    assert "PARTIAL" in bench_report.render(report)
    assert path.read_text() == text


@pytest.mark.parametrize("steps,seconds,warn", [(240, 0.25, False), (199, 1, True), (200, 0.1, True)])
def test_short_window_warning_uses_either_threshold(tmp_path, steps, seconds, warn):
    _benchmark(tmp_path, warmup=0, durations=[(seconds, 0)] * steps)
    report = _report(tmp_path)
    assert bool(report["warnings"]) is warn


@pytest.mark.parametrize("complete", [True, False])
def test_cli_exit_status_and_json(tmp_path, complete):
    _benchmark(tmp_path)
    if not complete:
        (tmp_path / "steps.rank0.jsonl").write_text('{"step":')
    out = tmp_path / "report.json"
    result = subprocess.run([
        sys.executable, str(bench_report.__file__), "--telemetry", str(tmp_path), "--out", str(out),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == (0 if complete else 1), result.stderr
    assert json.loads(out.read_text())["status"] == ("complete" if complete else "partial")


def _hardware(directory, peaks):
    names = {835.5: "NVIDIA H200 NVL", 989.5: "NVIDIA H200 SXM", 312.0: "NVIDIA A100-SXM4-80GB"}
    for rank, peak in enumerate(peaks):
        path = directory / f"rank{rank}.json"
        meta = json.loads(path.read_text())
        meta.update(
            host="fixture-host",
            gpu={"name": names.get(peak, "unknown GPU"), "peak_tflops_per_gpu": peak},
            flops_estimate={"dtype": "torch.bfloat16", "backbone_flops_per_token": 1e9,
                            "modality_forward_flops_per_token": {"image": 2e6, "spectrum": 4e6}},
            network={"max_active_link_speed_mbps": 10000, "fastest_active_interfaces": ["eth0"]},
        )
        _json(path, meta)


def test_benchmark_mfu_prices_measured_tokens_and_all_gpu_peaks(tmp_path):
    _benchmark(tmp_path, dp=2, tp=2, warmup=2)
    peaks = [835.5, 989.5, 312.0, 312.0]
    _hardware(tmp_path, peaks)
    path = tmp_path / "steps.rank3.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]["step_seconds"] += 20  # slow TP replica counts in elapsed, not FLOPs
    for record in records:
        record["model_flops"] = 1e30  # legacy estimates must not enter schema-v1 MFU
    _jsonl(path, records)
    report = _report(tmp_path)
    assert report["status"] == "complete", report["diagnostics"]
    mfu = report["metrics"]["mfu"]
    expected_flops = 4 * 2 * (900 * 1e9 + 3 * (80 * 2e6 + 20 * 4e6))
    assert mfu["model_flops"] == expected_flops
    assert mfu["peak_tflops_by_rank"] == peaks
    assert mfu["model_tflops_per_gpu"] == pytest.approx(expected_flops / (120 * 4 * 1e12))
    assert mfu["mfu"] == pytest.approx(expected_flops / (120 * sum(peaks) * 1e12))
    assert mfu["stall_share"] == pytest.approx(96 / 120)
    assert mfu["mfu_busy"] * (1 - mfu["stall_share"]) * mfu["utilisation_packing"] == pytest.approx(mfu["mfu"])
    text = bench_report.render(report)
    assert "MFU estimate:" in text and "835.5 TFLOP/s" in text
    assert "Estimated upper bound" in text
    assert text.count("Fastest active physical NIC [fixture-host]: 10 Gbit/s (eth0)") == 1
    assert "not internet/downlink capacity" in text


@pytest.mark.parametrize("problem", ["old_metadata", "unknown_gpu", "wrong_dtype"])
def test_unavailable_mfu_never_becomes_zero_or_breaks_throughput(tmp_path, problem):
    _benchmark(tmp_path, dp=2)
    _hardware(tmp_path, [835.5, None if problem == "unknown_gpu" else 835.5])
    path = tmp_path / "rank1.json"
    meta = json.loads(path.read_text())
    if problem == "old_metadata":
        meta.pop("flops_estimate")
        meta.pop("network")
        meta["host"] = "old-host"
    elif problem == "wrong_dtype":
        meta["flops_estimate"]["dtype"] = "torch.float32"
    _json(path, meta)
    report = _report(tmp_path)
    assert report["status"] == "complete", report["diagnostics"]
    assert report["metrics"]["useful_tokens_per_s"] == 72
    assert report["metrics"]["mfu"]["mfu"] is None
    assert "MFU: unavailable" in bench_report.render(report)
    if problem == "old_metadata":
        assert "NIC [old-host]: unavailable" in bench_report.render(report)


@pytest.mark.parametrize("field,value", [
    ("peak", 0), ("peak", -1), ("peak", float("nan")), ("peak", True),
    ("backbone", float("inf")), ("modality", -1),
])
def test_invalid_flop_or_peak_metadata_is_partial(tmp_path, field, value):
    _benchmark(tmp_path)
    _hardware(tmp_path, [835.5])
    path = tmp_path / "rank0.json"
    meta = json.loads(path.read_text())
    if field == "peak":
        meta["gpu"]["peak_tflops_per_gpu"] = value
    elif field == "backbone":
        meta["flops_estimate"]["backbone_flops_per_token"] = value
    else:
        meta["flops_estimate"]["modality_forward_flops_per_token"]["image"] = value
    _json(path, meta)
    report = _report(tmp_path)
    assert report["status"] == "partial"
    assert "metrics" not in report
    assert "finite" in report["diagnostics"][0]


def test_network_link_speed_is_kept_per_host_not_summed_across_ranks(tmp_path):
    _benchmark(tmp_path, dp=2)
    _hardware(tmp_path, [835.5, 835.5])
    path = tmp_path / "rank1.json"
    meta = json.loads(path.read_text())
    meta.update(host="second-host", network={
        "max_active_link_speed_mbps": None, "fastest_active_interfaces": []})
    _json(path, meta)
    report = _report(tmp_path)
    text = bench_report.render(report)
    assert "NIC [fixture-host]: 10 Gbit/s (eth0)" in text
    assert "NIC [second-host]: unavailable" in text
    assert report["ranks"]["0"]["network"]["max_active_link_speed_mbps"] == 10000
    assert report["wire"]["bytes_per_s"] == 0  # not the NIC speed
