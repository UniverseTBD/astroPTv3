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
def test_legacy_report_reads_migrated_world_rank_logs_without_run_manifest(tmp_path, tp):
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
    assert steps["tokens_total"] == 8000
    assert steps["rows"] == 400
    assert steps["loss_tokens"] == {"images": 800}
    assert steps["target_values"] == 153600
    assert steps["mfu"]["mfu"] == pytest.approx(4e14 / (100 * 312e12 * 2 * tp))


def test_legacy_report_accepts_mixed_filenames_and_no_identity_fields(tmp_path):
    _write(tmp_path)
    expected = bench_report.summarise_steps(tmp_path, 60)
    (tmp_path / "steps.dp1.jsonl").rename(tmp_path / "steps.rank1.jsonl")
    assert bench_report.summarise_steps(tmp_path, 60) == expected


def test_legacy_gpu_denominator_uses_world_size_not_file_count(tmp_path):
    _write(tmp_path, ranks=1)
    path = tmp_path / "steps.dp0.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        record.update(world_size=4, dp_rank=0, tp_rank=0)
    _jsonl(path, records)
    steps = bench_report.summarise_steps(tmp_path, 60)
    assert steps["world_size"] == 4
    assert steps["dp_ranks"] == 1
    assert steps["mfu"]["mfu"] == pytest.approx(4e14 / (100 * 312e12 * 4))


def _json(path, value):
    path.write_text(json.dumps(value))


def _jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _benchmark(directory, *, dp=1, tp=1, warmup=1, start=12, workers=0, durations=None):
    durations = durations or [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (97.0, 96.0)]
    world = dp * tp
    _json(directory / "run.json", {
        "schema_version": 1, "world_size": world,
        "parallelism": {"dp": dp, "tp": tp, "pp": 1,
                        "context_parallel_size": 1, "expert_parallel_size": 1},
        "warmup_steps": warmup, "steps": len(durations),
        "baseline_config": {"model": "70M"}, "overrides": {},
        "revision": "fixture-commit",
    })
    for rank in range(world):
        dp_rank, tp_rank = divmod(rank, tp)
        _json(directory / f"rank{rank}.json", {
            "rank": rank, "dp_rank": dp_rank, "tp_rank": tp_rank,
            "status": "complete", "start_step": start,
            "startup_s": 5.0 + rank, "first_batch_s": 2.0 + rank,
            "whole_run_s": 10.0 + 100 * warmup + sum(d for d, _ in durations) + rank,
            "num_loading_workers": workers, "hardware": {"device": "fixture GPU"},
        })
        for worker in range(workers) if workers else [None]:
            consumer = "main" if worker is None else f"w{worker}"
            _json(directory / f"probe.rank{rank}.{consumer}.json", {
                "rank": rank, "dp_rank": dp_rank, "worker": worker,
                "pid": rank + 100, "instruments": ["_fetch_range", "stream.read"],
            })
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


def _change_json(path, **changes):
    value = json.loads(path.read_text())
    value.update(changes)
    _json(path, value)


def test_benchmark_excludes_warmup_and_offsets_resumed_steps(tmp_path):
    _benchmark(tmp_path, warmup=2, start=100)
    report = _report(tmp_path)
    assert report["status"] == "complete", report["diagnostics"]
    assert report["schema_version"] == 1
    assert report["run"]["baseline_config"] == {"model": "70M"}
    assert report["ranks"]["0"]["hardware"]["device"] == "fixture GPU"
    metrics = report["metrics"]
    assert metrics["timing"]["warmup_s"] == 200
    assert metrics["timing"]["measured_s"] == 100
    assert metrics["counts"]["tokens_nonpad"] == 3600
    assert metrics["counts"]["rows"] == 40
    assert metrics["counts"]["micro_batches"] == 8
    assert metrics["throughput"]["useful_tokens_per_s"] == 36
    assert metrics["packing"] == 0.9
    assert metrics["modalities"]["image"]["share"] == 0.8
    assert metrics["modalities"]["image"]["loss_tokens_per_s"] == 3.2
    assert metrics["modalities"]["image"]["target_values_per_s"] == 614.4
    text = bench_report.render(report)
    for phrase in ("Useful tokens/s", "first batch", "p99", "Loader waits", "Payload", "presentations"):
        assert phrase in text
    assert "<200 steps OR <60s heuristic" in text
    assert "not unique-object counts or corpus coverage" in text


def test_benchmark_rates_and_stalls_are_weighted(tmp_path):
    _benchmark(tmp_path, warmup=0)
    metrics = _report(tmp_path)["metrics"]
    assert metrics["throughput"]["useful_tokens_per_s"] == pytest.approx(3600 / 100)
    assert metrics["waits"]["stall_share"] == pytest.approx(96 / 100)
    assert metrics["step_seconds"]["mean"] == 25
    assert metrics["step_seconds"]["p95"] == 97
    assert metrics["step_seconds"]["slow_steps"] == 1
    assert metrics["timing"]["warmup_s"] == 0


def test_dp2_tp2_counts_only_logical_data_but_times_all_ranks(tmp_path):
    _benchmark(tmp_path, dp=2, tp=2)
    path = tmp_path / "steps.rank3.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    # A TP replica is slow, and cumulative time includes overhead between steps.
    records[-1]["elapsed_s"] += 20
    _jsonl(path, records)
    report = _report(tmp_path)
    assert report["status"] == "complete", report["diagnostics"]
    metrics = report["metrics"]
    assert metrics["timing"]["measured_s"] == 120
    assert metrics["counts"]["tokens_nonpad"] == 7200
    assert metrics["counts"]["loss_tokens"] == {"image": 640, "spectrum": 160}
    assert metrics["throughput"]["useful_tokens_per_s"] == 60
    assert metrics["throughput"]["useful_tokens_per_s_per_gpu"] == 15
    assert metrics["counts"]["rows"] == 80
    assert metrics["counts"]["micro_batches"] == 16


def test_measurement_subtracts_warmup_per_rank_before_taking_max(tmp_path):
    _benchmark(tmp_path, dp=2)
    path = tmp_path / "steps.rank1.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        record["elapsed_s"] -= 50
    records[-1]["elapsed_s"] += 20
    _jsonl(path, records)
    # max(end) - max(warmup) = 100, but max(end - warmup) = 120.
    assert _report(tmp_path)["metrics"]["timing"]["measured_s"] == 120


@pytest.mark.parametrize("workers", [0, 2])
def test_payload_counts_returns_including_duplicates_and_legacy_logs(tmp_path, workers):
    _benchmark(tmp_path, dp=2, workers=workers)
    payload = {"source": "legacy", "bytes": bench_report.MIB, "t": 1234, "wait_s": 1.5}
    consumer0, consumer1 = ("w0", "w1") if workers else ("main", "main")
    _jsonl(tmp_path / f"bytes.rank0.{consumer0}.jsonl", [payload, payload])
    _jsonl(tmp_path / f"bytes.rank1.{consumer1}.jsonl", [payload])
    _jsonl(tmp_path / "bytes.dp0.w0.jsonl", [payload])
    report = _report(tmp_path)
    assert report["status"] == "complete", report["diagnostics"]
    assert report["wire"]["total_bytes"] == 4 * bench_report.MIB
    assert report["wire"]["by_source"]["legacy"]["fetches"] == 4
    assert report["wire"]["fetch_wait_s"] == 6
    assert report["wire"]["whole_run_bytes_per_s"] == pytest.approx(4 * bench_report.MIB / 211)
    assert "prefetch" in report["wire"]["interval"]
    assert "fetcher shutdown" in report["wire"]["interval"]
    assert "not cross-node clocks" in report["wire"]["interval"]
    assert bench_report.summarise_bytes(tmp_path)["total_mib"] == 4


def test_empty_payload_with_probe_is_valid(tmp_path):
    _benchmark(tmp_path)
    report = _report(tmp_path)
    assert report["status"] == "complete", report["diagnostics"]
    assert report["wire"]["total_bytes"] == 0
    assert report["wire"]["whole_run_bytes_per_s"] == 0


@pytest.mark.parametrize("missing", [
    "rank1.json", "steps.rank1.jsonl", "probe.rank1.main.json", "bytes.rank1.main.jsonl",
])
def test_missing_rank_or_instrumentation_is_partial(tmp_path, missing):
    _benchmark(tmp_path, dp=2)
    (tmp_path / missing).unlink()
    report = _report(tmp_path)
    assert report["status"] == "partial"
    assert report["metrics"] == {}
    assert report["wire"]["whole_run_bytes_per_s"] is None
    assert any(missing in message for message in report["diagnostics"])


@pytest.mark.parametrize("status", ["failed", "running", None])
def test_noncomplete_rank_is_partial(tmp_path, status):
    _benchmark(tmp_path)
    _change_json(tmp_path / "rank0.json", status=status)
    report = _report(tmp_path)
    assert report["status"] == "partial"
    assert report["metrics"] == {}
    assert any("status" in message for message in report["diagnostics"])


@pytest.mark.parametrize("problem", ["missing_warmup", "missing_measured", "duplicate", "extra", "wrong_rank", "truncated"])
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
        records.append({**records[-1], "step": 18, "elapsed_s": 201})
    elif problem == "wrong_rank":
        records[-1]["rank"] = 1
    _jsonl(path, records)
    if problem == "truncated":
        path.write_text(path.read_text() + '{"step":')
    before = path.read_bytes()
    report = _report(tmp_path)
    assert report["status"] == "partial"
    assert report["diagnostics"]
    assert report["metrics"] == {}
    assert "No headline throughput" in bench_report.render(report)
    assert path.read_bytes() == before


@pytest.mark.parametrize("filename,changes", [
    ("rank0.json", {"rank": 1}),
    ("rank1.json", {"dp_rank": 0}),
    ("rank1.json", {"start_step": 0}),
    ("rank1.json", {"tp_rank": 1}),
    ("run.json", {"world_size": 3}),
    ("probe.rank1.main.json", {"rank": 0}),
    ("probe.rank1.main.json", {"instruments": ["_fetch_range"]}),
])
def test_invalid_identity_topology_or_probe_is_partial(tmp_path, filename, changes):
    _benchmark(tmp_path, dp=2)
    _change_json(tmp_path / filename, **changes)
    report = _report(tmp_path)
    assert report["status"] == "partial"
    assert report["diagnostics"]
    assert report["metrics"] == {}


def test_all_configured_workers_must_have_probes(tmp_path):
    _benchmark(tmp_path, workers=2)
    (tmp_path / "probe.rank0.w1.json").unlink()
    assert _report(tmp_path)["status"] == "partial"


@pytest.mark.parametrize("filename", ["run.json", "rank0.json", "probe.rank0.main.json", "bytes.rank0.main.jsonl"])
def test_truncated_metadata_or_payload_is_partial(tmp_path, filename):
    _benchmark(tmp_path)
    (tmp_path / filename).write_text('{"source":')
    report = _report(tmp_path)
    assert report["status"] == "partial"
    assert report["metrics"] == {}
    assert any(filename in message for message in report["diagnostics"])


@pytest.mark.parametrize("steps,seconds,warn", [(240, 0.25, False), (199, 1, True), (200, 0.1, True)])
def test_short_window_warning_uses_either_threshold(tmp_path, steps, seconds, warn):
    _benchmark(tmp_path, warmup=0, durations=[(seconds, 0)] * steps)
    report = _report(tmp_path)
    assert bool(report["warnings"]) is warn


@pytest.mark.parametrize("complete", [True, False])
def test_cli_exit_status_json_and_raw_logs_unchanged(tmp_path, complete):
    _benchmark(tmp_path)
    if not complete:
        (tmp_path / "steps.rank0.jsonl").write_text('{"step":')
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}
    out = tmp_path / "report.json"
    result = subprocess.run([
        sys.executable, str(bench_report.__file__), "--telemetry", str(tmp_path), "--out", str(out),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == (0 if complete else 1), result.stderr
    report = json.loads(out.read_text())
    assert report["status"] == ("complete" if complete else "partial")
    assert all(path.read_bytes() == content for path, content in original.items())


def test_missing_manifest_is_not_misread_as_successful_legacy_report(tmp_path):
    _benchmark(tmp_path)
    (tmp_path / "run.json").unlink()
    report = _report(tmp_path)
    assert report["status"] == "partial"
    assert report["metrics"] == {}
    assert "run.json" in report["diagnostics"][0]


def test_cli_refuses_to_overwrite_raw_inputs(tmp_path, monkeypatch):
    _benchmark(tmp_path)
    path = tmp_path / "run.json"
    original = path.read_bytes()
    monkeypatch.setattr(sys, "argv", ["bench_report", "--telemetry", str(tmp_path), "--out", str(path)])
    with pytest.raises(SystemExit) as exc:
        bench_report.main()
    assert exc.value.code != 0
    assert path.read_bytes() == original
