"""ADR 0014 §3 instrumentation: wire bytes in, loss-bearing tokens out.

Two halves, because the numbers live in different processes:

- **Byte probe** — measures returned ``HfFileSystemFile._fetch_range`` and
  ``HfFileSystemStreamFile.read`` payloads, not request counts. The latter
  includes whole-file metadata reads that bypass the range reader. It runs
  inside each loader worker (or the main process with zero workers), writing
  its own world-rank/worker byte log: one line per returned payload with
  the fields §3 asks for — rank, worker, source, path, byte range, payload
  bytes, fetch wait.
- **Step counters** — accumulated in the MAIN process by :class:`TelemetryLoader`,
  because worker-process counters have no way back to the trainer. Everything
  §3 wants per step is recoverable from the flat micro-batch itself
  (non-padding tokens, loss-bearing tokens and target values per modality,
  packing utilisation) plus the measured wait on ``next()``.

``scripts/bench_report.py`` reports measured-step tokens separately from
whole-run payload bytes: prefetch is not measured-window consumption.
Its legacy optional ``--object-log`` join (distinct base
objects and replica counts via ``#n`` suffixes) needs a per-object log the
loader no longer writes (ADR 0015 dropped ``object_id_log`` with the rest of
the dataset-checkpoint machinery); byte and step-counter telemetry are
unaffected.

Everything here is a no-op unless ``$ASTROPT3_TELEMETRY_DIR`` is set, so
production runs pay nothing and the benchmark opts in.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache, wraps
from pathlib import Path
from threading import Lock
from typing import Any

from ..tokenization import BOS_ID, PAD_ID

TELEMETRY_DIR_ENV = "ASTROPT3_TELEMETRY_DIR"

_CATALOG_SOURCES = {
    "legacy": "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north",
    "desi": "hf://datasets/UniverseTBD/mmu_desi_edr_sv3",
}
_REPO_TO_SOURCE = {
    catalog.split("hf://datasets/")[-1].split("@")[0]: source
    for source, catalog in _CATALOG_SOURCES.items()
}


@lru_cache(maxsize=4)
def _resolved_dir(raw: str) -> Path:
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def telemetry_dir() -> Path | None:
    """The configured output directory, or None when instrumentation is off.

    Cached: the trainer calls this once per step, and an mkdir per step on
    BeeGFS is a needless round trip in the logging path.
    """
    raw = os.environ.get(TELEMETRY_DIR_ENV)
    return _resolved_dir(raw) if raw else None


def source_of_path(path: str) -> str:
    """Attribute a hub path to a catalog source, for per-source byte totals."""
    for repo, source in _REPO_TO_SOURCE.items():
        if repo in path:
            return source
    return "unknown"


# -- byte probe (runs inside each loader worker) ------------------------------

_probe_pid: int | None = None


def install_byte_probe(rank: int = 0, worker: int | None = None) -> bool:
    """Log returned range and streaming-read payloads, including duplicates.

    Idempotent per process (each DataLoader worker is its own process and
    calls this once from :meth:`PackedMicroBatches.__iter__`). Returns whether
    the probe is active.
    """
    global _probe_pid
    if _probe_pid == os.getpid():
        return True
    directory = telemetry_dir()
    if directory is None:
        return False
    try:
        from huggingface_hub.hf_file_system import (
            HfFileSystemFile,
            HfFileSystemStreamFile,
        )
    except ImportError:  # hub not installed (synthetic-only runs)
        return False

    world_rank = int(os.environ.get("RANK", rank))
    consumer = "main" if worker is None else f"w{worker}"
    stem = f"rank{world_rank}.{consumer}"
    log_path = directory / f"bytes.{stem}.jsonl"
    log_path.touch()
    lock = Lock()
    (directory / f"probe.{stem}.json").write_text(json.dumps({
        "rank": world_rank, "dp_rank": rank, "worker": worker,
        "pid": os.getpid(), "instruments": ["_fetch_range", "stream.read"],
    }))

    def record(file, payload, start, began, clock):
        path = str(getattr(file, "path", ""))
        entry = {
            "t": began, "completed_t": time.time(),
            "wait_s": time.perf_counter() - clock,
            "rank": world_rank, "dp_rank": rank, "worker": worker,
            "source": source_of_path(path), "path": path,
            "start": start, "end": start + len(payload), "bytes": len(payload),
        }
        # Dask fetches several partitions concurrently in this process.
        # pi-lens-ignore: unchecked-throwing-call-python -- failed logging must fail the benchmark
        with lock, open(log_path, "a") as log:
            log.write(json.dumps(entry) + "\n")

    original = getattr(HfFileSystemFile._fetch_range, "__wrapped__", HfFileSystemFile._fetch_range)
    original_read = getattr(HfFileSystemStreamFile.read, "__wrapped__", HfFileSystemStreamFile.read)

    @wraps(original)
    def _fetch_range(self, start: int, end: int) -> bytes:
        began, clock = time.time(), time.perf_counter()
        payload = original(self, start, end)
        record(self, payload, start, began, clock)
        return payload

    @wraps(original_read)
    def read(self, length=-1):
        began, clock, start = time.time(), time.perf_counter(), self.loc
        payload = original_read(self, length)
        record(self, payload, start, began, clock)
        return payload

    HfFileSystemFile._fetch_range = _fetch_range
    HfFileSystemStreamFile.read = read
    _probe_pid = os.getpid()
    return True


# -- per-step counters (main process) -----------------------------------------

_counters: dict[str, Any] = {}


def _blank() -> dict[str, Any]:
    return {
        "micro_batches": 0,
        "loader_wait_s": 0.0,
        "tokens_total": 0,
        "tokens_nonpad": 0,
        "rows": 0,
        "loss_tokens": {},
        "target_values": {},
    }


def observe_micro_batch(flat: dict, wait_s: float) -> None:
    """Accumulate one flat micro-batch into the current step's counters.

    ``{m}_mask`` counts loss-bearing tokens per modality (which is also the
    per-step composition measurement ADR 0014 §4 wants — exact, unlike
    inferring composition from object ids); ``{m}_values`` counts the valid
    target dimensions behind ``E_values``. ``rows`` counts galaxies (packed
    objects), via ``<|bos|>`` occurrences — packing puts several objects per
    row, so this is not ``input_ids.shape[0]``.
    """
    if not _counters:
        _counters.update(_blank())
    input_ids = flat.get("input_ids")
    if input_ids is None:
        return
    _counters["micro_batches"] += 1
    _counters["loader_wait_s"] += wait_s
    # pi-lens-ignore: unchecked-throwing-call-python
    _counters["tokens_total"] += int(input_ids.numel())
    # pi-lens-ignore: unchecked-throwing-call-python
    _counters["tokens_nonpad"] += int((input_ids != PAD_ID).sum().item())
    # pi-lens-ignore: unchecked-throwing-call-python
    _counters["rows"] += int((input_ids == BOS_ID).sum().item())
    for key, value in flat.items():
        if key.endswith("_mask"):
            name = key[: -len("_mask")]
            # pi-lens-ignore: unchecked-throwing-call-python
            count = int(value.sum().item())
            if count:
                _counters["loss_tokens"][name] = (
                    _counters["loss_tokens"].get(name, 0) + count
                )
        elif key.endswith("_values"):
            name = key[: -len("_values")]
            # pi-lens-ignore: unchecked-throwing-call-python
            count = int(value.numel())
            if count:
                _counters["target_values"][name] = (
                    _counters["target_values"].get(name, 0) + count
                )


def drain_step() -> dict[str, Any]:
    """Return and reset the accumulated counters, with derived ratios.

    ``utilisation_packing`` is the §2a MFU factor: padding earns no credit,
    so a padded-out packed row is wasted compute exactly as a stall is
    wasted time.

    ``rows_per_s`` is galaxies delivered per second of ``loader_wait_s`` —
    the main process was blocked on the loader for exactly that long, so
    this is the loader's own throughput, isolated from step compute time
    (the number a fair loader-backend comparison wants).
    """
    record = dict(_counters) if _counters else _blank()
    _counters.clear()
    total = record["tokens_total"]
    record["utilisation_packing"] = (
        record["tokens_nonpad"] / total if total else 0.0
    )
    record["rows_per_s"] = (
        record["rows"] / record["loader_wait_s"] if record["loader_wait_s"] else 0.0
    )
    return record


def write_step(step: int, record: dict, rank: int = 0) -> None:
    """Append one per-step record; ``rank`` is the WORLD rank, not DP rank."""
    directory = telemetry_dir()
    if directory is None:
        return
    # pi-lens-ignore: unchecked-throwing-call-python
    with open(directory / f"steps.rank{rank}.jsonl", "a") as log:
        log.write(json.dumps({"step": step, **record, "rank": rank}) + "\n")


# -- loader wrapper -----------------------------------------------------------


class TelemetryLoader:
    """Times ``next()`` on a DataLoader and observes each micro-batch.

    Everything else proxies through, so ``loader_state_dict`` (which asks for
    ``state_dict``/``num_workers``) and checkpoint resume are unaffected. The
    wait measured here IS the GPU-visible stall: it is the main process
    blocking on the loader, regardless of how many workers feed it.
    """

    _loader: Any = None  # class attribute: keeps __getattr__ from recursing

    def __init__(self, loader):
        self._loader = loader
        self._iterator = None
        self._first = None

    def prime(self) -> float:
        """Fetch and retain batch one; startup wait is not a training stall."""
        if self._iterator is not None:
            raise RuntimeError("loader already started")
        began = time.perf_counter()
        self._iterator = iter(self._loader)
        self._first = next(self._iterator)
        return time.perf_counter() - began

    def close(self) -> None:
        """Stop prefetch before reporting the whole-run payload total."""
        iterator, self._iterator = self._iterator, None
        self._first = None
        if iterator is None:
            return
        if hasattr(iterator, "_shutdown_workers"):
            workers = list(iterator._workers)
            iterator._shutdown_workers()
            # PyTorch may terminate overdue workers without joining them.
            for worker in workers:
                worker.join()
        else:
            # PyTorch exposes no public close for a single-process iterator.
            fetcher = getattr(iterator, "_dataset_fetcher", None)
            source = getattr(fetcher, "dataset_iter", iterator)
            if hasattr(source, "close"):
                source.close()

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self._loader)
        if self._first is not None:
            batch, self._first = self._first, None
            observe_micro_batch(batch, 0.0)
            yield batch
        while True:
            began = time.perf_counter()
            try:
                batch = next(self._iterator)
            except StopIteration:
                return
            observe_micro_batch(batch, time.perf_counter() - began)
            yield batch


def instrument(loader):
    """Wrap a loader when telemetry is on, else hand it back untouched."""
    return TelemetryLoader(loader) if telemetry_dir() is not None else loader
