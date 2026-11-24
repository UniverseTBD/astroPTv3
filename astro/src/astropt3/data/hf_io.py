"""Transport hygiene and bounded concurrency for Hugging Face Hub fsspec reads."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import Lock

from huggingface_hub import HfFileSystem
from huggingface_hub.hf_file_system import HfFileSystemFile

# Split single large range GETs (> 32 MiB) into bounded concurrent sub-ranges to avoid
# ~95 MB/s single-TCP-stream stalls on 200-250 MiB Parquet row groups (ADR 0014 §10a).
RANGE_CHUNK_BYTES = 32 * 1024 * 1024
MAX_CAT_RANGE_WORKERS = 4
MAX_CHUNK_WORKERS = 8

_pool_lock = Lock()
_pool_pid: int | None = None
_cat_pool: ThreadPoolExecutor | None = None
_chunk_pool: ThreadPoolExecutor | None = None


def _pools() -> tuple[ThreadPoolExecutor, ThreadPoolExecutor]:
    """Return distinct bounded pools for ``cat_ranges`` and sub-range splitting.

    Keeping the two pools separate prevents nested submission deadlocks when
    ``cat_ranges`` invokes ``cat_file`` -> ``_fetch_range`` on a >32 MiB range.
    """
    global _pool_pid, _cat_pool, _chunk_pool
    pid = os.getpid()
    if _pool_pid == pid and _cat_pool is not None and _chunk_pool is not None:
        return _cat_pool, _chunk_pool
    with _pool_lock:
        if _pool_pid != pid or _cat_pool is None or _chunk_pool is None:
            _cat_pool = ThreadPoolExecutor(
                max_workers=MAX_CAT_RANGE_WORKERS, thread_name_prefix="hf-cat-range"
            )
            _chunk_pool = ThreadPoolExecutor(
                max_workers=MAX_CHUNK_WORKERS, thread_name_prefix="hf-range-chunk"
            )
            _pool_pid = pid
        return _cat_pool, _chunk_pool


def disable_hf_cat_readahead() -> None:
    """Avoid fsspec's 5 MiB ReadAheadCache over-fetch on bounded ``cat_file`` reads.

    ``HfFileSystem`` inherits ``AbstractFileSystem.cat_file()``, which opens a temporary
    handle with the default ``cache_type="readahead"`` and 5 MiB block size. Every
    Parquet column range fetched through ``fsspec.parquet`` therefore over-fetches 5 MiB
    past ``end`` and immediately discards it when the handle closes. Passing
    ``cache_type="none"`` keeps range reads exact while still routing through
    ``HfFileSystemFile._fetch_range`` (and its byte telemetry probe).
    """
    current = HfFileSystem.cat_file
    if getattr(current, "_astropt3_no_readahead", False):
        return

    @wraps(current)
    def cat_file(self, path, start=None, end=None, **kwargs):
        if start is not None or end is not None:
            kwargs.setdefault("cache_type", "none")
        return current(self, path, start=start, end=end, **kwargs)

    cat_file._astropt3_no_readahead = True  # type: ignore[attr-defined]
    HfFileSystem.cat_file = cat_file


def enable_hf_concurrent_ranges(
    *, chunk_bytes: int = RANGE_CHUNK_BYTES, max_range_workers: int = MAX_CAT_RANGE_WORKERS
) -> None:
    """Parallelize ``HfFileSystem.cat_ranges`` and split large ``_fetch_range`` GETs.

    Works for both ``fsspec.parquet`` (which calls ``cat_ranges``) and direct
    ``pyarrow.parquet.ParquetFile`` reads (which call ``HfFileSystemFile._fetch_range``).
    Preserves exact byte accounting when combined with ``telemetry.install_byte_probe``.
    """
    disable_hf_cat_readahead()

    current_cat_ranges = HfFileSystem.cat_ranges
    if not getattr(current_cat_ranges, "_astropt3_concurrent_ranges", False):
        @wraps(current_cat_ranges)
        def cat_ranges(self, paths, starts, ends, max_gap=None, on_error="return", **kwargs):
            if max_gap is not None:
                raise NotImplementedError
            if not isinstance(paths, list):
                raise TypeError
            if not isinstance(starts, list):
                starts = [starts] * len(paths)
            if not isinstance(ends, list):
                ends = [ends] * len(paths)
            if len(starts) != len(paths) or len(ends) != len(paths):
                raise ValueError
            if len(paths) <= 1 or max_range_workers <= 1:
                return current_cat_ranges(
                    self, paths, starts, ends, max_gap=max_gap, on_error=on_error, **kwargs
                )

            def fetch_one(item):
                p, s, e = item
                try:
                    return self.cat_file(p, s, e, **kwargs)
                except Exception as exc:
                    if on_error == "return":
                        return exc
                    raise

            cat_pool, _ = _pools()
            return list(cat_pool.map(fetch_one, zip(paths, starts, ends)))

        cat_ranges._astropt3_concurrent_ranges = True  # type: ignore[attr-defined]
        HfFileSystem.cat_ranges = cat_ranges

    current_fetch = HfFileSystemFile._fetch_range
    if not getattr(current_fetch, "_astropt3_concurrent_fetch", False):
        def _fetch_range(self, start: int, end: int) -> bytes:
            if chunk_bytes <= 0 or (end - start) <= chunk_bytes:
                return current_fetch(self, start, end)
            spans = [
                (offset, min(offset + chunk_bytes, end))
                for offset in range(start, end, chunk_bytes)
            ]
            _, chunk_pool = _pools()
            parts = list(chunk_pool.map(lambda span: current_fetch(self, span[0], span[1]), spans))
            return b"".join(parts)

        _fetch_range.__name__ = current_fetch.__name__
        _fetch_range.__doc__ = current_fetch.__doc__
        _fetch_range._astropt3_concurrent_fetch = True  # type: ignore[attr-defined]
        # Intentionally do NOT set __wrapped__ so telemetry.install_byte_probe does not unwrap it.
        HfFileSystemFile._fetch_range = _fetch_range
