"""Tests for HfFileSystem transport hygiene and bounded concurrent range reads."""

from __future__ import annotations

import threading
import time

from fsspec.spec import AbstractBufferedFile
from huggingface_hub import HfFileSystem
from huggingface_hub.hf_file_system import HfFileSystemFile

from astropt3.data.hf_io import disable_hf_cat_readahead, enable_hf_concurrent_ranges


class _RecordingFile(AbstractBufferedFile):
    def __init__(self, fs, path, payload: bytes, **kwargs):
        self._payload = payload
        super().__init__(fs, path, size=len(payload), **kwargs)

    def _fetch_range(self, start: int, end: int) -> bytes:
        self.fs.fetches.append((start, end))
        return self._payload[start:end]


class _FakeHfFS(HfFileSystem):
    def __init__(self, payload: bytes, **kwargs):
        super().__init__(**kwargs)
        self._payload = payload
        self.fetches: list[tuple[int, int]] = []
        self.open_kwargs: list[dict] = []

    def _open(self, path, mode="rb", **kwargs):
        self.open_kwargs.append(dict(kwargs))
        return _RecordingFile(self, path, self._payload, **kwargs)


def test_disable_hf_cat_readahead_fetches_exact_range(monkeypatch):
    original = getattr(HfFileSystem.cat_file, "__wrapped__", HfFileSystem.cat_file)
    monkeypatch.setattr(HfFileSystem, "cat_file", original)

    payload = bytes(range(256)) * (40 * 1024)  # 10 MiB
    before = _FakeHfFS(payload, skip_instance_cache=True)
    assert before.cat_file("datasets/org/repo/data.parquet", start=128, end=1128) == payload[128:1128]
    assert before.fetches == [(128, 1128 + 5 * 1024 * 1024)]

    disable_hf_cat_readahead()
    wrapped = HfFileSystem.cat_file
    disable_hf_cat_readahead()
    assert HfFileSystem.cat_file is wrapped  # idempotent

    after = _FakeHfFS(payload, skip_instance_cache=True)
    assert after.cat_file("datasets/org/repo/data.parquet", start=128, end=1128) == payload[128:1128]
    assert after.fetches == [(128, 1128)]
    assert after.open_kwargs[-1]["cache_type"] == "none"

    after.fetches.clear()
    assert after.cat_file("datasets/org/repo/data.parquet", start=-256) == payload[-256:]
    assert after.fetches == [(len(payload) - 256, len(payload))]

    after.open_kwargs.clear()
    after.cat_file("datasets/org/repo/data.parquet", start=0, end=64, cache_type="bytes")
    assert after.open_kwargs[-1]["cache_type"] == "bytes"
    after.cat_file("datasets/org/repo/data.parquet")
    assert "cache_type" not in after.open_kwargs[-1]


def test_concurrent_cat_ranges_and_chunked_fetch_range(monkeypatch):
    orig_cat_file = getattr(HfFileSystem.cat_file, "__wrapped__", HfFileSystem.cat_file)
    orig_cat_ranges = getattr(HfFileSystem.cat_ranges, "__wrapped__", HfFileSystem.cat_ranges)
    orig_fetch = HfFileSystemFile._fetch_range
    monkeypatch.setattr(HfFileSystem, "cat_file", orig_cat_file)
    monkeypatch.setattr(HfFileSystem, "cat_ranges", orig_cat_ranges)

    payload = bytes(range(256)) * 4096  # 1 MiB
    calls: list[tuple[int, int]] = []
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_base_fetch(self, start: int, end: int) -> bytes:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            calls.append((start, end))
        time.sleep(0.02)
        with lock:
            active -= 1
        return payload[start:end]

    monkeypatch.setattr(HfFileSystemFile, "_fetch_range", fake_base_fetch)
    enable_hf_concurrent_ranges(chunk_bytes=256 * 1024)

    # 1) Large single _fetch_range splits into 4 concurrent 256 KiB sub-ranges.
    dummy_file = object.__new__(HfFileSystemFile)
    got = HfFileSystemFile._fetch_range(dummy_file, 0, len(payload))
    assert got == payload
    assert sorted(calls) == [(i * 256 * 1024, (i + 1) * 256 * 1024) for i in range(4)]
    assert max_active >= 2

    # 2) cat_ranges executes multiple ranges concurrently and returns results in order.
    fs = _FakeHfFS(payload, skip_instance_cache=True)
    orig_open = fs._open

    def slow_open(path, mode="rb", **kwargs):
        f = orig_open(path, mode=mode, **kwargs)
        orig_f_fetch = f._fetch_range

        def timed(s, e):
            time.sleep(0.02)
            return orig_f_fetch(s, e)

        f._fetch_range = timed
        return f

    fs._open = slow_open
    starts = [0, 1000, 2000, 3000]
    ends = [500, 1500, 2500, 3500]
    results = fs.cat_ranges(["p"] * 4, starts, ends)
    assert results == [payload[s:e] for s, e in zip(starts, ends)]
    assert sorted(fs.fetches) == list(zip(starts, ends))

    # Restore class method after test.
    monkeypatch.setattr(HfFileSystemFile, "_fetch_range", orig_fetch)
