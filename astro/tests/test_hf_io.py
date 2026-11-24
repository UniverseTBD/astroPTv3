"""Tests for HfFileSystem transport hygiene and bounded concurrent range reads."""

from __future__ import annotations

import threading
import time

import fsspec
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.spec import AbstractBufferedFile
from huggingface_hub import HfFileSystem
from huggingface_hub.hf_file_system import HfFileSystemFile
from pandas.testing import assert_frame_equal

from astropt3.data.hf_io import (
    _CoalescedReadUnsupported,
    _read_coalesced_parquet_to_nestedframe,
    disable_hf_cat_readahead,
    enable_hf_coalesced_parquet_reads,
    enable_hf_concurrent_ranges,
)


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


def test_coalesced_parquet_prunes_leaves_and_row_groups(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    size = 1000
    nested = pa.StructArray.from_arrays(
        [pa.array(list(rng.random((size, 80)))) for _ in range(3)],
        names=["flux", "ivar", "lam"],
        mask=pa.array(np.arange(size) == 205),
    )
    table = pa.Table.from_pandas(
        pd.DataFrame({"a": rng.random(size)}, index=pd.Index(np.arange(size, dtype=np.int64), name="_healpix_29"))
    ).append_column("nested", nested)
    pq_path = tmp_path / "part.parquet"
    pq.write_table(table, pq_path, row_group_size=100)
    file_bytes = pq_path.stat().st_size

    fs = fsspec.filesystem("memory")
    mpath = f"/{tmp_path.name}/part.parquet"
    fs.put_file(str(pq_path), mpath)
    try:
        sizes: list[int] = []
        orig_cat = MemoryFileSystem.cat_file

        def counting_cat(self, *args, **kwargs):
            data = orig_cat(self, *args, **kwargs)
            sizes.append(len(data))
            return data

        monkeypatch.setattr(MemoryFileSystem, "cat_file", counting_cat)

        # 1) Dotted struct leaves (out of file order) + DNF row-group filter + differing schema metadata.
        filters = [[("_healpix_29", ">=", 200), ("_healpix_29", "<", 300)]]
        cols = ["_healpix_29", "nested.lam", "a", "nested.flux"]
        schema_no_meta = pq.read_schema(pq_path).remove_metadata()
        expected = npd.read_parquet(pq_path, columns=cols, filters=filters, schema=schema_no_meta)
        got = _read_coalesced_parquet_to_nestedframe(
            fs, mpath, columns=cols, filters=filters, schema=schema_no_meta
        )
        assert_frame_equal(got, expected)
        # 1 of 10 row groups × 2 of 3 struct leaves => well under 25% of file bytes.
        assert 0 < sum(sizes) < 0.25 * file_bytes

        # 2) Whole-struct filtered read prunes row groups on the wire (e.g. galaxies-with-hats).
        sizes.clear()
        expected_whole = npd.read_parquet(pq_path, columns=["_healpix_29", "nested"], filters=filters)
        got_whole = _read_coalesced_parquet_to_nestedframe(
            fs, mpath, columns=["_healpix_29", "nested"], filters=filters
        )
        assert_frame_equal(got_whole, expected_whole)
        assert 0 < sum(sizes) < 0.25 * file_bytes

        # 3) Empty row-group selection fetches footer only and returns exact empty schema.
        sizes.clear()
        empty_filters = [[("_healpix_29", ">=", 99999), ("_healpix_29", "<", 100000)]]
        got_empty = _read_coalesced_parquet_to_nestedframe(
            fs, mpath, columns=["_healpix_29", "nested.flux"], filters=empty_filters
        )
        assert len(got_empty) == 0
        assert list(got_empty.columns) == ["nested"]
        assert got_empty.index.name == "_healpix_29"
        assert len(sizes) == 1 and sizes[0] <= 256_000  # single footer read, zero data ranges

        # 4) Unsupported options raise _CoalescedReadUnsupported for clean fallback.
        with pytest.raises(_CoalescedReadUnsupported):
            _read_coalesced_parquet_to_nestedframe(
                fs, mpath, columns=["_healpix_29", "nested.flux"], filters=pc.field("_healpix_29") < 5
            )
    finally:
        fs.rm(mpath)


def test_enable_hf_coalesced_parquet_reads_routes_only_hf_single_files(tmp_path, monkeypatch):
    import hats.io.file_io as hats_file_io_pkg
    import hats.io.file_io.file_io as hats_file_io_mod

    orig_fn = getattr(
        hats_file_io_mod.read_parquet_file_to_pandas,
        "__wrapped__",
        hats_file_io_mod.read_parquet_file_to_pandas,
    )
    monkeypatch.setattr(hats_file_io_mod, "read_parquet_file_to_pandas", orig_fn)
    monkeypatch.setattr(hats_file_io_pkg, "read_parquet_file_to_pandas", orig_fn)

    rng = np.random.default_rng(11)
    nested = pa.StructArray.from_arrays(
        [pa.array(list(rng.random((20, 10)))) for _ in range(2)], names=["flux", "ivar"]
    )
    pq_path = tmp_path / "hf_part.parquet"
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame({"a": rng.random(20)}, index=pd.Index(np.arange(20), name="_healpix_29")))
        .append_column("nested", nested),
        pq_path,
    )
    payload = pq_path.read_bytes()
    fake_fs = _FakeHfFS(payload, skip_instance_cache=True)
    fake_fs.size = lambda _p: len(payload)

    class _FakeHfUPath:
        protocol = "hf"
        path = "datasets/org/repo/Npix=0.parquet"
        fs = fake_fs

    fallback_calls = []

    def spy_orig(fp, is_dir=None, **kwargs):
        fallback_calls.append((fp, is_dir))
        return orig_fn(fp, is_dir=is_dir, **kwargs)

    monkeypatch.setattr(hats_file_io_mod, "read_parquet_file_to_pandas", spy_orig)
    monkeypatch.setattr(hats_file_io_pkg, "read_parquet_file_to_pandas", spy_orig)

    enable_hf_coalesced_parquet_reads()
    wrapped = hats_file_io_mod.read_parquet_file_to_pandas
    enable_hf_coalesced_parquet_reads()
    assert hats_file_io_mod.read_parquet_file_to_pandas is wrapped  # idempotent
    assert hats_file_io_pkg.read_parquet_file_to_pandas is wrapped

    # Local files still route to the original reader untouched.
    local_nf = hats_file_io_mod.read_parquet_file_to_pandas(
        pq_path, is_dir=False, columns=["_healpix_29", "nested.flux"]
    )
    assert len(fallback_calls) == 1

    # hf:// single files take the coalesced path without calling fallback.
    monkeypatch.setattr(hats_file_io_mod, "get_upath", lambda _fp: _FakeHfUPath())
    hf_nf = hats_file_io_mod.read_parquet_file_to_pandas(
        "hf://datasets/org/repo/Npix=0.parquet",
        is_dir=False,
        columns=["_healpix_29", "nested.flux"],
    )
    assert len(fallback_calls) == 1
    assert_frame_equal(hf_nf, local_nf)
    assert len(fake_fs.fetches) > 0

    # Restore original function after test.
    monkeypatch.setattr(hats_file_io_mod, "read_parquet_file_to_pandas", orig_fn)
    monkeypatch.setattr(hats_file_io_pkg, "read_parquet_file_to_pandas", orig_fn)
