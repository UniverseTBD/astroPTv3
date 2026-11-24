"""Tests for HfFileSystem transport hygiene."""

from __future__ import annotations

from fsspec.spec import AbstractBufferedFile
from huggingface_hub import HfFileSystem

from astropt3.data.hf_io import disable_hf_cat_readahead


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
    # Start from the unpatched fsspec cat_file to verify both before and after.
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

    # Explicit caller cache_type is respected; whole-file reads keep default caching.
    after.open_kwargs.clear()
    after.cat_file("datasets/org/repo/data.parquet", start=0, end=64, cache_type="bytes")
    assert after.open_kwargs[-1]["cache_type"] == "bytes"
    after.cat_file("datasets/org/repo/data.parquet")
    assert "cache_type" not in after.open_kwargs[-1]
