"""Transport hygiene for Hugging Face Hub fsspec range reads."""

from __future__ import annotations

from functools import wraps

from huggingface_hub import HfFileSystem


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
