"""Share immutable Hub directory and HATS catalog metadata without caching payloads."""

from __future__ import annotations

import json
import pickle
import re
import zlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Any

import pyarrow as pa
import torch.distributed as dist
from huggingface_hub import HfFileSystem

from .hf_io import disable_hf_cat_readahead

_PINNED_RE = re.compile(r"@([a-fA-F0-9]{40})(?:/|$)")


def _pinned_repos(repos: Iterable[str]) -> list[str]:
    return sorted({repo for repo in repos if _PINNED_RE.search(repo)})


def share_hf_listing(repos: Iterable[str]) -> bytes | None:
    """List pinned repositories on rank zero; call on every rank before spawning readers."""
    # List the whole revision, including collection siblings such as margin catalogs.
    roots = sorted({
        f"datasets/{repo.split('@')[0]}@{match[1]}"
        for repo in repos if (match := _PINNED_RE.search(repo))
    })
    if not roots:
        return None

    def collect() -> bytes:
        dircache: dict[str, Any] = {}
        revisions: set[tuple[Any, ...]] = set()

        def list_root(root: str):
            fs = HfFileSystem(skip_instance_cache=True)
            fs.find(root)
            return fs.dircache, [
                key for key, (exists, _) in fs._repo_and_revision_exists_cache.items()
                if exists and key[2] is not None
            ]

        if len(roots) == 1:
            dc, revs = list_root(roots[0])
            dircache.update(dc)
            revisions.update(revs)
        else:
            with ThreadPoolExecutor(max_workers=min(8, len(roots))) as pool:
                for dc, revs in pool.map(list_root, roots):
                    dircache.update(dc)
                    revisions.update(revs)
        # Full recipe trees can be tens of MB; send a compact snapshot to each worker.
        return zlib.compress(
            json.dumps((dircache, [list(k) for k in sorted(revisions)]), separators=(",", ":")).encode(),
            level=1,
        )

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return collect()
    listing: list[bytes | None] = [None]
    error = None
    if dist.get_rank() == 0:
        try:
            listing[0] = collect()
        except Exception as exc:
            error = exc
    dist.broadcast_object_list(listing, src=0)
    if listing[0] is None:
        raise RuntimeError("Rank zero failed to list the pinned AstroPT3 catalogs") from error
    return listing[0]


def seed_hf_listing(listing: bytes | None) -> None:
    """Populate LSDB's filesystem from the trusted rank-zero snapshot."""
    if listing is None:
        return
    fs = HfFileSystem()
    try:
        directories, revisions = json.loads(zlib.decompress(listing))
    except (ValueError, zlib.error) as exc:
        raise RuntimeError("Invalid AstroPT3 Hub directory snapshot") from exc
    fs.dircache.update(directories)
    fs._repo_and_revision_exists_cache.update({tuple(key): (True, None) for key in revisions})


def share_hats_metadata(repos: Iterable[str], *, listing: bytes | None = None) -> bytes | None:
    """Read pinned HATS metadata + MOCs once on rank zero and broadcast to all ranks.

    Avoids every DataLoader worker downloading ~574 MB of ``point_map.fits`` and
    HATS metadata files on startup and after transport retries.
    """
    pinned = _pinned_repos(repos)
    if not pinned:
        return None

    def collect() -> bytes:
        import hats as hc

        seed_hf_listing(listing)
        disable_hf_cat_readahead()
        with suppress(pa.ArrowKeyError):
            pa.unregister_extension_type("datasets.features.features.Array2DExtensionType")

        def load_one(repo: str):
            hc_cat = hc.read_hats(f"hf://datasets/{repo}")
            margin_hc = None
            if isinstance(hc_cat, hc.catalog.CatalogCollection) and hc_cat.default_margin_catalog_dir is not None:
                margin_hc = hc.read_hats(
                    hc_cat.default_margin_catalog_dir, single_catalog=True, read_moc=False
                )
            return repo, (hc_cat, margin_hc)

        if len(pinned) == 1:
            entries = [load_one(pinned[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(8, len(pinned))) as pool:
                entries = list(pool.map(load_one, pinned))
        return zlib.compress(pickle.dumps(dict(entries), protocol=pickle.HIGHEST_PROTOCOL), level=1)

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return collect()
    payload: list[bytes | None] = [None]
    error = None
    if dist.get_rank() == 0:
        try:
            payload[0] = collect()
        except Exception as exc:
            error = exc
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("Rank zero failed to load HATS metadata for pinned AstroPT3 catalogs") from error
    return payload[0]


def load_hats_metadata(blob: bytes | None) -> dict[str, tuple[Any, Any]]:
    """Decode the trusted rank-zero HATS metadata bundle inside a worker."""
    if blob is None:
        return {}
    try:
        loaded = pickle.loads(zlib.decompress(blob))
    except (ValueError, zlib.error, pickle.UnpicklingError) as exc:
        raise RuntimeError("Invalid AstroPT3 HATS metadata snapshot") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError("Invalid AstroPT3 HATS metadata snapshot")
    return loaded
