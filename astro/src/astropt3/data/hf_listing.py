"""Share immutable Hub directory metadata without caching catalog payloads."""

import json
import re
import zlib
from collections.abc import Iterable

import torch.distributed as dist
from huggingface_hub import HfFileSystem


def share_hf_listing(repos: Iterable[str]) -> bytes | None:
    """List pinned repositories on rank zero; call on every rank before spawning readers."""
    # List the whole revision, including collection siblings such as margin catalogs.
    roots = sorted({
        f"datasets/{repo.split('@')[0]}@{match[1]}"
        for repo in repos if (match := re.search(r"@([a-fA-F0-9]{40})(?:/|$)", repo))
    })
    if not roots:
        return None

    def collect():
        fs = HfFileSystem(skip_instance_cache=True)
        for root in roots:
            fs.find(root)
        # Share only successful checks for immutable revisions, not the unpinned repo entry.
        revisions = [key for key, (exists, _) in fs._repo_and_revision_exists_cache.items()
                     if exists and key[2] is not None]
        # Full recipe trees can be tens of MB; send a compact snapshot to each worker.
        return zlib.compress(json.dumps((fs.dircache, revisions), separators=(",", ":")).encode(), level=1)

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
