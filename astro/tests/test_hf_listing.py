"""Exercise pinned Hub listings and broadcast HATS metadata through real LSDB reads."""

from functools import partial
from types import SimpleNamespace

import hats as hc
import httpx
import pandas as pd
import pytest
from huggingface_hub import HfFileSystem, set_client_factory
from huggingface_hub.utils import _http
from lsdb.loaders.dataframe.from_dataframe import from_dataframe

from astropt3.configuration_astropt3 import DEFAULT_MODALITIES
from astropt3.data import nanotron_loader as loader
from astropt3.data.hf_listing import share_hats_metadata, share_hf_listing

SHA = "a" * 40
REPO = "offline/catalog"


def _hub_client(root, requests, *, list_tree=False):
    def respond(request):
        path = request.url.path
        with requests.open("a") as log:
            log.write(f"{path}\n")
        if list_tree and path == f"/api/datasets/{REPO}/revision/{SHA}":
            return httpx.Response(200, json={"id": REPO, "sha": SHA})
        if list_tree and path == f"/api/datasets/{REPO}/tree/{SHA}":
            assert request.url.params["recursive"] == "true"
            return httpx.Response(200, json=[
                {"path": file.relative_to(root).as_posix(), "oid": SHA,
                 "type": "directory" if file.is_dir() else "file", "size": file.stat().st_size}
                for file in sorted(root.rglob("*"))
            ])
        prefix = f"/datasets/{REPO}/resolve/{SHA}/"
        assert path.startswith(prefix), f"Unexpected Hub request: {path}"
        payload = (root / path.removeprefix(prefix)).read_bytes()
        if range_header := request.headers.get("range"):
            start, end = map(int, range_header.removeprefix("bytes=").split("-"))
            return httpx.Response(206, content=payload[start:end + 1])
        return httpx.Response(200, content=payload)

    return httpx.Client(transport=httpx.MockTransport(respond))


def _install_worker_hub(root, requests, _worker_id):
    set_client_factory(partial(_hub_client, root, requests))


def test_training_worker_reads_two_dirs_without_hub_api_or_metadata_calls(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    requests = tmp_path / "requests"
    local = from_dataframe(
        pd.DataFrame({"ra": [10.0, 170.0], "dec": [0.0, 20.0], "object_id": ["x", "y"], "dust": [0.1, 0.2]}),
        lowest_order=6, catalog_name="dust", margin_threshold=None,
    )
    local.write_catalog(root / "train", as_collection=False, progress_bar=False, should_write_skymap=True)
    url = f"{REPO}@{SHA}/train"
    spec = {"anchors": [{"repo": url, "name": "dust", "decoder": "scalars", "draw_weight": 1,
                         "scalar_columns": {"ebv": "dust"}}], "catalogs": []}
    model = SimpleNamespace(modalities=[m for m in DEFAULT_MODALITIES if m["name"] == "ebv"])
    monkeypatch.delenv("ASTROPT3_TELEMETRY_DIR", raising=False)
    with monkeypatch.context() as patch, _hub_client(root, requests, list_tree=True) as client:
        patch.setattr(loader.dist, "is_initialized", lambda: True)
        patch.setattr(loader.dist, "get_world_size", lambda: 1)
        patch.setattr(_http, "_GLOBAL_CLIENT", client)
        batches = loader.build_astropt3_dataloader(
            SimpleNamespace(catalog_stream=spec), model, 1, 8, 0, 1, num_workers=1,
        )
    rank0_requests = requests.read_text().splitlines()
    assert f"/api/datasets/{REPO}/revision/{SHA}" in rank0_requests
    assert f"/api/datasets/{REPO}/tree/{SHA}" in rank0_requests
    assert any("point_map.fits" in path for path in rank0_requests)
    requests.unlink()
    batches.worker_init_fn = partial(_install_worker_hub, root, requests)
    batches.timeout = 30
    iterator = iter(batches)
    try:
        assert next(iterator)["ebv_values"].shape == (2, 1)
    finally:
        iterator._shutdown_workers()
    worker_requests = requests.read_text().splitlines()
    assert worker_requests
    assert all(f"/resolve/{SHA}/" in path for path in worker_requests)
    # Workers read ONLY partition parquet files — zero point_map.fits or metadata fetches.
    assert all(path.endswith(".parquet") and "/Dir=" in path for path in worker_requests)
    assert len({path.rsplit("/", 1)[0] for path in worker_requests}) == 2


def test_listing_only_queries_distinct_commit_pins(monkeypatch):
    paths = []
    monkeypatch.setattr(HfFileSystem, "find", lambda self, path: paths.append(path))
    assert share_hf_listing([REPO, f"{REPO}@main", f"{REPO}@refs/pr/12"]) is None
    assert share_hf_listing([f"{REPO}@{SHA}/train", f"{REPO}@{SHA}/other"])
    assert paths == [f"datasets/{REPO}@{SHA}"]


def test_listing_failure_is_published(monkeypatch):
    monkeypatch.setattr(loader.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(loader.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(loader.dist, "get_rank", lambda: 0)
    calls = []
    monkeypatch.setattr(loader.dist, "broadcast_object_list", lambda listing, *, src: calls.append((listing, src)))

    def fail(self, path):
        raise OSError("Hub unavailable")

    monkeypatch.setattr(HfFileSystem, "find", fail)
    with pytest.raises(RuntimeError, match="Rank zero failed to list") as error:
        share_hf_listing([f"{REPO}@{SHA}"])
    assert isinstance(error.value.__cause__, OSError)
    assert calls == [([None], 0)]


def test_hats_metadata_skips_unpinned_and_publishes_failure(monkeypatch):
    assert share_hats_metadata([REPO, f"{REPO}@main", f"{REPO}@refs/pr/12"]) is None
    monkeypatch.setattr(loader.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(loader.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(loader.dist, "get_rank", lambda: 0)
    calls = []
    monkeypatch.setattr(loader.dist, "broadcast_object_list", lambda payload, *, src: calls.append((payload, src)))

    def fail(*args, **kwargs):
        raise OSError("Hub metadata unavailable")

    monkeypatch.setattr(hc, "read_hats", fail)
    with pytest.raises(RuntimeError, match="Rank zero failed to load HATS metadata") as error:
        share_hats_metadata([f"{REPO}@{SHA}"])
    assert isinstance(error.value.__cause__, OSError)
    assert calls == [([None], 0)]
