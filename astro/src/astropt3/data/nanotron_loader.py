"""Adapter from the LSDB LegacySurvey stream to nanotron micro-batches.

Each DataLoader worker opens the fixed catalog and owns one synchronous
``lsdb.streams.InfiniteStream`` (plain LegacySurvey) or
``lsdb.streams.CrossMatchStream`` (DESI x LegacySurvey, ``crossmatch_desi``).
Items yielded by this module are already whole micro-batches, flattened
because nanotron only moves top-level tensors to the device.
"""

from __future__ import annotations

import gc
import math
import re
import time
import zlib
from collections.abc import Mapping
from contextlib import closing
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx
import numpy as np
import pandas as pd
import torch
from dask.distributed import Client, LocalCluster
from lsdb.loaders.hats.read_hats import open_catalog
from lsdb.streams.catalog_streams import CrossMatchStream, InfiniteStream

from ..configuration_astropt3 import AstroPT3Config
from .band_registry import _DIV_FACTOR
from .outer_crossmatch import OuterKdTreeCrossmatch
from .packing import ObjectSequencer, PackedCollator, span_order
from .spectral import _DIV_FACTOR as _SPECTRA_DIV_FACTOR
from .telemetry import install_byte_probe, instrument

LEGACY_CATALOG = "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north"
DESI_CATALOG = "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"
IMAGE_SHAPE = (3, 152, 152)
_MAX_NET_RETRIES = 60
_MAX_NET_RETRY_WAIT = 120
_MAX_REPLICA_ATTEMPTS = 32
# Lab book (EXPERIMENTS.md Part I) pilot crossmatch radius (mmu_desi_edr_sv3 x mmu_ssl_legacysurvey_north).
_CROSSMATCH_RADIUS_ARCSEC = 1.0
_CROSSMATCH_LEGACY_SUFFIX = "_legacy"
# CrossMatchStream (lsdb PR astronomy-commons/lsdb#1584, selective-x-match)
# skips the Legacy crossmatch for a drawn pixel -- no right-partition fetch
# at all, not even for OuterKdTreeCrossmatch to recover from -- when the
# skymap-density-estimated match fraction falls below this. Was forced to
# 0.0 (never skip) 2026-09-03 for a confirmed upstream KeyError bug in
# _to_delayed (10/336 pixels in our alignment); fixed upstream the same day
# in commit d7302b548 (_to_delayed now builds from build.keys instead of
# indexing pixel_to_key_map, and handles the zero-pixel case). 0.1 showed no
# throughput or composition-mix change vs. no-skip (see wandb run 013174gy
# vs u1kyayfj); 0.5 was benched 2026-09-24 and REJECTED: it saved 31%
# Legacy bytes but cost ~20% end-to-end non-pad tokens/s AND dropped image
# coverage (loss-token share 72% -> 58%) -- the skip removes the image-heavy
# rows while skipped pixels still pay the per-pixel fixed costs, and the
# loader is latency-bound, not bandwidth-bound (EXPERIMENTS.md §9). Set to
# 0.0 on v3 adoption: CrossMatchStream is the pipeline for its multi-
# catalog capability and upstream alignment; re-enable only after lsdb
# moves the skip decision off the loader's critical path.
_CROSSMATCH_COUNT_FRACTION_THRESHOLD = 0.0
# Partitions fetched per InfiniteStream draw. >1 gives each worker a bigger
# ready-buffer to drain, widening the wall-clock window during which the
# background prefetch (see _open_records' dask_client) has a chance to
# finish the NEXT draw before the current one is exhausted -- depth-1
# lookahead (partitions_per_chunk=1) measured no stall_share improvement on
# a real run (bounded by DataLoader prefetch_factor, not partition drain
# time). First tuned WITHOUT OuterKdTreeCrossmatch (astropt3-70m-jetformer,
# DP=2 x 8 workers): 1 -> stall_share ~85%, 1.0k rows/s; 4 -> 65%, 2.0k
# rows/s; 8 -> 46%, 3.3k rows/s (best); 16 -> 62%, 2.3k rows/s (regression,
# RAM past 500GiB). Once OuterKdTreeCrossmatch started recovering
# image-only rows too (outer_crossmatch.py), each chunk got heavier
# (~277KB/recovered image), and 8 regressed the same way 16 previously did
# -- re-tuned: 8 -> stall_share 50%, 2.3k rows/s, RAM 529GiB;
# 4 -> 33%, 3.0k rows/s, RAM 349GiB (best again). Re-tune again if the
# per-record byte weight changes materially (e.g. recovering unmatched
# spectra too, or a bigger image modality).
_PARTITIONS_PER_CHUNK = 4
# nested (struct/list) column -> sub-field names, for map_rows-based decode
_LEGACY_NESTED = {"image": ("band", "flux", "psf_fwhm")}
_CROSSMATCH_NESTED = {
    "spectrum": ("flux", "lambda", "mask"),
    f"image{_CROSSMATCH_LEGACY_SUFFIX}": ("band", "flux", "psf_fwhm"),
}


def hf_config_from_modalities(modalities, **extra) -> AstroPT3Config:
    """Build the HF-side config used by the shared sequencer and collator."""
    return AstroPT3Config(
        modalities=[dict(modality) for modality in modalities],
        **extra,
    )


def flatten_packed_batch(batch: dict, config: AstroPT3Config, seq_len: int) -> dict:
    """Convert ``PackedCollator`` output to nanotron's flat input contract."""
    registry = config.modality_registry()
    batch_size = batch["input_ids"].shape[0]
    flat = {
        "input_ids": batch["input_ids"],
        "position_ids": batch["position_ids"],
    }
    for name in registry.names():
        modality = registry.get_config(name)
        if name in batch["modality_masks"]:
            flat[f"{name}_mask"] = batch["modality_masks"][name]
            flat[f"{name}_values"] = batch["modality_values"][name]
            flat[f"{name}_positions"] = batch["modality_positions"][name]
        else:
            flat[f"{name}_mask"] = torch.zeros(
                (batch_size, seq_len), dtype=torch.bool
            )
            flat[f"{name}_values"] = torch.empty(
                (0, modality.input_size), dtype=torch.float32
            )
            if modality.pos_type == "index":
                flat[f"{name}_positions"] = torch.empty((0,), dtype=torch.long)
            else:
                flat[f"{name}_positions"] = torch.empty(
                    (0, modality.pos_input_size), dtype=torch.float32
                )
    return flat


def regroup_micro_batch(flat: dict, names) -> dict:
    """Convert a flat nanotron micro-batch to HF model keyword arguments."""
    return {
        "input_ids": flat["input_ids"],
        "position_ids": flat["position_ids"],
        "modality_values": {
            name: flat[f"{name}_values"]
            for name in names
            if flat[f"{name}_values"].shape[0]
        },
        "modality_masks": {
            name: flat[f"{name}_mask"]
            for name in names
            if flat[f"{name}_mask"].any()
        },
        "modality_positions": {
            name: flat[f"{name}_positions"]
            for name in names
            if flat[f"{name}_values"].shape[0]
        },
    }


def _as_mapping(value: Any) -> Mapping:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
        if isinstance(value, Mapping):
            return value
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict(orient="list")
        except TypeError:
            value = value.to_dict()
        if isinstance(value, Mapping):
            return value
    raise ValueError(f"image payload has unsupported type {type(value).__name__}")


def _stack_nested(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        return np.stack([_stack_nested(item) for item in array])
    return array


def _finite_scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _row_from_map_rows(mapped: Mapping, nested: dict[str, tuple[str, ...]]) -> dict:
    """Reassemble a ``NestedFrame.map_rows`` dotted-key row into the plain
    nested-mapping shape ``decode_legacy_row``/``decode_crossmatch_row`` expect.

    ``map_rows`` hands nested sub-columns to its callback as numpy arrays
    straight from Arrow storage. ``frame.iterrows()`` instead materializes
    each nested cell as its own per-row pandas DataFrame, which made
    ``_as_mapping``'s ``DataFrame.to_dict()`` fallback the dominant cost of
    crossmatch decode (profiled: ~0.44s of an 0.47s/micro-batch total, from
    boxing every one of a 7781-row spectrum's ~39k cells individually). An
    absent/unmatched nested value (e.g. a crossmatch row with no LegacySurvey
    counterpart) arrives as an empty array, folded to ``None`` here to match
    the old "no image" case.
    """
    row = dict(mapped)
    for name, subfields in nested.items():
        sub: dict = {}
        present = False
        for field in subfields:
            key = f"{name}.{field}"
            if key not in row:
                continue
            value = row.pop(key)
            sub[field] = value
            if isinstance(value, np.ndarray):
                present = present or bool(value.size)
            else:
                present = present or value is not None
        row[name] = sub if present else None
    return row


def _map_rows_columns(frame: Any, nested: dict[str, tuple[str, ...]]) -> list[str]:
    """Column list for ``map_rows``: every plain column plus nested sub-fields.

    Built from ``frame.columns`` rather than the catalog's requested
    projection, so HATS-always-present columns (e.g. ``ra``/``dec``, which
    ``decode_row``'s generic scalar sweep also picks up even when not
    explicitly requested) aren't silently dropped.
    """
    scalar = [name for name in frame.columns if name not in nested]
    subcols = [f"{name}.{field}" for name, fields in nested.items() for field in fields]
    return scalar + subcols


def decode_legacy_row(row: Mapping) -> dict:
    """Decode one LegacySurvey row to the ``ObjectSequencer`` contract.

    Duck-typed: any mapping with ``image``/scalar columns works (pandas
    ``Series``, plain dicts, test fixtures).
    """
    image = _as_mapping(row.get("image"))
    bands = [str(band) for band in image.get("band", ())]
    flux = _stack_nested(image.get("flux")).astype(np.float32, copy=False)
    if flux.shape != IMAGE_SHAPE:
        raise ValueError(f"image flux has shape {flux.shape}, expected {IMAGE_SHAPE}")
    if len(bands) != IMAGE_SHAPE[0]:
        raise ValueError(f"image has {len(bands)} bands, expected {IMAGE_SHAPE[0]}")

    object_id = row.get("object_id")
    if object_id is None:
        raise ValueError("LegacySurvey row has no object_id")
    record = {
        "object_id": str(object_id),
        "image": {"flux": flux, "band": bands},
    }
    for key, value in row.items():
        if key in {"object_id", "image"}:
            continue
        number = _finite_scalar(value)
        if number is not None:
            record[str(key)] = number

    fwhm = image.get("psf_fwhm")
    if fwhm is not None and len(fwhm) == len(bands):
        for band, value in zip(bands, fwhm, strict=True):
            number = _finite_scalar(value)
            if number is not None and 0 < number < 5:
                record[f"psf_fwhm_{band}"] = number
    return record


def _catalog_columns(config: AstroPT3Config, *, include_position: bool = False) -> list[str]:
    """Project the fixed catalog to active Legacy image/scalar fields only."""
    columns = {"object_id"}
    has_image = False
    for name in config.modality_registry().names():
        modality = config.modality_registry().get_config(name)
        if modality.source != "legacy":
            continue
        if modality.family == "image":
            if tuple(modality.record_keys) != ("image",):
                raise ValueError(
                    f"Legacy image modality {name!r} must use record key 'image'"
                )
            has_image = True
            columns.add("image")
        elif modality.family == "scalar":
            for key in modality.record_keys:
                columns.add("image" if key.startswith("psf_fwhm_") else key)
    if not has_image:
        raise ValueError("LSDB training requires an active Legacy image modality")
    if include_position:
        columns |= {"ra", "dec"}
    return ["object_id", "image", *sorted(columns - {"object_id", "image"})]


def _desi_columns(config: AstroPT3Config) -> list[str] | None:
    """Project the fixed DESI catalog to active desi-source fields.

    Returns ``None`` when no modality sources from DESI (the crossmatch
    stream is then skipped entirely).
    """
    columns = {"object_id", "ra", "dec"}
    active = False
    for name in config.modality_registry().names():
        modality = config.modality_registry().get_config(name)
        if modality.source != "desi":
            continue
        active = True
        if modality.family == "spectrum":
            if tuple(modality.record_keys) != ("spectrum",):
                raise ValueError(
                    f"DESI spectrum modality {name!r} must use record key 'spectrum'"
                )
            columns.add("spectrum")
        elif modality.family == "scalar":
            columns.update(modality.record_keys)
            if name == "Z":
                columns.add("ZWARN")
    return sorted(columns) if active else None


def _decode_spectrum(value: Any) -> dict:
    if isinstance(value, pd.DataFrame):
        # A per-record spectrum arrives as one row per wavelength bin
        # (~7781 rows x flux/ivar/lsf_sigma/lambda/mask). The generic
        # _as_mapping -> DataFrame.to_dict(orient="list") path boxes every
        # cell through pandas' maybe_box_native (~39k Python-level boxings
        # per record), which dominates crossmatch decode time end to end
        # (profiled: ~0.44s/micro-batch of an 0.47s total). Column access
        # is vectorized and skips that entirely.
        return {
            "flux": value["flux"].to_numpy(dtype=np.float32, copy=False),
            "lambda": value["lambda"].to_numpy(dtype=np.float32, copy=False),
            "mask": value["mask"].to_numpy(dtype=bool, copy=False),
        }
    spec = _as_mapping(value)
    return {
        "flux": _stack_nested(spec.get("flux")).astype(np.float32, copy=False),
        "lambda": _stack_nested(spec.get("lambda")).astype(np.float32, copy=False),
        "mask": _stack_nested(spec.get("mask")).astype(bool, copy=False),
    }


def decode_crossmatch_row(row: Mapping) -> dict:
    """Decode one ``desi ⋈ legacy`` crossmatch row (ADR 0015 spectra test).

    Most rows carry a spectrum (DESI drives the pixel-level join); the
    legacy image and legacy-sourced scalars (suffixed ``_legacy``) are
    present only when a LegacySurvey counterpart matched within the radius.
    With ``outer_crossmatch.OuterKdTreeCrossmatch``, a row can also be
    Legacy-only (no DESI match at all -- ``object_id`` itself is null),
    recovered from otherwise-discarded bytes; ``object_id`` then falls back
    to ``object_id_legacy``. A null scalar comes through as ``pandas.NA``,
    not ``None`` (confirmed against real crossmatch data), so this checks
    ``pd.isna`` rather than ``is None`` -- that also covers ``None``, so
    plain-dict test fixtures are unaffected.
    """
    object_id = row.get("object_id")
    if bool(pd.isna(object_id)):
        object_id = row.get(f"object_id{_CROSSMATCH_LEGACY_SUFFIX}")
        if bool(pd.isna(object_id)):
            raise ValueError("crossmatched row has no object_id or object_id_legacy")
    record: dict = {"object_id": str(object_id)}

    spectrum_value = row.get("spectrum")
    if spectrum_value is not None:
        record["spectrum"] = _decode_spectrum(spectrum_value)

    image_key = f"image{_CROSSMATCH_LEGACY_SUFFIX}"
    image_value = row.get(image_key)
    fwhm_value = None
    if image_value is not None:
        image = _as_mapping(image_value)
        bands = [str(band) for band in image.get("band", ())]
        flux = _stack_nested(image.get("flux")).astype(np.float32, copy=False)
        if flux.shape == IMAGE_SHAPE and len(bands) == IMAGE_SHAPE[0]:
            record["image"] = {"flux": flux, "band": bands}
            fwhm_value = image.get("psf_fwhm")

    skip = {
        "object_id",
        f"object_id{_CROSSMATCH_LEGACY_SUFFIX}",
        "spectrum",
        image_key,
        "ra",
        "dec",
        f"ra{_CROSSMATCH_LEGACY_SUFFIX}",
        f"dec{_CROSSMATCH_LEGACY_SUFFIX}",
        "_dist_arcsec",
    }
    for key, value in row.items():
        if key in skip:
            continue
        base_key = (
            key.removesuffix(_CROSSMATCH_LEGACY_SUFFIX)
        )
        number = _finite_scalar(value)
        if number is not None:
            record[base_key] = number

    if fwhm_value is not None:
        bands = record["image"]["band"]
        if len(fwhm_value) == len(bands):
            for band, value in zip(bands, fwhm_value, strict=True):
                number = _finite_scalar(value)
                if number is not None and 0 < number < 5:
                    record[f"psf_fwhm_{band}"] = number
    return record


def consumer_seed(seed: int, rank: int, worker: int, retry_generation: int) -> int:
    """Stable independent seed for one DP-rank/worker/fresh-stream consumer."""
    return zlib.crc32(f"{seed}:{rank}:{worker}:{retry_generation}".encode())


_EXCEPTION_AS_DATA_SUBSCRIPT = re.compile(
    r"'(\w*(Error|Exception))' object is not subscriptable"
)


def _retryable(error: Exception) -> bool:
    if isinstance(error, (httpx.HTTPError, OSError, TimeoutError)):
        return True
    if isinstance(error, RuntimeError) and "client has been closed" in str(error):
        return True
    # fsspec's parquet reader feeds cat_ranges() results -- gathered with
    # return_exceptions=True, so transient read errors sit in the list as
    # objects -- straight into KnownPartsOfAFile block data and footer
    # handling without exception checks. The swallowed error then surfaces
    # as a TypeError whose message names the exception class, in one of two
    # shapes seen in the wild: ``can't concat HfHubHTTPError to bytes``
    # (block merge) or ``'HfHubHTTPError' object is not subscriptable``
    # (footer slicing). The original error is unrecoverable (never raised,
    # only used as data), so match the family and let the bounded retry
    # loop reopen the stream. (fsspec 2026.4.0; regressions: v3 run died at
    # step 46,767 2026-09-04; bench arm died at step 302 2026-09-24.)
    if isinstance(error, TypeError):
        message = str(error)
        if "can't concat" in message and "to bytes" in message:
            return True
        return bool(_EXCEPTION_AS_DATA_SUBSCRIPT.search(message))
    return False


def _lsdb_version() -> str:
    try:
        return version("lsdb")
    except PackageNotFoundError:
        return "unknown"


def _log_provenance(
    catalog, columns: list[str], rank: int, worker: int, catalog_desc: str = LEGACY_CATALOG
) -> None:
    info = getattr(getattr(catalog, "hc_structure", None), "catalog_info", None)
    metadata = {}
    for key in ("catalog_name", "catalog_type", "hats_builder", "hats_version"):
        value = getattr(info, key, None)
        if value is not None:
            metadata[key] = str(value)
    suffix = f" provenance={metadata}" if metadata else ""
    print(
        f"[data] dp={rank} worker={worker} lsdb={_lsdb_version()} "
        f"catalog={catalog_desc} columns={columns}{suffix}",
        flush=True,
    )


class PackedMicroBatches(torch.utils.data.IterableDataset):
    """Endless LSDB records packed into fixed-shape nanotron micro-batches."""

    def __init__(
        self,
        config: AstroPT3Config,
        micro_batch_size: int,
        seq_len: int,
        *,
        rank: int = 0,
        seed: int = 0,
        ar_replicas: int = 1,
        replica_placement: str = "decorrelated",
        crossmatch_desi: bool = False,
    ):
        super().__init__()
        if ar_replicas < 1:
            raise ValueError(f"ar_replicas must be >= 1, got {ar_replicas}")
        if replica_placement not in ("decorrelated", "adjacent"):
            raise ValueError(
                "replica_placement must be 'decorrelated' or 'adjacent', "
                f"got {replica_placement!r}"
            )
        if replica_placement == "decorrelated" and ar_replicas > micro_batch_size:
            raise ValueError(
                f"ar_replicas={ar_replicas} exceeds micro_batch_size="
                f"{micro_batch_size}; each replica needs its own packed row"
            )
        self.config = config
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.seed = seed
        self.ar_replicas = ar_replicas
        self.replica_placement = replica_placement
        self.desi_columns = _desi_columns(config) if crossmatch_desi else None
        if crossmatch_desi and self.desi_columns is None:
            raise ValueError(
                "crossmatch_desi=True but no active modality sources from DESI"
            )
        self.columns = _catalog_columns(config, include_position=self.desi_columns is not None)
        self._epoch = 0  # InfiniteStream partition-draw nonce for span ordering
        self.sequencer = ObjectSequencer(config)
        self.collator = PackedCollator(config, seq_len=seq_len)

    def _open_records(self, worker):
        # InfiniteStream's own submit-next-before-returning-current prefetch
        # (catalog_streams.CatalogIterator.__next__) only overlaps with
        # anything when given a real dask client -- with client=None it calls
        # Future.compute() synchronously, so partition N+1's fetch fully
        # blocks the call that returns partition N (profiled: draining a
        # partition's records now costs ~10ms, entirely hidden behind
        # 10-40s network fetches otherwise). processes=False keeps this to
        # in-process threads, not a distributed cluster -- one scheduler +
        # one worker thread per DataLoader worker (~30ms/1.5MiB to start).
        # dashboard_address must go through LocalCluster directly: Client()
        # doesn't forward it to the LocalCluster it builds implicitly, and
        # falls back to :8787. The scheduler's status HTTP server binds a
        # port regardless of dashboard_address (only the bokeh UI routes are
        # actually optional) -- ":0" picks a free one per worker instead of
        # every one of the 8 workers colliding on the fixed default.
        # threads_per_worker > 1 so a >1 partitions_per_chunk draw fetches its
        # partitions concurrently too, not serialized on a single thread.
        dask_cluster = LocalCluster(
            processes=False,
            n_workers=1,
            threads_per_worker=_PARTITIONS_PER_CHUNK,
            dashboard_address=":0",
        )
        executors = [worker.executor for worker in dask_cluster.workers.values()]
        try:
            with dask_cluster, Client(dask_cluster) as dask_client:
                yield from self._stream_records(worker, dask_client)
        finally:
            # Dask's close has a timeout; the job wall-time limit bounds this final join.
            for executor in executors:
                executor.shutdown(wait=True)

    def _stream_records(self, worker, dask_client):
        worker_id = worker.id if worker else 0
        retry_generation = 0
        consecutive_failures = 0
        draw = 0
        while True:
            catalog = stream = iterator = None
            try:
                legacy_catalog = open_catalog(
                    LEGACY_CATALOG, columns=self.columns
                )
                if self.desi_columns is not None:
                    catalog = desi_catalog = open_catalog(
                        DESI_CATALOG, columns=self.desi_columns
                    )
                    catalog_desc = (
                        f"{DESI_CATALOG} x {LEGACY_CATALOG} (CrossMatchStream, "
                        f"count_fraction_threshold={_CROSSMATCH_COUNT_FRACTION_THRESHOLD})"
                    )
                    columns_desc = self.desi_columns + self.columns
                    _log_provenance(
                        catalog, columns_desc, self.rank, worker_id, catalog_desc
                    )
                    stream = CrossMatchStream(
                        desi_catalog,
                        {
                            # CrossMatchStream derives its column suffix from
                            # the right catalog's own .name ("_" +
                            # other.name), not from a suffixes= kwarg (it
                            # overwrites that unconditionally) -- rename to
                            # "legacy" so the suffix lands on "_legacy",
                            # matching _CROSSMATCH_LEGACY_SUFFIX and every
                            # decode function keyed on it. Metadata-only, no
                            # data fetch.
                            "other": legacy_catalog.rename_catalog("legacy"),
                            "algorithm": OuterKdTreeCrossmatch(
                                radius_arcsec=_CROSSMATCH_RADIUS_ARCSEC
                            ),
                        },
                        client=dask_client,
                        partitions_per_chunk=_PARTITIONS_PER_CHUNK,
                        seed=consumer_seed(
                            self.seed, self.rank, worker_id, retry_generation
                        ),
                        count_fraction_threshold=_CROSSMATCH_COUNT_FRACTION_THRESHOLD,
                    )
                else:
                    catalog = legacy_catalog
                    catalog_desc = LEGACY_CATALOG
                    columns_desc = self.columns
                    _log_provenance(
                        catalog, columns_desc, self.rank, worker_id, catalog_desc
                    )
                    stream = InfiniteStream(
                        catalog,
                        client=dask_client,
                        partitions_per_chunk=_PARTITIONS_PER_CHUNK,
                        seed=consumer_seed(
                            self.seed, self.rank, worker_id, retry_generation
                        ),
                    )
                iterator = iter(stream)
            except Exception as error:
                if not _retryable(error):
                    raise
                consecutive_failures += 1
                if consecutive_failures > _MAX_NET_RETRIES:
                    raise
                wait = min(
                    5 * 2 ** (consecutive_failures - 1), _MAX_NET_RETRY_WAIT
                )
                print(
                    f"[data] {type(error).__name__}({str(error)[:200]}): "
                    f"opening a fresh LSDB stream "
                    f"in {wait}s (retry {consecutive_failures}/{_MAX_NET_RETRIES})",
                    flush=True,
                )
                time.sleep(wait)
                retry_generation += 1
                continue

            if iterator is None:
                raise AssertionError("LSDB stream opened without an iterator")
            active_iterator: Any = iterator
            reopen = False
            while not reopen:
                try:
                    frame: Any = next(active_iterator)
                except StopIteration:
                    reopen = True
                    retry_generation += 1
                    continue
                except Exception as error:
                    if not _retryable(error):
                        raise
                    consecutive_failures += 1
                    if consecutive_failures > _MAX_NET_RETRIES:
                        raise
                    wait = min(
                        5 * 2 ** (consecutive_failures - 1), _MAX_NET_RETRY_WAIT
                    )
                    print(
                        f"[data] {type(error).__name__}({str(error)[:200]}): "
                        f"discarding the LSDB "
                        f"iterator and opening a fresh stream in {wait}s "
                        f"(retry {consecutive_failures}/{_MAX_NET_RETRIES})",
                        flush=True,
                    )
                    active_iterator = iter(())
                    iterator = stream = catalog = None
                    gc.collect()
                    time.sleep(wait)
                    retry_generation += 1
                    reopen = True
                    continue

                consecutive_failures = 0
                self._epoch = draw
                draw += 1
                decode_row = (
                    decode_crossmatch_row
                    if self.desi_columns is not None
                    else decode_legacy_row
                )
                nested = (
                    _CROSSMATCH_NESTED
                    if self.desi_columns is not None
                    else _LEGACY_NESTED
                )
                # map_rows delivers nested sub-columns as numpy arrays (no
                # per-row DataFrame materialization), which is what makes
                # this ~2.5-4x faster end to end than frame.iterrows() +
                # decode_row -- see _row_from_map_rows.
                decoded = frame.map_rows(
                    lambda mapped, decode_row=decode_row, nested=nested: {
                        "record": decode_row(_row_from_map_rows(mapped, nested))
                    },
                    columns=_map_rows_columns(frame, nested),
                    infer_nesting=False,
                )
                yield from decoded["record"]

    def _replica_objects(self, record: dict) -> list:
        """Build the base sequence and distinct autoregressive reorderings."""
        base_id = str(record.get("object_id", ""))
        first = self.sequencer.build(record, epoch=self._epoch)
        names = sorted(first.order)
        if self.ar_replicas == 1 or len(names) < 2:
            return [first]

        wanted = min(self.ar_replicas, math.factorial(len(names)))
        objects, seen = [first], {first.order}
        attempt = 1
        while len(objects) < wanted and attempt <= _MAX_REPLICA_ATTEMPTS:
            replica_id = f"{base_id}#{attempt}"
            order = tuple(span_order(names, replica_id, self._epoch))
            attempt += 1
            if order in seen:
                continue
            seen.add(order)
            objects.append(
                self.sequencer.build(
                    {**record, "object_id": replica_id},
                    epoch=self._epoch,
                    modality_order=list(order),
                )
            )
        return objects

    def _place(self, objects: list, used: list[int]) -> list[int] | None:
        """Assign a record's base sequence and replicas to packed rows."""
        order = sorted(range(len(used)), key=lambda row: (used[row], row))
        if self.replica_placement == "adjacent":
            span = sum(len(obj) for obj in objects)
            for row_index in order:
                if used[row_index] + span <= self.seq_len:
                    return [row_index] * len(objects)
            return None
        chosen: list[int] = []
        for obj in objects:
            for row_index in order:
                if row_index not in chosen and used[row_index] + len(obj) <= self.seq_len:
                    chosen.append(row_index)
                    break
            else:
                return None
        return chosen

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        install_byte_probe(self.rank, worker.id if worker else None)
        with closing(self._open_records(worker)) as records:
            rows: list[list] = [[] for _ in range(self.micro_batch_size)]
            used = [0] * self.micro_batch_size

            for record in records:
                objects = self._replica_objects(record)
                for obj in objects:
                    if len(obj) > self.seq_len:
                        raise ValueError(
                            f"object of length {len(obj)} exceeds seq_len {self.seq_len}"
                        )
                placement = self._place(objects, used)
                if placement is None:
                    batch = self.collator.collate_rows(rows)
                    expected_shape = (self.micro_batch_size, self.seq_len)
                    if batch["input_ids"].shape != expected_shape:
                        raise AssertionError(
                            f"packing mismatch: {batch['input_ids'].shape} != {expected_shape}"
                        )
                    yield flatten_packed_batch(batch, self.config, self.seq_len)
                    rows = [[] for _ in range(self.micro_batch_size)]
                    used = [0] * self.micro_batch_size
                    placement = self._place(objects, used)
                    if placement is None:
                        raise AssertionError(
                            f"record {record.get('object_id')!r} cannot be placed "
                            f"into {self.micro_batch_size} rows x {self.seq_len} tokens"
                        )
                for obj, row_index in zip(objects, placement, strict=True):
                    rows[row_index].append(obj)
                    used[row_index] += len(obj)


def build_astropt3_dataloader(
    dataset_args,
    model_config,
    micro_batch_size: int,
    sequence_length: int,
    dp_rank: int,
    dp_size: int,
    num_workers: int = 0,
    seed: int = 0,
    multiprocessing_context: str | None = None,
) -> Any:
    """Build the plain DataLoader used by nanotron's astropt3 dataset type."""
    config = hf_config_from_modalities(
        model_config.modalities,
        **{
            field: getattr(model_config, field, default)
            for field, default in [
                ("jetformer_flow_steps", 4),
                ("jetformer_flow_hidden", 128),
                ("jetformer_gmm_k", 4),
                ("jetformer_noise_max", 0.1),
                ("jetformer_noise_min", 0.0),
                ("scalar_gmm_k", 5),
                ("image_norm_divisor", _DIV_FACTOR),
                ("spectra_norm_divisor", _SPECTRA_DIV_FACTOR),
                ("spiral", True),
            ]
        },
    )
    dataset = PackedMicroBatches(
        config,
        micro_batch_size,
        sequence_length,
        rank=dp_rank,
        seed=seed,
        ar_replicas=getattr(dataset_args, "ar_replicas", 1) or 1,
        replica_placement=getattr(dataset_args, "replica_placement", None)
        or "decorrelated",
        crossmatch_desi=getattr(dataset_args, "crossmatch_desi", False),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context=multiprocessing_context,
    )
    return instrument(loader)
