"""Configured MMU catalogs -> finite decoded draws, independent of modality count."""

from __future__ import annotations

import copy
import inspect
import io
import math
import re
import time
from contextlib import suppress
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from dask._task_spec import Task, TaskRef
from dask.delayed import Delayed, delayed
from dask.distributed import Client, LocalCluster
from hats.pixel_math.healpix_pixel import HealpixPixel
from lsdb.loaders.hats.read_hats import open_catalog
from lsdb.streams.catalog_streams import (
    CrossMatchStream,
    InfiniteStream,
    _assemble_star_pixel,
    _data_coverage,
    _partition_delayeds,
    _restrict_to_footprint,
)

_LSDB_CROSSMATCH_STREAM = CrossMatchStream
_MAX_ROWGROUP_FILTERS = 10
_ORDER0_PIXEL_ARCSEC = 210_962.0  # sqrt(4*pi / 12) in arcseconds
from mocpy import MOC
from PIL import Image

from . import nanotron_loader as loader
from . import telemetry
from .hf_io import enable_hf_coalesced_parquet_reads, enable_hf_concurrent_ranges
from .hf_listing import load_hats_metadata, seed_hf_listing
from .packing import has_modality
from .scalar_registry import CATEGORICAL, QUALITY_GATES, _transforms

# Decoder names describe verified payload contracts, not arbitrary calibrations.
_PROFILES = {
    "legacy_image": ("image", ("band", "flux"), "image"),
    "legacy_south_image": ("image", ("band", "flux"), "image"),
    "hsc_image": ("image", ("band", "flux", "scale"), "image"),
    "desi_spectrum": ("spectrum", ("flux", "lambda", "mask"), "spectrum"),
    "sdss_spectrum": ("spectrum", ("flux", "lambda", "mask"), "spectrum"),
    "scalars": (None, (), "scalar"),
    # 8-bit RGB PNG/JPEG struct {bytes, path} (e.g. galaxies-with-hats); dequantized to (k + U(0,1)) / 255
    "rgb_png": ("image", (), "image"),
}
_IMAGE_SOURCE_DECODERS = {
    "legacy_north": "legacy_image",
    "legacy_south": "legacy_south_image",
    "hsc": "hsc_image",
    "galaxies_with_hats": "rgb_png",
}
# Catalogs whose identifier column is not ``object_id``.
_ID_COLUMNS = {"rgb_png": "dr8_id"}
# OS-seeded per process, so every DataLoader worker draws independent noise.
_DEQUANTIZE = np.random.default_rng()
_DEFAULTS = {
    "how": "outer", "radius_arcsec": 1.0,
    "partitions_per_chunk": 4, "worker_memory_limit": "auto",
}


def _unregister_hf_array2d_extension() -> None:
    """Keep HSC's Arrow schema readable when Hugging Face ``datasets`` is imported.

    ``datasets`` (imported by nanotron) registers ``Array2DExtensionType``
    globally. HSC stores that metadata on its image fields, and LSDB's
    ``Schema.empty_table()`` cannot build empty extension arrays
    (``ArrowNotImplementedError: extension``). Spawned loader workers never
    import ``datasets``; the in-process path (``num_loading_workers: 0``) does.
    Dropping the registration makes Arrow read the plain list storage type.
    """
    with suppress(pa.ArrowKeyError):
        pa.unregister_extension_type("datasets.features.features.Array2DExtensionType")


class _Star:
    """One anchor + partners: LSDB column projection and row decoder."""

    def __init__(self, catalogs, registry):
        self.catalogs = catalogs
        self.registry = registry
        self.nested: dict[str, tuple[str, ...]] = {}
        self.columns = []
        self.outputs = outputs = {}
        reserved = {"object_id", "anchor", "anchor_path", "source_ids"}
        names = set()
        for index, catalog in enumerate(self.catalogs):
            if not isinstance(catalog, dict) or set(catalog) - {
                "name", "repo", "decoder", "record_key", "scalar_columns", "how",
            }:
                raise ValueError("catalog entry has unknown fields or is not a mapping")
            name = catalog.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in names:
                raise ValueError("catalog names must be unique identifiers")
            names.add(name)
            repo = catalog.get("repo")
            if not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+(?:@[\w./-]+)?", repo):
                raise ValueError(f"catalog {name!r} needs a Hugging Face namespace/name[@revision]")
            decoder = catalog.get("decoder")
            if not isinstance(decoder, str) or decoder not in _PROFILES:
                raise ValueError(f"catalog {name!r} has unsupported decoder {decoder!r}")
            payload, fields, family = _PROFILES[decoder]
            suffix = "" if index == 0 else f"_{name}"
            self.columns.append(f"{_ID_COLUMNS.get(decoder, 'object_id')}{suffix}")
            if payload:
                key = catalog.get("record_key")
                if not isinstance(key, str) or not key or key in reserved or key in outputs:
                    raise ValueError("catalog record keys must be nonempty and unique")
                outputs[key] = (family, decoder, name)
                column = f"{payload}{suffix}"
                if fields:
                    self.nested[column] = fields
                    self.columns.extend(f"{column}.{field}" for field in fields)
                else:
                    self.columns.append(column)  # plain struct column, read whole
            elif "record_key" in catalog:
                raise ValueError("scalar-only catalogs use scalar_columns, not record_key")
            scalars = catalog.setdefault("scalar_columns", {})
            if not isinstance(scalars, dict) or (not payload and not scalars):
                raise ValueError("scalar_columns must map record keys to flat catalog columns")
            for key, column in scalars.items():
                if (not isinstance(key, str) or not key or key in reserved or key in outputs
                        or not isinstance(column, str) or not column or "." in column
                        or column in {"image", "spectrum"}):
                    raise ValueError("scalar_columns needs unique record keys and flat column names")
                outputs[key] = ("scalar", decoder, name)
                self.columns.append(f"{column}{suffix}")
        self.columns = list(dict.fromkeys(self.columns))

    @staticmethod
    def validate_outputs(registry, outputs):
        if not registry.names():
            raise ValueError("at least one model modality is required")
        for name in registry.names():
            modality = registry.get_config(name)
            if modality.family == "scalar":
                _transforms(name)  # Reject uncalibrated scalars before opening catalogs.
                if modality.categories != CATEGORICAL.get(name, 0):
                    raise ValueError(f"modality {name!r} categories disagree with scalar_registry.CATEGORICAL")
            decoder = None
            if modality.family == "image":
                decoder = _IMAGE_SOURCE_DECODERS.get(modality.source or "")
            elif modality.family == "spectrum":
                decoder = f"{modality.source}_spectrum"
            for key in modality.record_keys:
                if key not in outputs or outputs[key][0] != modality.family:
                    raise ValueError(f"modality {name!r} has no compatible catalog output for {key!r}")
                if decoder is not None and outputs[key][1] != decoder:
                    raise ValueError(f"modality {name!r} source disagrees with its {modality.family} decoder")
            for flag in QUALITY_GATES.get(name, ()):
                if flag not in outputs or any(
                    outputs[key][2] != outputs[flag][2] for key in modality.record_keys
                ):
                    raise ValueError(f"{name} requires an explicit {flag} scalar column from the same catalog")

    def decode_row(self, mapped):
        row = loader._row_from_map_rows(mapped, self.nested)
        record, source_ids = {}, {}
        for index, catalog in enumerate(self.catalogs):
            name, decoder = catalog["name"], catalog["decoder"]
            payload, _, _ = _PROFILES[decoder]
            suffix = "" if index == 0 else f"_{name}"
            object_id = row.get(f"{_ID_COLUMNS.get(decoder, 'object_id')}{suffix}")
            if not bool(pd.isna(object_id)):
                source_ids[name] = str(object_id)
            value = row.get(f"{payload}{suffix}") if payload else None
            if value is not None and not isinstance(value, dict) and pd.isna(value):
                value = None  # a left-joined struct column with no match arrives as NA
            scalars = {
                key: loader._finite_scalar(row.get(f"{column}{suffix}"))
                for key, column in catalog["scalar_columns"].items()
            }
            scalars = {key: number for key, number in scalars.items() if number is not None}
            if value is None and not scalars:
                continue
            if name not in source_ids:
                raise ValueError(f"{name} row has no object_id")
            record.update(scalars)
            if value is None:
                continue
            if decoder == "legacy_image":
                result = loader.decode_legacy_image(value)
            elif decoder == "legacy_south_image":
                result = loader.decode_legacy_south_image(value)
            elif decoder == "hsc_image":
                flux = loader._stack_nested(value.get("flux")).astype(np.float32, copy=False)
                bands = [str(band) for band in value.get("band", ())]
                if flux.shape != (5, 160, 160) or not np.isfinite(flux).all():
                    raise ValueError("HSC Wide flux must be a finite (5, 160, 160) cube")
                if bands != [f"hsc-{band}" for band in "grizy"]:
                    raise ValueError("HSC Wide bands must be ordered hsc-g/r/i/z/y")
                scale = np.asarray(value.get("scale"), dtype=np.float32)
                if scale.shape != (5,) or not np.allclose(scale, 0.168, rtol=1e-6, atol=0):
                    raise ValueError("HSC Wide pixel scale must be 0.168 arcsec in all five bands")
                result = {"flux": flux, "band": bands}
            elif decoder == "rgb_png":
                with Image.open(io.BytesIO(value["bytes"])) as png:
                    rgb = np.asarray(png.convert("RGB"), dtype=np.float32)
                # Uniform dequantization: a continuous likelihood on exact k/255 levels
                # rewards piling density onto the 8-bit grid instead of modelling galaxies.
                rgb = (rgb + _DEQUANTIZE.random(rgb.shape, dtype=np.float32)) / 255.0
                result = {"flux": rgb.transpose(2, 0, 1), "band": ["rgb-r", "rgb-g", "rgb-b"]}
            else:
                result = loader._decode_spectrum(value)
                if decoder == "sdss_spectrum":
                    # Nonpositive SDSS wavelengths are padding; NaNs remain validation errors.
                    keep = ~(result["lambda"] <= 0)
                    result = {key: values[keep] for key, values in result.items()}
            record[catalog["record_key"]] = result
        if not source_ids:
            raise ValueError("crossmatched row has no supported data or object_id")
        if any(has_modality(self.registry.get_config(name), record) for name in self.registry.names()):
            first = next(iter(source_ids))
            path = self.catalogs[0]["name"]
            record.update(object_id=f"{first}:{source_ids[first]}", source_ids=source_ids,
                          anchor=path if path in source_ids else None, anchor_path=path)
            return {"record": record}
        return {"record": None}


@lru_cache(maxsize=2)
def _split_sky(split):
    # A fixed all-sky holdout, independent of catalog IDs and the training seed.
    pixels = np.random.default_rng(58).permutation(12 * 4**4)[:154].astype(np.uint64)
    validation = MOC.from_healpix_cells(pixels, np.full(len(pixels), 4, dtype=np.uint8), 4)
    # One order-10 pixel collar (a few arcmin) separates cutouts across the split.
    return validation if split == "validation" else validation.to_order(10).extended().complement()


def _shard_sky(anchor, footprint, index, count):
    # Whole HATS partitions (similar row counts) per consumer, so no two consumers fetch
    # the same anchor file; more consumers than partitions share them round-robin.
    # shortcut: no seam collar, so an outer partner row within radius_arcsec of a shard
    # seam can appear in two shards; add a guard like the anchors' if that matters.
    pixels = anchor.moc_search(footprint).get_healpix_pixels()
    step = min(count, len(pixels))
    if not step:
        return footprint
    mine = pixels[index % step::step]
    orders = np.array([p.order for p in mine], dtype=np.uint8)
    cells = MOC.from_healpix_cells(np.array([p.pixel for p in mine], dtype=np.uint64), orders, int(orders.max(initial=0)))
    return footprint.intersection(cells)


def _project_anchor_margin(anchor):
    """Strip payload columns from an anchor's margin catalog.

    In outer crossmatches, LSDB only uses ``left.margin`` coordinates to suppress
    unmatched right rows near a partition seam; every visitor match is dropped before
    output assembly, so payload columns on ``anchor.margin`` are fetched only to be
    discarded.
    """
    margin = getattr(anchor, "margin", None)
    if (
        margin is None
        or not hasattr(margin, "hc_structure")
        or not getattr(margin._operation, "is_reloadable", False)
        or margin.hc_structure.catalog_base_dir is None
        or margin.hc_structure.original_schema is None
    ):
        return anchor
    info = margin.hc_structure.catalog_info
    coords = [col for col in (info.ra_column, info.dec_column) if col]
    if not coords or list(margin.columns) == coords:
        return anchor
    import hats as hc
    from lsdb.loaders.hats.hats_loading_config import HatsLoadingConfig
    from lsdb.loaders.hats.read_hats import _load_catalog

    old = getattr(margin, "loading_config", None)
    hc_margin = margin._create_modified_hc_structure(updated_schema=margin.hc_structure.original_schema)
    config = HatsLoadingConfig(
        columns=coords,
        margin_cache=None,
        error_empty_filter=False,
        path_generator=old.path_generator if old and old.path_generator else hc.io.pixel_catalog_file,
        filters=old.filters if old else None,
        kwargs=dict(old.kwargs) if old and old.kwargs else {},
        show_statistics=old.show_statistics if old else False,
    )
    anchor.margin = _load_catalog(hc_margin, config)
    return anchor


def _open_from_hats_metadata(hc_catalog, margin_hc_catalog, *, columns, margin_columns=None):
    """Build an LSDB Catalog directly from broadcast HATS metadata without Hub reads."""
    import hats as hc
    from lsdb.catalog.margin_catalog import _validate_margin_catalog
    from lsdb.loaders.hats.read_hats import _read_dataset

    if isinstance(hc_catalog, hc.catalog.CatalogCollection):
        catalog = _read_dataset(hc_catalog.main_catalog, columns=columns)
        catalog.hc_collection = hc_catalog
    else:
        catalog = _read_dataset(hc_catalog, columns=columns)
    if margin_hc_catalog is not None:
        margin = _read_dataset(margin_hc_catalog, columns=margin_columns or columns)
        if margin_columns is None:
            _validate_margin_catalog(margin, catalog)
        catalog.margin = margin
    return catalog


def _healpix_29_filters_for_pixel(
    pixel: HealpixPixel, radius_arcsec: float, index_column: str = "_healpix_29"
) -> list[list[tuple[str, str, int]]]:
    """Return DNF ``_healpix_29`` row-group bounds covering ``pixel`` plus a >=4x collar."""
    collar_order = int(math.floor(math.log2(_ORDER0_PIXEL_ARCSEC / max(4.0 * radius_arcsec, 1.0))))
    collar_order = max(pixel.order, min(14, collar_order))
    moc = MOC.from_healpix_cells(
        np.array([pixel.pixel], dtype=np.uint64),
        np.array([pixel.order], dtype=np.uint8),
        pixel.order,
    ).to_order(collar_order).extended()
    ranges = moc.to_depth29_ranges
    if len(ranges) > _MAX_ROWGROUP_FILTERS:
        starts, ends = ranges[:, 0], ranges[:, 1]
        gaps = starts[1:] - ends[:-1]
        cuts = np.sort(
            np.argpartition(gaps, -(_MAX_ROWGROUP_FILTERS - 1))[-(_MAX_ROWGROUP_FILTERS - 1):]
        )
        merged = []
        prev = 0
        for cut in cuts:
            merged.append((int(starts[prev]), int(ends[cut])))
            prev = int(cut) + 1
        merged.append((int(starts[prev]), int(ends[-1])))
        return [[(index_column, ">=", lo), (index_column, "<", hi)] for lo, hi in merged]
    return [[(index_column, ">=", int(lo)), (index_column, "<", int(hi))] for lo, hi in ranges]


def _remap_task_refs(obj: Any, key_map: dict) -> Any:
    if isinstance(obj, TaskRef):
        return TaskRef(key_map.get(obj.key, obj.key))
    if isinstance(obj, tuple):
        return tuple(_remap_task_refs(item, key_map) for item in obj)
    if isinstance(obj, list):
        return [_remap_task_refs(item, key_map) for item in obj]
    if isinstance(obj, dict):
        return {k: _remap_task_refs(v, key_map) for k, v in obj.items()}
    return obj


def _push_coarse_pixel_filters(
    part: Delayed, target_pixel: HealpixPixel, radius_arcsec: float
) -> Delayed:
    """Push ``target_pixel``'s collared ``_healpix_29`` bounds into coarser ``read_pixel`` tasks."""
    graph = dict(part.dask)
    rewritten: dict[Any, list[list[tuple[str, str, int]]]] = {}
    cached_filters: dict[str, list[list[tuple[str, str, int]]]] = {}
    for key, node in graph.items():
        if not isinstance(node, Task) or not node.args:
            continue
        part_pixel = node.args[0]
        if not isinstance(part_pixel, HealpixPixel) or part_pixel.order >= target_pixel.order:
            continue
        if "filters" not in node.kwargs:
            continue
        index_col = node.kwargs.get("index_column")
        schema = node.kwargs.get("schema")
        if index_col != "_healpix_29" or schema is None or index_col not in schema.names:
            continue
        if index_col not in cached_filters:
            cached_filters[index_col] = _healpix_29_filters_for_pixel(
                target_pixel, radius_arcsec, index_column=index_col
            )
        rewritten[key] = cached_filters[index_col]

    if not rewritten:
        return part

    # Propagate rekeying to every downstream task so sibling pixels in one chunk never collide.
    dirty = set(rewritten)
    changed = True
    while changed:
        changed = False
        for key, node in graph.items():
            if key not in dirty and isinstance(node, Task) and (node.dependencies & dirty):
                dirty.add(key)
                changed = True

    tag = (target_pixel.order, target_pixel.pixel)
    key_map = {key: (key, tag) for key in dirty}
    new_graph = {}
    for key, node in graph.items():
        if key not in dirty:
            new_graph[key] = node
            continue
        new_key = key_map[key]
        new_args = _remap_task_refs(node.args, key_map)
        new_kwargs = _remap_task_refs(node.kwargs, key_map)
        if key in rewritten:
            new_kwargs = {**new_kwargs, "filters": rewritten[key]}
        new_graph[new_key] = Task(new_key, node.func, *new_args, **new_kwargs)
    return Delayed(key_map.get(part.key, part.key), new_graph)


class _FilteredCrossMatchStream(CrossMatchStream):
    """CrossMatchStream that pushes fine aligned-pixel bounds into coarser Parquet reads."""

    def __init__(self, *args, radius_arcsec: float = 1.0, **kwargs):
        self._pushdown_radius_arcsec = radius_arcsec
        super().__init__(*args, **kwargs)

    def _delayed_partition(self, pixel: HealpixPixel) -> Delayed:
        right_parts = [
            [_push_coarse_pixel_filters(p, pixel, self._pushdown_radius_arcsec) for p in _partition_delayeds(result, pixel)]
            for result in self._results
        ]
        anchor_parts = [
            _push_coarse_pixel_filters(p, pixel, self._pushdown_radius_arcsec)
            for p in _partition_delayeds(self._anchor, pixel)
        ]
        return delayed(_assemble_star_pixel)(pixel, anchor_parts, right_parts, self._assembly)


def _draw_paths(weights, offset):
    while True:
        for turn in range(max(weights)):
            for index in range(len(weights)):
                path = (index + offset) % len(weights)
                if turn < weights[path]:
                    yield path


class CatalogStream:
    """Compose LSDB stars in disjoint sky, with a deterministic weighted draw cycle."""

    def __init__(self, spec, config):
        if not isinstance(spec, dict) or set(spec) - {*_DEFAULTS, "anchors", "catalogs", "min_catalogs", "split"}:
            raise ValueError("catalog_stream has unknown settings or is not a mapping")
        # Concatenating multiple unjoined image partitions can exceed Arrow's offset limit.
        chunk_default = 1 if spec.get("catalogs") == [] else _DEFAULTS["partitions_per_chunk"]
        self.spec = spec = {
            **_DEFAULTS, "min_catalogs": 1, "partitions_per_chunk": chunk_default, **copy.deepcopy(spec),
        }
        anchors = spec.get("anchors")
        if not isinstance(anchors, list) or not anchors or not all(isinstance(c, dict) for c in anchors):
            raise ValueError("catalog_stream.anchors must be a nonempty list of catalogs with draw_weight")
        self.weights = [c.get("draw_weight") for c in anchors]
        if any(type(w) is not int or w < 1 for w in self.weights):
            raise ValueError("draw_weight must be a positive integer for every anchor")
        anchors = [{k: v for k, v in c.items() if k != "draw_weight"} for c in anchors]
        partners = spec.get("catalogs")
        if not isinstance(partners, list):
            raise ValueError("catalog_stream.catalogs must be a list of partner catalogs")
        self.catalogs = [*anchors, *partners]
        self.hf_listing: bytes | None = None
        self.hats_metadata: bytes | None = None
        registry = config.modality_registry()
        self.stars = [_Star([anchor, *partners], registry) for anchor in anchors]
        if len({c['name'] for c in self.catalogs}) != len(self.catalogs):
            raise ValueError("catalog names must be unique identifiers")
        outputs = {}
        for star in self.stars:
            for key, contract in star.outputs.items():
                if key in outputs and outputs[key] != contract:
                    raise ValueError(f"ambiguous catalog output {key!r}; use source-distinct record keys")
                outputs[key] = contract
        _Star.validate_outputs(registry, outputs)
        if spec['how'] not in ('left', 'outer') or any(c.get('how', spec['how']) not in ('left', 'outer') for c in self.catalogs):
            raise ValueError("catalog_stream.how must be left or outer")
        minimum = spec['min_catalogs']
        outer = sum(c.get('how', spec['how']) == 'outer' for c in partners)
        if type(minimum) is not int or not 1 <= minimum <= 1 + outer:
            raise ValueError("min_catalogs must be in 1..number of anchor + outer catalogs")
        radius = spec['radius_arcsec']
        if type(radius) not in (int, float) or not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius_arcsec must be a positive finite number")
        if 'split' in spec and spec['split'] not in ('train', 'validation'):
            raise ValueError("split must be train or validation")
        if (len(anchors) > 1 or 'split' in spec) and radius > 1:
            raise ValueError("buffered footprints require radius_arcsec <= 1")
        if type(spec['partitions_per_chunk']) is not int or spec['partitions_per_chunk'] < 1:
            raise ValueError("partitions_per_chunk must be a positive integer")
        if not isinstance(spec['worker_memory_limit'], str) or not spec['worker_memory_limit']:
            raise ValueError("worker_memory_limit must be a Dask memory-limit string")
        if partners and 'min_catalogs' not in inspect.signature(CrossMatchStream).parameters:
            raise ValueError("configured catalog streams require LSDB's multi-catalog/min_catalogs API")

    def _streams(self, opened, client, seed, shard=(0, 1)):
        claimed = None
        streams = []
        minimum = self.spec['min_catalogs']
        for i, star in enumerate(self.stars):
            root, *rights = star.catalogs
            anchor = opened[root['name']]
            joins = [{'other': opened[c['name']], 'how': c.get('how', self.spec['how']),
                      'suffixes': ('', f"_{c['name']}"), 'radius_arcsec': self.spec['radius_arcsec']}
                     for c in rights]
            outer = [j['other'] for j in joins if j['how'] == 'outer']
            footprint = None
            if minimum > 1 and len(outer) != len(rights):
                _, footprint = _restrict_to_footprint(anchor, None, outer, minimum)
            if len(self.stars) > 1 or 'split' in self.spec or shard[1] > 1:
                if footprint is None:
                    if minimum > 1:
                        _, footprint = _restrict_to_footprint(anchor, None, outer, minimum)
                    else:
                        footprint = _data_coverage(anchor)
                if claimed is not None:
                    footprint = footprint.difference(claimed)
                if i < len(self.stars) - 1:
                    # A one-cell collar is wider than two 1-arcsec matches, so a partner
                    # cannot attach on both sides of an ownership seam.
                    guard = footprint.to_order(14).extended()
                    claimed = guard if claimed is None else claimed.union(guard)
                if 'split' in self.spec:
                    footprint = footprint.intersection(_split_sky(self.spec['split']))
                if shard[1] > 1:
                    footprint = _shard_sky(anchor, footprint, *shard)
                if footprint.empty():
                    raise ValueError(f"The selected {root['name']} footprint contains no partitions")
            kwargs: dict[str, Any] = {'client': client, 'partitions_per_chunk': self.spec['partitions_per_chunk'],
                                      'seed': (seed + i) % 2**32}
            if joins:
                if footprint is not None:
                    kwargs['footprint'] = footprint
                if CrossMatchStream is _LSDB_CROSSMATCH_STREAM:
                    stream = _FilteredCrossMatchStream(
                        anchor, *joins, min_catalogs=minimum, radius_arcsec=self.spec['radius_arcsec'], **kwargs
                    )
                else:
                    stream = CrossMatchStream(anchor, *joins, min_catalogs=minimum, **kwargs)
            else:
                stream = InfiniteStream(anchor if footprint is None else anchor.moc_search(footprint), **kwargs)
            if not stream.npartitions:
                raise ValueError("The selected catalog footprint contains no partitions")
            streams.append(stream)
        return streams

    def draws(self, *, seed, rank, worker_id, shard=(0, 1)):
        """Yield (epoch, finite records) with bounded transport retries, no cursor.

        ``shard=(index, count)`` keeps only every count-th anchor partition, from index.
        """
        seed_hf_listing(self.hf_listing)
        enable_hf_concurrent_ranges()
        # TODO(upstream): Remove once nested-pandas#586 & #589 are merged and released.
        enable_hf_coalesced_parquet_reads()
        _unregister_hf_array2d_extension()
        cached_hats = load_hats_metadata(self.hats_metadata)
        telemetry._REPO_TO_SOURCE.update({cat["repo"].split("@")[0]: cat["name"] for cat in self.catalogs})
        failures: dict[int | None, int] = {}
        generation = draw = 0
        while True:
            star_index = None  # Startup has its own retry budget.
            try:
                with LocalCluster(
                    processes=False, n_workers=1,
                    threads_per_worker=self.spec["partitions_per_chunk"],
                    dashboard_address=None,  # pyright: ignore[reportArgumentType] -- documented disable value
                    protocol="inproc", memory_limit=self.spec["worker_memory_limit"],
                ) as cluster, Client(cluster) as client:
                    opened = {}
                    anchor_names = {star.catalogs[0]["name"] for star in self.stars}
                    for catalog in self.catalogs:
                        decoder = catalog["decoder"]
                        payload, fields, family = _PROFILES[decoder]
                        columns = [_ID_COLUMNS.get(decoder, "object_id"), "ra", "dec"]
                        if payload and fields:
                            columns.extend(f"{payload}.{field}" for field in fields)
                        elif payload:
                            columns.append(payload)
                        columns.extend(catalog["scalar_columns"].values())
                        dedup_cols = list(dict.fromkeys(columns))
                        if catalog["repo"] in cached_hats:
                            hc_cat, margin_hc = cached_hats[catalog["repo"]]
                            margin_cols = None
                            if catalog["name"] in anchor_names and margin_hc is not None:
                                info = margin_hc.catalog_info
                                margin_cols = [c for c in (info.ra_column, info.dec_column) if c] or None
                            opened_cat = _open_from_hats_metadata(
                                hc_cat, margin_hc, columns=dedup_cols, margin_columns=margin_cols
                            )
                        else:
                            opened_cat = open_catalog(f"hf://datasets/{catalog['repo']}", columns=dedup_cols)
                            if catalog["name"] in anchor_names:
                                opened_cat = _project_anchor_margin(opened_cat)
                        opened[catalog['name']] = opened_cat
                    reader_seed = loader.consumer_seed(seed, rank, worker_id, generation)
                    streams = self._streams(opened, client, reader_seed, shard)
                    failures.pop(None, None)
                    print(f"[catalog-stream] worker={worker_id} shard={shard} pixels={[s.npartitions for s in streams]} "
                          f"generation={generation} catalogs={len(opened)} "
                          f"min_catalogs={self.spec['min_catalogs']}", flush=True)
                    readers = {}
                    for star_index in _draw_paths(self.weights, reader_seed % len(streams)):
                        if star_index not in readers:
                            readers[star_index] = iter(streams[star_index])
                        frame = next(readers[star_index], None)
                        if frame is None:
                            raise RuntimeError("Infinite catalog stream ended unexpectedly")
                        failures.pop(star_index, None)
                        epoch = draw
                        draw += 1
                        if frame.empty:
                            continue
                        star = self.stars[star_index]
                        decoded = frame.map_rows(star.decode_row, columns=star.columns, infer_nesting=False)
                        records = filter(None, decoded["record"].tolist())
                        del decoded, frame
                        yield epoch, records
            except Exception as error:
                if not loader._retryable(error):
                    raise
                attempts = failures[star_index] = failures.get(star_index, 0) + 1
                if attempts > loader._MAX_NET_RETRIES:
                    raise
                wait = min(5 * 2 ** (attempts - 1), loader._MAX_NET_RETRY_WAIT)
                anchor = 'startup' if star_index is None else self.stars[star_index].catalogs[0]['name']
                print(f"[catalog-stream] worker={worker_id} anchor={anchor} retry={attempts} "
                      f"error={type(error).__name__} wait_s={wait}", flush=True)
                time.sleep(wait)
                generation += 1
