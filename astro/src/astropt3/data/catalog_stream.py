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
from dask.distributed import Client, LocalCluster
from lsdb.loaders.hats.read_hats import open_catalog
from lsdb.streams.catalog_streams import (
    CrossMatchStream,
    InfiniteStream,
    _data_coverage,
    _restrict_to_footprint,
)
from mocpy import MOC
from PIL import Image

from . import nanotron_loader as loader
from . import telemetry
from .packing import has_modality
from .scalar_registry import _transforms

# Decoder names describe verified payload contracts, not arbitrary calibrations.
_PROFILES = {
    "legacy_image": ("image", ("band", "flux"), "image"),
    "legacy_south_image": ("image", ("band", "flux", "ivar", "mask", "scale"), "image"),
    "hsc_image": ("image", ("band", "flux", "scale"), "image"),
    "desi_spectrum": ("spectrum", ("flux", "lambda", "mask"), "spectrum"),
    "sdss_spectrum": ("spectrum", ("flux", "lambda", "mask"), "spectrum"),
    "scalars": (None, (), "scalar"),
    # 8-bit RGB PNG/JPEG struct {bytes, path} (e.g. galaxies-with-hats); dequantized to (k + U(0,1)) / 255
    "rgb_png": ("image", (), "image"),
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
            for key in modality.record_keys:
                if key not in outputs or outputs[key][0] != modality.family:
                    raise ValueError(f"modality {name!r} has no compatible catalog output for {key!r}")
                if modality.family == "spectrum" and outputs[key][1] != f"{modality.source}_spectrum":
                    raise ValueError(f"modality {name!r} source disagrees with its spectrum decoder")
            if name == "Z" and ("ZWARN" not in outputs or any(
                outputs[key][2] != outputs["ZWARN"][2] for key in modality.record_keys
            )):
                raise ValueError("Z requires an explicit ZWARN scalar column from the same catalog")

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
            # A missing/invalid quality flag must not make a redshift look reliable.
            if ("ZWARN" in scalars and scalars["ZWARN"] is None
                    and (value is not None or any(number is not None for number in scalars.values()))):
                scalars["ZWARN"] = 1.0
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
        self.spec = spec = {**_DEFAULTS, "min_catalogs": 1, **copy.deepcopy(spec)}
        entries = spec.get("catalogs")
        if not isinstance(entries, list) or (not entries and "anchors" not in spec):
            raise ValueError("catalog_stream.catalogs must be a list with at least one catalog in the star")
        self.weights = [1]
        if "anchors" in spec:
            anchors = spec["anchors"]
            if not isinstance(anchors, list) or not anchors or not all(isinstance(c, dict) for c in anchors):
                raise ValueError("anchors must be a nonempty list of catalogs with draw_weight")
            self.weights = [c.get("draw_weight") for c in anchors]
            if any(type(w) is not int or w < 1 for w in self.weights):
                raise ValueError("draw_weight must be a positive integer for every anchor")
            partners = entries
            anchors = [{k: v for k, v in c.items() if k != "draw_weight"} for c in anchors]
        else:
            anchors, partners = entries[:1], entries[1:]
        self.catalogs = [*anchors, *partners]
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

    def _streams(self, opened, client, seed):
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
            if len(self.stars) > 1 or 'split' in self.spec:
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
                if footprint.empty():
                    raise ValueError(f"The selected {root['name']} footprint contains no partitions")
            kwargs: dict[str, Any] = {'client': client, 'partitions_per_chunk': self.spec['partitions_per_chunk'],
                                      'seed': (seed + i) % 2**32}
            if joins:
                if footprint is not None:
                    kwargs['footprint'] = footprint
                stream = CrossMatchStream(anchor, *joins, min_catalogs=minimum, **kwargs)
            else:
                stream = InfiniteStream(anchor if footprint is None else anchor.moc_search(footprint), **kwargs)
            if not stream.npartitions:
                raise ValueError("The selected catalog footprint contains no partitions")
            streams.append(stream)
        return streams

    def draws(self, *, seed, rank, worker_id):
        """Yield (epoch, finite records) with bounded transport retries, no cursor."""
        _unregister_hf_array2d_extension()
        telemetry._REPO_TO_SOURCE.update({cat["repo"].split("@")[0]: cat["name"] for cat in self.catalogs})
        failures = generation = draw = 0
        while True:
            try:
                with LocalCluster(
                    processes=False, n_workers=1,
                    threads_per_worker=self.spec["partitions_per_chunk"],
                    dashboard_address=None,  # pyright: ignore[reportArgumentType] -- documented disable value
                    protocol="inproc", memory_limit=self.spec["worker_memory_limit"],
                ) as cluster, Client(cluster) as client:
                    opened = {}
                    for catalog in self.catalogs:
                        payload = _PROFILES[catalog["decoder"]][0]
                        columns = [_ID_COLUMNS.get(catalog["decoder"], "object_id"), "ra", "dec",
                                   *([payload] if payload else []),
                                   *catalog["scalar_columns"].values()]
                        opened[catalog['name']] = open_catalog(
                            f"hf://datasets/{catalog['repo']}", columns=list(dict.fromkeys(columns)),
                        )
                    reader_seed = loader.consumer_seed(seed, rank, worker_id, generation)
                    streams = self._streams(opened, client, reader_seed)
                    print(f"[catalog-stream] worker={worker_id} pixels={[s.npartitions for s in streams]} "
                          f"generation={generation} catalogs={len(opened)} "
                          f"min_catalogs={self.spec['min_catalogs']}", flush=True)
                    readers = {}
                    for path in _draw_paths(self.weights, reader_seed % len(streams)):
                        if path not in readers:
                            readers[path] = iter(streams[path])
                        frame = next(readers[path], None)
                        if frame is None:
                            raise RuntimeError("Infinite catalog stream ended unexpectedly")
                        failures = 0
                        epoch = draw
                        draw += 1
                        if frame.empty:
                            continue
                        star = self.stars[path]
                        decoded = frame.map_rows(star.decode_row, columns=star.columns, infer_nesting=False)
                        records = filter(None, decoded["record"].tolist())
                        del decoded, frame
                        yield epoch, records
            except Exception as error:
                if not loader._retryable(error):
                    raise
                failures += 1
                if failures > loader._MAX_NET_RETRIES:
                    raise
                wait = min(5 * 2 ** (failures - 1), loader._MAX_NET_RETRY_WAIT)
                print(f"[catalog-stream] worker={worker_id} retry={failures} "
                      f"error={type(error).__name__} wait_s={wait}", flush=True)
                time.sleep(wait)
                generation += 1
