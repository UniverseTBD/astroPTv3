"""Configured MMU catalogs -> finite decoded draws, independent of modality count."""

from __future__ import annotations

import copy
import inspect
import math
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
from lsdb.loaders.hats.read_hats import open_catalog
from lsdb.streams.catalog_streams import CrossMatchStream, InfiniteStream

from . import nanotron_loader as loader
from . import telemetry
from .scalar_registry import _transforms

# Decoder names describe verified payload contracts, not arbitrary calibrations.
_PROFILES = {
    "legacy_image": ("image", ("band", "flux", "psf_fwhm"), "image"),
    "hsc_image": ("image", ("band", "flux", "scale"), "image"),
    "desi_spectrum": ("spectrum", ("flux", "lambda", "mask"), "spectrum"),
    "sdss_spectrum": ("spectrum", ("flux", "lambda", "mask"), "spectrum"),
    "scalars": (None, (), "scalar"),
}
_DEFAULTS = {
    "how": "outer", "radius_arcsec": 1.0, "count_fraction_threshold": 0.0,
    "partitions_per_chunk": 4, "worker_memory_limit": "auto",
}


class CatalogStream:
    """Validate a saved stream specification, then decode its optional modalities.

    The first catalog is the anchor. Right-only rows remain separate objects;
    right catalogs are not crossmatched to each other without an anchor match.
    """

    def __init__(self, spec: dict, config):
        if not isinstance(spec, dict) or set(spec) - {*_DEFAULTS, "catalogs", "min_surveys"}:
            raise ValueError("catalog_stream has unknown settings or is not a mapping")
        catalogs = spec.get("catalogs")
        if not isinstance(catalogs, list) or not catalogs:
            raise ValueError("catalog_stream.catalogs must be a nonempty list")
        self.spec = {**_DEFAULTS, "min_surveys": 1, **copy.deepcopy(spec)}
        self.catalogs = self.spec["catalogs"]
        self.registry = config.modality_registry()
        self.nested: dict[str, tuple[str, ...]] = {}
        self.columns = []
        outputs = {}
        names = set()
        for index, catalog in enumerate(self.catalogs):
            if not isinstance(catalog, dict) or set(catalog) - {
                "name", "repo", "decoder", "record_key", "scalar_columns",
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
            self.columns.append(f"object_id{suffix}")
            if payload:
                key = catalog.get("record_key")
                if not isinstance(key, str) or not key or key == "object_id" or key in outputs:
                    raise ValueError("catalog record keys must be nonempty and unique")
                outputs[key] = (family, decoder, name)
                column = f"{payload}{suffix}"
                self.nested[column] = fields
                self.columns.extend(f"{column}.{field}" for field in fields)
            elif "record_key" in catalog:
                raise ValueError("scalar-only catalogs use scalar_columns, not record_key")
            scalars = catalog.setdefault("scalar_columns", {})
            if not isinstance(scalars, dict) or (not payload and not scalars):
                raise ValueError("scalar_columns must map record keys to flat catalog columns")
            for key, column in scalars.items():
                if (not isinstance(key, str) or not key or key == "object_id" or key in outputs
                        or not isinstance(column, str) or not column or "." in column
                        or column in {"image", "spectrum"}):
                    raise ValueError("scalar_columns needs unique record keys and flat column names")
                outputs[key] = ("scalar", decoder, name)
                self.columns.append(f"{column}{suffix}")
        if not self.registry.names():
            raise ValueError("at least one model modality is required")
        for name in self.registry.names():
            modality = self.registry.get_config(name)
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
        minimum = self.spec["min_surveys"]
        if type(minimum) is not int or not 1 <= minimum <= len(self.catalogs):
            raise ValueError("min_surveys must be an integer in 1..number of catalogs")
        if self.spec["how"] not in ("left", "outer"):
            raise ValueError("catalog_stream.how must be left or outer")
        for key in ("radius_arcsec", "count_fraction_threshold"):
            value = self.spec[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
        if self.spec["radius_arcsec"] <= 0 or not 0 <= self.spec["count_fraction_threshold"] <= 1:
            raise ValueError("radius must be positive and count_fraction_threshold in [0, 1]")
        if type(self.spec["partitions_per_chunk"]) is not int or self.spec["partitions_per_chunk"] < 1:
            raise ValueError("partitions_per_chunk must be a positive integer")
        if not isinstance(self.spec["worker_memory_limit"], str) or not self.spec["worker_memory_limit"]:
            raise ValueError("worker_memory_limit must be a Dask memory-limit string")
        self.columns = list(dict.fromkeys(self.columns))

    def decode_row(self, mapped):
        row = loader._row_from_map_rows(mapped, self.nested)
        record = {}
        for index, catalog in enumerate(self.catalogs):
            name, decoder = catalog["name"], catalog["decoder"]
            payload, _, _ = _PROFILES[decoder]
            suffix = "" if index == 0 else f"_{name}"
            value = row.get(f"{payload}{suffix}") if payload else None
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
            if "object_id" not in record:
                object_id = row.get(f"object_id{suffix}")
                if bool(pd.isna(object_id)):
                    raise ValueError(f"{name} row has no object_id")
                record["object_id"] = str(object_id) if index == 0 else f"{name}:{object_id}"
            record.update(scalars)
            if value is None:
                continue
            if decoder == "legacy_image":
                decoded = loader.decode_legacy_row({"object_id": record["object_id"], "image": value})
                result = decoded["image"]
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
            else:
                result = loader._decode_spectrum(value)
                if decoder == "sdss_spectrum":
                    # Nonpositive SDSS wavelengths are padding; NaNs remain validation errors.
                    keep = ~(result["lambda"] <= 0)
                    result = {key: values[keep] for key, values in result.items()}
            record[catalog["record_key"]] = result
        if not record:
            if any(not bool(pd.isna(row.get(f"object_id{'' if i == 0 else '_' + cat['name']}")))
                   for i, cat in enumerate(self.catalogs)):
                return {"record": None}
            raise ValueError("crossmatched row has no supported data or object_id")
        for name in self.registry.names():
            modality = self.registry.get_config(name)
            if (all(key in record for key in modality.record_keys)
                    and (name != "Z" or record.get("ZWARN") == 0)):
                return {"record": record}
        return {"record": None}

    def draws(self, *, seed, rank, worker_id):
        """Yield (epoch, finite records) with bounded transport retries, no cursor."""
        if len(self.catalogs) > 1 and "min_surveys" not in inspect.signature(CrossMatchStream).parameters:
            raise RuntimeError("configured catalog streams require LSDB's multi-catalog/min_surveys API")
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
                    catalogs = []
                    for catalog in self.catalogs:
                        payload = _PROFILES[catalog["decoder"]][0]
                        columns = ["object_id", "ra", "dec", *([payload] if payload else []),
                                   *catalog["scalar_columns"].values()]
                        catalogs.append(open_catalog(
                            f"hf://datasets/{catalog['repo']}", columns=list(dict.fromkeys(columns)),
                        ))
                    kwargs = {
                        "client": client, "partitions_per_chunk": self.spec["partitions_per_chunk"],
                        "seed": loader.consumer_seed(seed, rank, worker_id, generation),
                    }
                    stream: Any
                    if len(catalogs) == 1:
                        stream = InfiniteStream(catalogs[0], **kwargs)
                    else:
                        stream = CrossMatchStream(
                            catalogs[0], *[
                                {"other": catalog, "how": self.spec["how"],
                                 "suffixes": ("", f"_{spec['name']}"),
                                 "radius_arcsec": self.spec["radius_arcsec"]}
                                for catalog, spec in zip(catalogs[1:], self.catalogs[1:], strict=True)
                            ],
                            min_surveys=self.spec["min_surveys"],
                            count_fraction_threshold=self.spec["count_fraction_threshold"], **kwargs,
                        )
                    if not stream.npartitions:
                        raise ValueError("The selected catalog footprint contains no partitions")
                    print(f"[catalog-stream] worker={worker_id} pixels={stream.npartitions} "
                          f"generation={generation} catalogs={len(catalogs)} "
                          f"min_surveys={self.spec['min_surveys']}", flush=True)
                    for frame in stream:
                        failures = 0
                        epoch = draw
                        draw += 1
                        if frame.empty:
                            continue
                        decoded = frame.map_rows(self.decode_row, columns=self.columns, infer_nesting=False)
                        records = filter(None, decoded["record"].tolist())
                        del decoded, frame
                        yield epoch, records
                raise RuntimeError("Infinite catalog stream ended unexpectedly")
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
