#!/usr/bin/env python
"""Bounded, finite sky audit using the recipe's actual LSDB star joins.

Run from astro/: uv run python scripts/audit_modalities.py --help
No model is loaded. Output is separate from the training configuration and caches.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import astropy.units as u
from astropy.coordinates import angular_separation
import cdshealpix
import dask
from huggingface_hub import HfFileSystem, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
import lsdb
from lsdb.streams.catalog_streams import _data_coverage
from mocpy import MOC
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from upath import UPath

from astropt3.configuration_astropt3 import AstroPT3Config
from astropt3.data import catalog_stream as streaming
from astropt3.data import telemetry
from astropt3.data.nanotron_loader import _finite_scalar
from astropt3.data.packing import ObjectSequencer
from astropt3.data.scalar_registry import CATEGORICAL, QUALITY_GATES, VALID_RANGES


DIAGNOSTICS = {
    "legacy_image": ["flux_g", "flux_r", "flux_z", "ebv"],
    "legacy_south_image": ["FLUX_G", "FLUX_R", "FLUX_I", "FLUX_Z", "SHAPE_R", "EBV"],
    "desi_spectrum": ["Z", "ZERR", "ZWARN"],
    "sdss_spectrum": ["Z", "Z_ERR", "ZWARNING"],
    "hsc_image": ["g_cmodel_mag", "r_cmodel_mag", "i_cmodel_mag"],
    "rgb_png": [],
    "scalars": [],
}


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def open_projected(spec, cache):
    """Cache immutable HATS metadata, then leave all primary/margin data remote."""
    repo, ref = spec['repo'].split('@')
    revision, _, prefix = ref.partition('/')
    local = cache / spec['name'] / revision
    metadata_files = []

    def download(relative, optional=False):
        filename = '/'.join(p for p in (prefix, relative) if p)
        try:
            path = Path(hf_hub_download(repo, filename, repo_type='dataset', revision=revision,
                                       token=False, local_dir=local))
        except EntryNotFoundError:
            if optional:
                return None
            raise
        metadata_files.append({'file': filename, 'bytes': path.stat().st_size, 'sha256': fingerprint(path)})
        return path

    collection = download('collection.properties')
    props = dict(line.split('=', 1) for line in collection.read_text(encoding='utf-8').splitlines()
                 if line and not line.startswith('#') and '=' in line)
    primary = props['hats_primary_table_url']
    margin = props['default_margin']
    if any('://' in name or name.startswith('/') or '..' in name.split('/') for name in (primary, margin)):
        raise ValueError('Metadata cache expects collection-relative primary and margin paths')
    for table in (primary, margin):
        download(f'{table}/hats.properties')
        download(f'{table}/partition_info.csv')
        if download(f'{table}/dataset/_common_metadata', optional=True) is None:
            download(f'{table}/dataset/_metadata')
        download(f'{table}/point_map.fits', optional=True)
    primary_uri = f"hf://datasets/{spec['repo']}/{primary}"
    schema_file = local / prefix / primary / 'dataset/_common_metadata'
    if not schema_file.exists():
        schema_file = schema_file.with_name('_metadata')
    schema = pq.read_schema(schema_file)
    diagnostics = [c for c in DIAGNOSTICS[spec['decoder']] if c in schema.names]
    columns = list(dict.fromkeys(['ra', 'dec', '_healpix_29', streaming._ID_COLUMNS.get(spec['decoder'], 'object_id'),
                                 *spec['scalar_columns'].values(), *diagnostics]))
    projection_key = hashlib.sha256(json.dumps(columns).encode()).hexdigest()[:16]

    def remote_partition(base, pixel, query, suffix):
        relative = Path(str(base)).relative_to(local).as_posix()
        relative += (f'/dataset/Norder={pixel.order}/Dir={pixel.pixel // 10000 * 10000}/'
                     f'Npix={pixel.pixel}{suffix}')
        projected = local / 'projections' / projection_key / relative
        if not projected.exists():
            uri = f'hf://datasets/{repo}@{revision}/{relative}'
            # Arrow coalesces column ranges; a second readahead cache would fetch
            # gaps repeatedly and make sparse scalar projections latency-bound.
            with HfFileSystem(token=False).open(uri, block_size=131072, cache_type='none') as remote:
                table = pq.ParquetFile(remote, pre_buffer=True).read(columns=columns)
            projected.parent.mkdir(parents=True, exist_ok=True)
            temporary = projected.with_suffix('.tmp')
            pq.write_table(table, temporary)
            temporary.replace(projected)
        return UPath(projected)

    catalog = lsdb.open_catalog(local / prefix, columns=columns, path_generator=remote_partition)
    projection_base = local / 'projections' / projection_key / prefix / primary
    return catalog, primary_uri, metadata_files, projection_base


def present(value):
    return value is not None and not bool(pd.isna(value))


def describe(values):
    """Finite raw-value distribution; no fitted normalization or science cuts."""
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not len(values):
        return {"n": 0}
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    np.quantile(values, [0, .05, .5, .95, 1]).tolist()), n=len(values))


def scalar_status(modality, record, sequencer):
    raw = [record.get(key) for key in modality.record_keys]
    if any(v is None for v in raw):
        return "missing"
    if any(not math.isfinite(float(v)) for v in raw):
        return "nonfinite"
    low, high = VALID_RANGES.get(modality.name, (-math.inf, math.inf))
    if any(not low < float(v) < high for v in raw):
        return "range_rejected"
    classes = CATEGORICAL.get(modality.name)
    if classes is not None and any(not float(v).is_integer() or not 0 <= float(v) < classes for v in raw):
        return "range_rejected"
    if sequencer._scalar_value(modality.name, record) is None:
        return "quality_rejected"
    if not torch.isfinite(sequencer._scalar_tokens(modality.name, record)[0]).all():
        return "nonfinite_transform"
    return "usable"


def scalar_record(row, star):
    """Mirror the stream's scalar mapping, preserving nonfinite values for attribution."""
    record = {}
    for index, cat in enumerate(star.catalogs):
        suffix = "" if index == 0 else f"_{cat['name']}"
        for key, column in cat["scalar_columns"].items():
            value = row.get(column + suffix)
            record[key] = None if value is None or value is pd.NA else float(value)
    # The decoder drops nonfinite quality flags before the sequencer sees them.
    # Preserve nonfinite target values above so their rejection stays visible.
    for gates in QUALITY_GATES.values():
        for key in gates:
            if key in record:
                record[key] = _finite_scalar(record[key])
    return record


def source_ids(row, star):
    result = {}
    for index, cat in enumerate(star.catalogs):
        suffix = "" if index == 0 else f"_{cat['name']}"
        value = row.get(streaming._ID_COLUMNS.get(cat["decoder"], "object_id") + suffix)
        if present(value):
            result[cat["name"]] = str(value)
    return result


def summarize(frame, star, sequencer):
    names = [c["name"] for c in star.catalogs]
    scalar_mods = [sequencer.registry.get_config(n) for n in sequencer.registry.names()
                   if sequencer.registry.get_config(n).family == "scalar"]
    combinations, sources, statuses = Counter(), Counter(), {m.name: Counter() for m in scalar_mods}
    values = {m.name: [] for m in scalar_mods}
    diagnostics = {}
    seen = set()
    duplicates = 0
    for row in frame.to_dict("records"):
        ids = source_ids(row, star)
        identity = tuple(ids.items())
        duplicates += identity in seen
        seen.add(identity)
        sources.update(ids.keys())
        combinations["+".join(ids)] += 1
        record = scalar_record(row, star)
        for mod in scalar_mods:
            state = scalar_status(mod, record, sequencer)
            statuses[mod.name][state] += 1
            if state == "usable":
                values[mod.name].extend(sequencer._scalar_value(mod.name, record))
        for index, cat in enumerate(star.catalogs):
            suffix = "" if index == 0 else f"_{cat['name']}"
            for column in DIAGNOSTICS[cat["decoder"]]:
                if column + suffix in row:
                    key = f"{cat['name']}.{column}"
                    diagnostics.setdefault(key, []).append(_finite_scalar(row[column + suffix]))
    return {
        "rows": len(frame), "anchor_rows": sources[names[0]],
        "outer_only_rows": len(frame) - sources[names[0]],
        "repeated_source_tuples": duplicates,
        "source_presence": {n: sources[n] for n in names},
        "source_combinations": dict(sorted(combinations.items())),
        "scalars": {m.name: {"counts": dict(statuses[m.name]), "raw_usable": describe(values[m.name])}
                    for m in scalar_mods},
        "diagnostics_raw_finite": {k: describe(v) for k, v in diagnostics.items()},
    }


def cone_pixels(pixels, ra, dec, radius):
    """Find all output partitions intersecting a cone, including edge cells."""
    by_order = {}
    for order in {p.order for p in pixels}:
        selected, _, _ = cdshealpix.cone_search(ra * u.deg, dec * u.deg, radius * u.arcsec,
                                               depth=order, flat=True)
        by_order[order] = set(selected.tolist())
    return [p for p in pixels if p.pixel in by_order[p.order]]


def in_cone(frame, field):
    ra = frame["_ra"].to_numpy(dtype=float)
    dec = frame["_dec"].to_numpy(dtype=float)
    sep = angular_separation(np.radians(field["ra"]), np.radians(field["dec"]),
                             np.radians(ra), np.radians(dec))
    return frame[np.degrees(sep) * 3600 <= field["radius_arcsec"]].copy()


def verify_star(frame, star, opened, footprint, field, radius):
    """Verify core anchors, their matches, and outer singletons from primary rows.

    Anchors out to R+r can consume a core right row. Their nearest neighbour can
    lie out to R+2r, so both halos are needed before classifying an unmatched row.
    The reference uses Astropy distances, no LSDB crossmatch or margin rows.
    Exact-distance ties accept any nearest ID; ambiguous outer IDs are withheld
    and counted because a halo anchor's tie-breaking choice is unobserved.
    """
    def primary(name, halo):
        return opened[name].cone_search(field['ra'], field['dec'],
                                        field['radius_arcsec'] + halo + .01).compute(progress_bar=False)

    def inside_footprint(rows):
        mask = footprint.contains_lonlat(rows['ra'].to_numpy(dtype=float) * u.deg,
                                         rows['dec'].to_numpy(dtype=float) * u.deg)
        return rows[mask]

    def core(rows):
        return in_cone(rows.assign(_ra=rows['ra'], _dec=rows['dec']), field)

    anchor_name = star.catalogs[0]['name']
    halo_anchors = inside_footprint(primary(anchor_name, radius))
    expected_anchors = core(halo_anchors)
    anchors = frame[frame['object_id'].notna()]
    expected_ids = Counter(expected_anchors['object_id'].astype(str))
    if Counter(anchors['object_id'].astype(str)) != expected_ids:
        raise AssertionError(f"{field['name']}/{anchor_name}: core anchor rows differ from primary reference")
    if any(count != 1 for count in expected_ids.values()):
        raise ValueError('Independent match validation requires unique anchor IDs within the aperture')
    actual_anchors = {str(row['object_id']): row for row in anchors.to_dict('records')}
    outer = frame[frame['object_id'].isna()]
    if any(len(source_ids(row, star)) != 1 for row in outer.to_dict('records')):
        raise AssertionError('Outer records must contain exactly one source')
    checked, ambiguous, checked_outer, withheld_outer = 0, 0, 0, 0
    for cat in star.catalogs[1:]:
        candidates = primary(cat['name'], 2 * radius)
        id_col = streaming._ID_COLUMNS.get(cat['decoder'], 'object_id')
        lon = np.radians(candidates['ra'].to_numpy(dtype=float))
        lat = np.radians(candidates['dec'].to_numpy(dtype=float))
        candidate_ids = candidates[id_col].astype(str).to_numpy()
        consumed, uncertain = set(), set()
        for row in halo_anchors.to_dict('records'):
            choices = set()
            if len(candidates):
                distance = np.degrees(angular_separation(np.radians(row['ra']), np.radians(row['dec']), lon, lat)) * 3600
                nearest = int(np.argmin(distance))
                if distance[nearest] <= radius:
                    choices = set(candidate_ids[np.abs(distance - distance[nearest]) < 1e-8])
            identifier = str(row['object_id'])
            if len(choices) > 1:
                ambiguous += 1
                if identifier not in actual_anchors:
                    uncertain.update(choices)
            elif choices:
                consumed.update(choices)
            if identifier not in actual_anchors:
                continue
            actual = actual_anchors[identifier].get(id_col + '_' + cat['name'])
            actual = str(actual) if present(actual) else None
            if actual not in (choices or {None}):
                raise AssertionError(f"{field['name']}/{anchor_name}/{cat['name']}: "
                                     f"anchor {identifier} selects {actual}, reference {sorted(choices)}")
            if actual is not None:
                consumed.add(actual)
            checked += 1
        uncertain -= consumed
        if cat.get('how', 'outer') == 'left':
            if any(present(value) for value in outer[id_col + '_' + cat['name']]):
                raise AssertionError(f"{cat['name']}: left-joined labels must not appear in outer records")
            continue
        right_core = core(inside_footprint(candidates))
        expected_outer = Counter(identifier for identifier in right_core[id_col].astype(str)
                                 if identifier not in consumed | uncertain)
        actual_outer = Counter(str(value) for value in outer[id_col + '_' + cat['name']]
                               if present(value) and str(value) not in uncertain)
        if actual_outer != expected_outer:
            missing = list((expected_outer - actual_outer).items())[:5]
            extra = list((actual_outer - expected_outer).items())[:5]
            raise AssertionError(f"{field['name']}/{anchor_name}/{cat['name']}: outer rows differ; "
                                 f"missing={missing}, extra={extra}")
        checked_outer += sum(expected_outer.values())
        withheld_outer += sum(identifier in uncertain for identifier in right_core[id_col].astype(str))
    return {'checked_anchor_rows': len(expected_anchors), 'checked_anchor_partner_pairs': checked,
            'checked_outer_rows': checked_outer, 'distance_ties': ambiguous,
            'outer_ids_withheld_for_halo_ties': withheld_outer,
            'reference': 'Astropy primary-row reference; anchors R+r, partners R+2r; recipe footprint'}


def choose_fields(opened, streams, per_region, radius, seed, separation_deg=5):
    """Select reproducible eligible order-10 cell centres, with angular separation.

    The region label describes the centre, not every point in the aperture.
    These are stratified pilot fields, never a catalogue-wide probability sample.
    """
    north, south = [_data_coverage(opened[n]) for n in ("legacy", "legacy_south")]
    regions = {
        "north_only": streams[0].footprint.difference(south),
        "south_only": streams[1].footprint.difference(north),
        "overlap": streams[0].footprint.intersection(south),
    }
    rng = np.random.default_rng(seed)
    result = []
    for label, moc in regions.items():
        cells = rng.permutation(moc.to_order(10).flatten())
        if not len(cells):
            raise ValueError(f"No eligible {label} cells")
        longitudes, latitudes = cdshealpix.healpix_to_lonlat(cells.astype(np.uint64), depth=10)
        ras, decs = np.asarray(longitudes.deg), np.asarray(latitudes.deg)
        eligible = moc.contains_lonlat(longitudes, latitudes)
        chosen = []
        for _ in range(per_region):
            candidates = np.flatnonzero(eligible)
            if not len(candidates):
                break
            index = candidates[0]
            ra, dec, cell = float(ras[index]), float(decs[index]), cells[index]
            chosen.append({"name": f"{label}_{len(chosen) + 1}", "region": label,
                           "ra": ra, "dec": dec, "radius_arcsec": radius,
                           "selection_seed": seed, "minimum_separation_deg": separation_deg,
                           "centre_order10": int(cell)})
            distances = np.degrees(angular_separation(np.radians(ra), np.radians(dec),
                                                       np.radians(ras), np.radians(decs)))
            eligible &= distances >= separation_deg
        if len(chosen) != per_region:
            raise ValueError(f"Only {len(chosen)} {label} fields fit separation={separation_deg} deg; requested {per_region}")
        result.extend(chosen)
    return result


def validate_fields(fields):
    if not isinstance(fields, list) or not fields:
        raise ValueError('Fields must be a nonempty list')
    names = set()
    for field in fields:
        name = field.get('name', '')
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', name) or name in names:
            raise ValueError('Field names must be unique safe identifiers')
        names.add(name)
        ra, dec, radius = (float(field[k]) for k in ('ra', 'dec', 'radius_arcsec'))
        if not (math.isfinite(ra) and 0 <= ra < 360 and math.isfinite(dec) and -90 <= dec <= 90
                and math.isfinite(radius) and 0 < radius < 180 * 3600):
            raise ValueError(f'Invalid spherical aperture for {name}')


def select_rows(frame, star, per_combination):
    """Hash selection within source combinations, independent of catalogue sort order."""
    groups = {}
    for row in frame.to_dict("records"):
        ids = source_ids(row, star)
        key = "+".join(ids)
        digest = hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest()
        groups.setdefault(key, []).append((digest, row))
    return [row for group in groups.values() for _, row in sorted(group, key=lambda x: x[0])[:per_combination]]


def bound_payload_selection(selected, limit):
    """Cap the purposive sample, covering sources first, then field/path pairs.

    Hash ordering makes selection independent of frame and field iteration order.
    Coverage is best effort when the cap cannot accommodate every group.
    """
    def key(item):
        field, star, row = item
        return hashlib.sha256(json.dumps([field, star.catalogs[0]['name'], source_ids(row, star)],
                                          sort_keys=True).encode()).hexdigest()

    remaining = sorted(selected, key=key)
    chosen, sources, fields = [], set(), set()
    while remaining and len(chosen) < limit:
        index = max(range(len(remaining)), key=lambda index: (
            len(set(source_ids(remaining[index][2], remaining[index][1])) - sources),
            (remaining[index][0], remaining[index][1].catalogs[0]['name']) not in fields))
        item = remaining.pop(index)
        chosen.append(item)
        sources.update(source_ids(item[2], item[1]))
        fields.add((item[0], item[1].catalogs[0]['name']))
    return chosen


def payload_quality(record, sequencer):
    """Audit the resized/padded input and production sequence, without a model."""
    obj = sequencer.build(record)
    modalities = {}
    for name, tokens in obj.values.items():
        mod = sequencer.registry.get_config(name)
        details = {"tokens": len(tokens), "values": tokens.numel(),
                   "finite_tokens": bool(torch.isfinite(tokens).all())}
        if mod.family in ("image", "spectrum"):
            value = record[mod.record_keys[0]]
            flux = np.asarray(value["flux"])
            if mod.family == "image":
                channel = slice(None) if mod.image_format == "rgb" else [list(value["band"]).index(mod.band)]
                flux = flux[channel]
                fill = np.asarray(value["fill_mask"], dtype=bool)[channel] if "fill_mask" in value else None
                if mod.image_crop:
                    h, w = flux.shape[-2:]
                    y, x = max(0, (h - mod.image_crop) // 2), max(0, (w - mod.image_crop) // 2)
                    region = (..., slice(y, y + mod.image_crop), slice(x, x + mod.image_crop))
                    flux = flux[region]
                    if fill is not None:
                        fill = fill[region]
                if mod.image_resize and flux.shape[-2:] != (mod.image_resize, mod.image_resize):
                    def resize(array):
                        return torch.nn.functional.interpolate(
                            torch.as_tensor(array, dtype=torch.float32).unsqueeze(0),
                            size=(mod.image_resize, mod.image_resize), mode="area").squeeze(0).numpy()
                    flux = resize(flux)
                    if fill is not None:
                        fill = resize(fill) > 0
                if fill is None:
                    fill = flux == 0
                details["validity_rule"] = "published mask/ivar" if "fill_mask" in value else "zero-fill proxy"
            else:
                fill = np.asarray(value["mask"], dtype=bool) | (flux == 0)
                padding = tokens.numel() - flux.size
                flux = np.pad(flux, (0, padding))
                fill = np.pad(fill, (0, padding), constant_values=True)
                details["padding_values"] = padding
                details["validity_rule"] = "published spectrum mask, exact zeros and tail padding"
            details.update(input_values=int(flux.size), filled_values=int(fill.sum()),
                           nonfinite_input_values=int((~np.isfinite(flux)).sum()),
                           all_input_filled=bool(fill.all()))
        modalities[name] = details
    return {"sequence_length": len(obj), "modalities": modalities}


def primary_pixel(catalog, ra, dec):
    pixels = catalog.get_healpix_pixels()
    cells = {order: int(cdshealpix.lonlat_to_healpix(np.array([ra]) * u.deg, np.array([dec]) * u.deg,
                                                   depth=order)[0]) for order in {p.order for p in pixels}}
    for pixel in pixels:
        if cells[pixel.order] == pixel.pixel:
            return pixel
    raise ValueError("Source coordinate is outside its primary partitions")


def fetch_payloads(catalog, primary_uri, spec, requested, output, max_group_mib, max_total_mib,
                   projection_base=None):
    """Read only selected primary row groups; persist the small selected-row tables.

    Parquet cannot fetch an individual row. Reject oversized projected groups
    before reading their payload; never silently count skipped payloads as valid.
    """
    id_col = streaming._ID_COLUMNS.get(spec["decoder"], "object_id")
    payload, fields, _ = streaming._PROFILES[spec["decoder"]]
    columns = [id_col, "ra", "dec", *spec["scalar_columns"].values()]
    if payload:
        columns += [f"{payload}.{f}" for f in fields] if fields else [payload]
    columns = list(dict.fromkeys(columns))
    files = {}
    for identifier, (ra, dec) in requested.items():
        pixel = primary_pixel(catalog, ra, dec)
        relative = f"dataset/Norder={pixel.order}/Dir={pixel.pixel // 10000 * 10000}/Npix={pixel.pixel}.parquet"
        files.setdefault(relative, set()).add(identifier)
    fs = HfFileSystem(token=False)
    found, log = {}, []
    projected_bytes = 0
    for file_index, (relative, ids) in enumerate(sorted(files.items())):
        uri = f"{primary_uri}/{relative}"
        with fs.open(uri, block_size=131072, cache_type="readahead") as remote:
            parquet = pq.ParquetFile(remote, pre_buffer=False)
            cached_ids = None
            projection = projection_base / relative if projection_base is not None else None
            if projection is not None and projection.exists():
                cached_ids = pq.ParquetFile(projection).read(columns=[id_col]).column(id_col).to_pylist()
                if len(cached_ids) != parquet.metadata.num_rows:
                    raise ValueError(f'Projected primary row count changed: {relative}')
            offset = 0
            for group in range(parquet.num_row_groups):
                meta = parquet.metadata.row_group(group)
                identifiers = (cached_ids[offset:offset + meta.num_rows] if cached_ids is not None else
                               parquet.read_row_group(group, columns=[id_col]).column(id_col).to_pylist())
                offset += meta.num_rows
                indices = [i for i, identifier in enumerate(identifiers) if str(identifier) in ids]
                if not indices:
                    continue
                size = sum(meta.column(i).total_uncompressed_size for i in range(meta.num_columns)
                           if any(meta.column(i).path_in_schema == c or
                                  meta.column(i).path_in_schema.startswith(c + ".") for c in columns))
                entry = {"file": relative, "row_group": group, "selected_rows": len(indices),
                         "projected_uncompressed_bytes": size}
                if size > max_group_mib * 1024**2 or projected_bytes + size > max_total_mib * 1024**2:
                    log.append(dict(entry, status="skipped_budget"))
                    continue
                projected_bytes += size
                selected = parquet.read_row_group(group, columns=columns).take(pa.array(indices, type=pa.int64()))
                if selected.column(id_col).to_pylist() != [identifiers[i] for i in indices]:
                    raise ValueError(f'Projected primary row order changed: {relative}')
                path = output / f"{spec['name']}-{file_index}-{group}.parquet"
                pq.write_table(selected, path)
                for row in selected.to_pylist():
                    found[str(row[id_col])] = row
                log.append(dict(entry, status="read", cache=path.name, sha256=fingerprint(path)))
    return found, log


def decode_selected(row, star, payloads):
    mapped = dict(row)
    for index, cat in enumerate(star.catalogs):
        suffix = "" if index == 0 else f"_{cat['name']}"
        id_col = streaming._ID_COLUMNS.get(cat["decoder"], "object_id")
        identifier = row.get(id_col + suffix)
        if not present(identifier):
            continue
        raw = payloads[cat["name"]].get(str(identifier))
        if raw is None:
            return None
        payload, fields, _ = streaming._PROFILES[cat["decoder"]]
        if payload:
            if fields:
                for field in fields:
                    mapped[f"{payload}{suffix}.{field}"] = np.asarray(raw[payload][field])
            else:
                mapped[payload + suffix] = raw[payload]
    return star.decode_row(mapped)["record"]


def check_payloads(selected, payloads, sequencer, seed):
    checks = []
    streaming._DEQUANTIZE = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for field, star, row in selected:
        item = {'field': field, 'path': star.catalogs[0]['name'], 'source_ids': source_ids(row, star)}
        try:
            record = decode_selected(row, star, payloads)
            if record is None:
                item['status'] = 'payload_unavailable'
            else:
                item.update(payload_quality(record, sequencer))
                item['status'] = 'passed' if all(m['finite_tokens'] for m in item['modalities'].values()) else 'nonfinite_tokens'
        except (ValueError, TypeError, KeyError, NotImplementedError, RuntimeError, OSError) as error:
            item.update(status='decode_or_sequence_error', error=f'{type(error).__name__}: {error}')
        checks.append(item)
    return checks


def plot_results(output, summaries, checks):
    """Portable static figures; exact counts remain in the JSON results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nonempty = [s for s in summaries if s['rows']]
    if nonempty:
        sources = sorted({k for s in nonempty for k in s['source_presence']})
        scalars = sorted(nonempty[0]['scalars'])
        columns = sources + scalars
        values = [[100 * s['source_presence'].get(k, 0) / s['rows'] for k in sources] +
                  [100 * s['scalars'][k]['counts'].get('usable', 0) / s['rows'] for k in scalars]
                  for s in nonempty]
        labels = {
            'desi': 'DESI', 'galaxies': 'Galaxies', 'hsc': 'HSC',
            'legacy': 'North', 'legacy_south': 'South', 'sdss': 'SDSS',
            'gwh_disk-edge-on_yes_fraction': 'Edge-on',
            'gwh_has-spiral-arms_yes_fraction': 'Spiral arms',
            'gwh_mass_med_photoz': 'Mass', 'gwh_merging_merger_fraction': 'Merger',
            'gwh_redshift': 'Redshift',
            'gwh_smooth-or-featured_featured-or-disk_fraction': 'Featured/disk',
            'gwh_smooth-or-featured_smooth_fraction': 'Smooth',
            'gwh_total_sfr_median': 'SFR', 'gwh_total_ssfr_median': 'sSFR',
        }
        fig, ax = plt.subplots(figsize=(13, max(4, len(nonempty) * .55 + 2)))
        im = ax.imshow(values, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
        ax.set_xticks(range(len(columns)), [labels.get(c, c) for c in columns], rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(nonempty)),
                      [f"{s['field']} / {labels.get(s['path'], s['path'])} (n={s['rows']})" for s in nonempty],
                      fontsize=9)
        ax.axvline(len(sources) - .5, color='white', lw=3)
        ax.set_title('Catalogue presence and usable scalar targets in finite fields')
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', fontsize=7,
                        color='white' if value > 65 else 'black')
        fig.colorbar(im, ax=ax, label='% of all emitted rows in this field/path', shrink=.85)
        fig.tight_layout()
        fig.savefig(output / 'coverage.png', dpi=160)
        plt.close(fig)
    fills = {}
    for check in checks:
        for name, mod in check.get('modalities', {}).items():
            if 'filled_values' in mod:
                fills.setdefault(name, []).append(100 * mod['filled_values'] / mod['input_values'])
    if fills:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(list(fills.values()), tick_labels=[f'{n} (n={len(v)})' for n, v in fills.items()],
                   showfliers=True)
        ax.tick_params(axis='x', rotation=60)
        ax.set_ylabel('Input values filled (%)')
        ax.set_title('Bounded payload checks; selected by source combination, not population weighted')
        ax.set_ylim(-1, 101)
        fig.tight_layout()
        fig.savefig(output / 'payload-fill.png', dpi=160)
        plt.close(fig)


def write_summary(output, manifest, fields, summaries, checks, reads):
    """Save a compact, reproducible report alongside the full per-field JSON."""
    regions = {field['name']: field.get('region', 'custom') for field in fields}
    groups = {}
    for region in sorted(set(regions.values())):
        rows = [row for row in summaries if regions[row['field']] == region]
        sources, combinations, scalars = Counter(), Counter(), Counter()
        for row in rows:
            sources.update(row['source_presence'])
            combinations.update(row['source_combinations'])
            scalars.update({key: value['counts'].get('usable', 0) for key, value in row['scalars'].items()})
        groups[region] = {
            'fields': sorted({row['field'] for row in rows}),
            'rows': sum(row['rows'] for row in rows),
            'anchor_rows': sum(row['anchor_rows'] for row in rows),
            'outer_only_rows': sum(row['outer_only_rows'] for row in rows),
            'single_source_rows': sum(value for key, value in combinations.items() if '+' not in key),
            'source_presence': dict(sources), 'source_combinations': dict(combinations),
            'usable_scalars': dict(scalars),
        }
    quality = {}
    for name in sorted({name for check in checks for name in check.get('modalities', {})}):
        values = [check['modalities'][name] for check in checks if name in check.get('modalities', {})]
        quality[name] = {
            'records': len(values), 'tokens': sum(value['tokens'] for value in values),
            'nonfinite_token_records': sum(not value['finite_tokens'] for value in values),
            'all_filled_records': sum(value.get('all_input_filled', False) for value in values),
            'nonfinite_input_values': sum(value.get('nonfinite_input_values', 0) for value in values),
            'fill_fraction': describe([value['filled_values']/value['input_values']
                                       for value in values if 'filled_values' in value]),
        }
    byte_counts, request_counts = Counter(), Counter()
    log = output / 'bytes.rank0.main.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            item = json.loads(line)
            byte_counts[item['source']] += item['bytes']
            request_counts[item['source']] += 1
    validation = Counter()
    for row in summaries:
        validation.update({key: value for key, value in row['match_validation'].items() if isinstance(value, int)})
    compact = [{**{key: row[key] for key in ('field', 'path', 'rows', 'anchor_rows', 'outer_only_rows',
                                           'source_presence', 'source_combinations', 'repeated_source_tuples')},
                'scalar_statuses': {key: value['counts'] for key, value in row['scalars'].items()},
                'row_regions': row.get('row_regions', {})} for row in summaries]
    save(output / 'summary.json', {
        'schema_version': 2, 'code_commit': manifest['code_commit'], 'script_sha256': manifest['script_sha256'],
        'recipe_sha256': manifest['recipe_sha256'], 'catalogs': manifest['catalogs'], 'packages': manifest['packages'],
        'fields': fields, 'groups': groups, 'coverage': compact, 'join_validation': dict(validation),
        'payload_candidate_records': manifest['payload_candidate_records'],
        'payload_selected_records': len(checks), 'payload_status_counts': manifest['payload_status_counts'],
        'payload_samples_by_field': dict(Counter(check['field'] for check in checks)),
        'payload_quality': quality,
        'payload_budget_skips': {name: sum(entry['status'] == 'skipped_budget' for entry in entries)
                                 for name, entries in reads.items()},
        'projected_payload_bytes': {name: sum(entry['projected_uncompressed_bytes'] for entry in entries
                                             if entry['status'] == 'read') for name, entries in reads.items()},
        'returned_range_bytes': dict(byte_counts), 'returned_range_requests': dict(request_counts),
        'validation': manifest['validation'], 'elapsed_seconds': manifest['elapsed_seconds'],
        'artifact_sha256': {name: fingerprint(output/name) for name in
                           ('fields.json', 'coverage.json', 'payload-checks.json', 'payload-reads.json')},
    })


def validate_catalogs(configured):
    """Require pinned North/South inputs with outer payloads and left-only labels."""
    if [s.catalogs[0]["name"] for s in configured.stars] != ["legacy", "legacy_south"]:
        raise ValueError("This stratified audit requires the North/South anchor recipe")
    if configured.spec["how"] != "outer":
        raise ValueError("This audit requires default outer joins")
    for cat in configured.catalogs:
        expected = "left" if cat["decoder"] == "scalars" else "outer"
        if cat.get("how", configured.spec["how"]) != expected:
            raise ValueError("This audit requires outer payload joins and left scalar-only joins")
        if not re.search(r"@[0-9a-f]{40}(?:/|$)", cat["repo"]):
            raise ValueError("Audit inputs must use immutable 40-character revisions")


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    os.environ[telemetry.TELEMETRY_DIR_ENV] = str(output)
    telemetry.install_byte_probe()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    streaming._unregister_hf_array2d_extension()
    recipe_text = args.recipe.read_text(encoding="utf-8")
    shutil.copyfile(args.recipe, output / "recipe.yaml")
    recipe = yaml.safe_load(recipe_text)
    spec = recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    config = AstroPT3Config(**{k: v for k, v in recipe["model"]["model_config"].items()
                              if k != "is_astropt3_config"})
    configured = streaming.CatalogStream(spec, config)
    sequencer = ObjectSequencer(config)
    validate_catalogs(configured)
    manifest = {"schema_version": 1, "status": "running", "recipe_sha256": fingerprint(args.recipe),
                "seed": args.seed, "argv": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "packages": {p: version(p) for p in ("lsdb", "hats", "numpy", "pyarrow", "torch",
                                                      "huggingface-hub", "astropy", "mocpy")},
                "catalogs": configured.catalogs}
    root = Path(__file__).resolve().parents[2]
    manifest["code_commit"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    manifest["script_sha256"] = fingerprint(__file__)
    shutil.copyfile(__file__, output / 'audit_script.py')
    save(output / "manifest.json", manifest)
    opened, primary_uris, metadata, projections = {}, {}, {}, {}
    cache = args.metadata_cache.resolve()
    for cat in configured.catalogs:
        print(f"Opening {cat['name']} metadata", flush=True)
        telemetry._REPO_TO_SOURCE[cat["repo"].split("@")[0]] = cat["name"]
        base, primary_uris[cat['name']], metadata[cat['name']], projections[cat['name']] = open_projected(cat, cache)
        opened[cat['name']] = base
        save(output / 'metadata.json', metadata)
        margin = base.margin
        is_anchor = cat['name'] in {star.catalogs[0]['name'] for star in configured.stars}
        factor = 1 if is_anchor or cat.get('how') == 'left' else 2
        threshold = margin.hc_structure.catalog_info.margin_threshold if margin is not None else None
        if threshold is None or threshold < factor * configured.spec["radius_arcsec"]:
            raise ValueError(f"{cat['name']} has no sufficient margin")
    print('Planning exact recipe footprints and star joins', flush=True)
    streams = configured._streams(opened, None, args.seed)
    fields = json.loads(args.fields.read_text(encoding='utf-8')) if args.fields else choose_fields(
        opened, streams, args.fields_per_region, args.radius_arcsec, args.seed, args.field_separation_deg)
    validate_fields(fields)
    save(output / "fields.json", fields)
    summaries, selected = [], []
    with dask.config.set(scheduler="synchronous"):
        for field in fields:
            for star, stream in zip(configured.stars, streams, strict=True):
                path = star.catalogs[0]["name"]
                pixels = cone_pixels(stream._pixels, field["ra"], field["dec"], field["radius_arcsec"])
                if len(pixels) > args.max_partitions:
                    raise ValueError(f"{field['name']}/{path}: {len(pixels)} output partitions exceeds limit")
                print(f"Projecting {field['name']}/{path}: {len(pixels)} output partitions", flush=True)
                frames = [in_cone(stream._delayed_partition(pixel).compute(), field) for pixel in pixels]
                frame = pd.concat(frames) if frames else stream._assembly.meta.copy()
                file = output / f"{field['name']}-{path}.parquet"
                frame.to_parquet(file)
                summary = summarize(frame, star, sequencer)
                summary['match_validation'] = verify_star(
                    frame, star, opened, stream.footprint, field, configured.spec['radius_arcsec'])
                if len(frame):
                    north = _data_coverage(opened['legacy']).contains_lonlat(
                        frame['_ra'].to_numpy(dtype=float) * u.deg, frame['_dec'].to_numpy(dtype=float) * u.deg)
                    south = _data_coverage(opened['legacy_south']).contains_lonlat(
                        frame['_ra'].to_numpy(dtype=float) * u.deg, frame['_dec'].to_numpy(dtype=float) * u.deg)
                    summary['row_regions'] = {'north_only': int((north & ~south).sum()),
                                              'south_only': int((south & ~north).sum()),
                                              'overlap': int((north & south).sum()),
                                              'neither': int((~north & ~south).sum())}
                summary.update(field=field["name"], path=path,
                               output_partitions=[[int(p.order), int(p.pixel)] for p in pixels],
                               projection_sha256=fingerprint(file))
                summaries.append(summary)
                selected.extend((field["name"], star, row) for row in select_rows(frame, star, args.payloads_per_combination))
                save(output / "coverage.json", summaries)
    save(output / "coverage.json", summaries)
    manifest['payload_candidate_records'] = len(selected)
    selected = bound_payload_selection(selected, args.max_payload_records)
    manifest['payload_selected_records'] = len(selected)
    payload_dir = output / "payloads"
    payload_dir.mkdir()
    payloads, payload_log = {}, {}
    for cat in configured.catalogs:
        requested = {}
        for _, star, row in selected:
            for i, source in enumerate(star.catalogs):
                if source["name"] != cat["name"]:
                    continue
                suffix = "" if i == 0 else "_" + cat["name"]
                value = row.get(streaming._ID_COLUMNS.get(cat["decoder"], "object_id") + suffix)
                if present(value):
                    requested[str(value)] = (float(row['ra' + suffix]), float(row['dec' + suffix]))
        print(f"Payload sample {cat['name']}: {len(requested)} requested sources", flush=True)
        payloads[cat["name"]], payload_log[cat["name"]] = fetch_payloads(
            opened[cat["name"]], primary_uris[cat['name']], cat, requested, payload_dir,
            args.max_group_mib, args.max_payload_mib_per_catalog, projections[cat['name']])
        save(output / "payload-reads.json", payload_log)
    checks = check_payloads(selected, payloads, sequencer, args.seed)
    save(output / "payload-checks.json", checks)
    plot_results(output, summaries, checks)
    manifest.update(status="complete", elapsed_seconds=time.monotonic() - started,
                    payload_status_counts=dict(Counter(c['status'] for c in checks)),
                    validation=('passed' if checks else 'coverage_only') if all(c['status'] == 'passed' for c in checks) else 'failed',
                    rows=sum(s['rows'] for s in summaries))
    if manifest['validation'] != 'failed' and any(
            row['match_validation']['outer_ids_withheld_for_halo_ties'] for row in summaries):
        manifest['validation'] = 'partial_ties'
    save(output / "manifest.json", manifest)
    write_summary(output, manifest, fields, summaries, checks, payload_log)
    print(json.dumps({k: manifest[k] for k in ('status', 'rows', 'elapsed_seconds', 'payload_status_counts')}), flush=True)
    if manifest['validation'] not in ('passed', 'coverage_only'):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path("configs/nanotron/astropt3-70m.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--metadata-cache", type=Path, required=True,
                        help="Reusable immutable metadata and scalar/coordinate projections; no image archive")
    parser.add_argument("--fields", type=Path, help="Replay saved fields.json")
    parser.add_argument("--fields-per-region", type=int, default=2)
    parser.add_argument("--field-separation-deg", type=float, default=5)
    parser.add_argument("--radius-arcsec", type=float, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-partitions", type=int, default=12)
    parser.add_argument("--payloads-per-combination", type=int, default=1)
    parser.add_argument("--max-payload-records", type=int, default=64,
                        help="Global cap on the source/field-balanced purposive payload sample")
    parser.add_argument("--max-group-mib", type=float, default=192)
    parser.add_argument("--max-payload-mib-per-catalog", type=float, default=512)
    args = parser.parse_args()
    for name in ("fields_per_region", "field_separation_deg", "radius_arcsec", "max_partitions", "max_group_mib", "max_payload_mib_per_catalog", "max_payload_records"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.payloads_per_combination < 0:
        parser.error("--payloads-per-combination cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
