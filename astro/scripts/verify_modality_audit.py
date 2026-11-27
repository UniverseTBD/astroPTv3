#!/usr/bin/env python
"""Verify a completed audit and replay its saved payloads without network access."""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

import audit_modalities as audit
from astropt3.data.packing import PackedCollator
from astropt3.tokenization import BOS_ID, PAD_ID


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_primary_values(row, star, payloads):
    """Check joined coordinates/scalars against the independently fetched primary rows."""
    coordinates = scalars = 0
    for index, cat in enumerate(star.catalogs):
        suffix = '' if index == 0 else '_' + cat['name']
        column = audit.streaming._ID_COLUMNS.get(cat['decoder'], 'object_id')
        identifier = row.get(column + suffix)
        if not audit.present(identifier):
            continue
        raw = payloads[cat['name']][str(identifier)]
        for column in ['ra', 'dec', *cat['scalar_columns'].values()]:
            left, right = row.get(column + suffix), raw.get(column)
            equal = (pd.isna(left) and pd.isna(right)) if pd.isna(left) or pd.isna(right) else left == right
            require(equal, f'Joined value differs from primary row: {cat["name"]}/{identifier}/{column}')
            coordinates += column in ('ra', 'dec')
            scalars += column not in ('ra', 'dec')
    return coordinates, scalars


def verify_run(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest['status'] == 'complete' and manifest['validation'] == 'passed',
            'Audit must have completed with validation=passed')
    require(audit.fingerprint(root / 'audit_script.py') == manifest['script_sha256'],
            'Measurement script checksum differs')
    require(audit.fingerprint(root / 'recipe.yaml') == manifest['recipe_sha256'],
            'Recipe checksum differs')
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    require(summary['schema_version'] == 2, 'Expected audit summary schema 2')
    for name in ('fields.json', 'coverage.json', 'payload-checks.json', 'payload-reads.json'):
        require(audit.fingerprint(root / name) == summary['artifact_sha256'][name], f'Checksum differs: {name}')

    recipe = yaml.safe_load((root / 'recipe.yaml').read_text(encoding='utf-8'))
    config = audit.AstroPT3Config(**{key: value for key, value in recipe['model']['model_config'].items()
                                   if key != 'is_astropt3_config'})
    configured = audit.streaming.CatalogStream(recipe['data_stages'][0]['data']['dataset']['catalog_stream'], config)
    stars = {star.catalogs[0]['name']: star for star in configured.stars}
    audit.streaming._unregister_hf_array2d_extension()
    torch.set_num_threads(1)

    coverage = json.loads((root / 'coverage.json').read_text(encoding='utf-8'))
    checks = json.loads((root / 'payload-checks.json').read_text(encoding='utf-8'))
    reads = json.loads((root / 'payload-reads.json').read_text(encoding='utf-8'))
    require(sum(row['rows'] for row in coverage) == manifest['rows'], 'Record accounting differs')
    lookup, paths, projection_files = {}, {}, 0
    for row in coverage:
        field, path = row['field'], row['path']
        require(field.replace('_', '').replace('-', '').isalnum() and path in stars, 'Invalid field/path')
        file = root / f'{field}-{path}.parquet'
        require(audit.fingerprint(file) == row['projection_sha256'], f'Projection checksum differs: {file.name}')
        projection_files += 1
        table = pq.ParquetFile(file).read()
        require(table.num_rows == row['rows'] == row['anchor_rows'] + row['outer_only_rows'],
                f'Projection row accounting differs: {file.name}')
        require(sum(row['source_combinations'].values()) == row['rows'], 'Combination accounting differs')
        require(all(sum(value['counts'].values()) == row['rows'] for value in row['scalars'].values()),
                'Scalar denominator differs')
        match = row['match_validation']
        require(match['checked_anchor_rows'] == row['anchor_rows'] and match['checked_outer_rows'] == row['outer_only_rows']
                and match['outer_ids_withheld_for_halo_ties'] == 0, 'Incomplete join validation')
        star = stars[path]
        for record in table.to_pylist():
            ids = audit.source_ids(record, star)
            identity = (field, path, tuple(sorted(ids.items())))
            require(identity not in lookup, 'Repeated source-ID tuple within a field/path')
            lookup[identity] = record
            for name, identifier in ids.items():
                paths.setdefault(name, {}).setdefault(path, set()).add(identifier)
    duplicates = {}
    for source, anchor_paths in paths.items():
        for left, right in combinations(sorted(anchor_paths), 2):
            common = anchor_paths[left] & anchor_paths[right]
            duplicates[f'{source}:{left}/{right}'] = len(common)
            require(not common, f'Shared source IDs emitted on both anchor paths: {source}')

    payloads, payload_files = {}, 0
    for source, entries in reads.items():
        payloads[source] = {}
        for entry in entries:
            require(entry['status'] == 'read', f'Payload read did not complete: {source}')
            require(Path(entry['cache']).name == entry['cache'], 'Invalid payload cache filename')
            file = root / 'payloads' / entry['cache']
            require(audit.fingerprint(file) == entry['sha256'], f'Payload checksum differs: {file.name}')
            payload_files += 1
            spec = next(cat for cat in configured.catalogs if cat['name'] == source)
            column = audit.streaming._ID_COLUMNS.get(spec['decoder'], 'object_id')
            for record in pq.ParquetFile(file).read().to_pylist():
                identifier = str(record[column])
                require(identifier not in payloads[source], f'Repeated saved payload ID: {source}/{identifier}')
                payloads[source][identifier] = record

    require(checks and len(checks) == manifest['payload_selected_records'], 'Payload selection accounting differs')
    selected = []
    coordinate_checks = scalar_checks = 0
    for check in checks:
        require(check['status'] == 'passed', 'Failed payload check')
        identity = (check['field'], check['path'], tuple(sorted(check['source_ids'].items())))
        require(identity in lookup, 'Selected record is absent from the field projections')
        star, row = stars[check['path']], lookup[identity]
        coordinates, scalars = check_primary_values(row, star, payloads)
        coordinate_checks += coordinates
        scalar_checks += scalars
        selected.append((check['field'], star, row))
    sequencer = audit.ObjectSequencer(config)
    replay = audit.check_payloads(selected, payloads, sequencer, manifest['seed'])
    require(replay == checks, 'Saved payload checks differ from offline decoder/sequencer replay')

    audit.streaming._DEQUANTIZE = np.random.default_rng(manifest['seed'])
    torch.manual_seed(manifest['seed'])
    objects = [sequencer.build(audit.decode_selected(row, star, payloads)) for _, star, row in selected]
    sequence_length = recipe['tokens']['sequence_length']
    batch = PackedCollator(config, seq_len=sequence_length)(objects)
    nonpad = batch['input_ids'] != PAD_ID
    require(torch.equal(batch['input_ids'][nonpad], torch.cat([obj.input_ids for obj in objects])),
            'Packed object tokens were lost or reordered')
    require(torch.equal(batch['position_ids'][nonpad], torch.cat([torch.arange(len(obj)) for obj in objects])),
            'Packed object position resets differ')
    require(int((batch['input_ids'] == BOS_ID).sum()) == len(objects), 'Packed object count differs')
    counts = {}
    for name, values in batch['modality_values'].items():
        expected = torch.cat([obj.values[name] for obj in objects if name in obj.values])
        positions = torch.cat([obj.positions[name] for obj in objects if name in obj.positions])
        require(torch.isfinite(values).all() and torch.equal(values, expected), f'Packed values differ: {name}')
        require(torch.isfinite(batch['modality_positions'][name]).all()
                and torch.equal(batch['modality_positions'][name], positions), f'Packed positions differ: {name}')
        require(int(batch['modality_masks'][name].sum()) == len(values) == len(positions),
                f'Packed mask/value alignment differs: {name}')
        counts[name] = len(values)
    return {'status': 'passed', 'rows': manifest['rows'], 'projection_files_verified': projection_files,
            'payload_files_verified': payload_files, 'cross_path_shared_ids': duplicates,
            'primary_coordinate_value_checks': coordinate_checks, 'primary_scalar_value_checks': scalar_checks,
            'replayed_records': len(objects), 'packed_rows': len(batch['input_ids']),
            'sequence_length': sequence_length, 'nonpad_tokens': int(nonpad.sum()),
            'modality_tokens': counts,
            'packing_scope': 'Sequential greedy integrity check, not streaming packer fill or throughput',
            'measurement_script_sha256': manifest['script_sha256'],
            'replay_script_sha256': audit.fingerprint(audit.__file__),
            'verifier_sha256': audit.fingerprint(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Completed local audit output directory')
    args = parser.parse_args()
    try:
        report = verify_run(args.run)
    except Exception as error:
        audit.save(args.run / 'verification.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    audit.save(args.run / 'verification.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
