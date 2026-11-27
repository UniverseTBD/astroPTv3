"""Audit denominators, missingness, finite joins, and payload quality on local data."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import dask
import lsdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from astropt3.configuration_astropt3 import AstroPT3Config
from astropt3.data.catalog_stream import CatalogStream, _Star
from astropt3.data.packing import ObjectSequencer

spec = spec_from_file_location("audit_modalities", Path(__file__).resolve().parents[1] / "scripts/audit_modalities.py")
audit = module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def scalar_case():
    config = AstroPT3Config(modalities=[
        {"name": "gwh_mass_med_photoz", "family": "scalar", "source": "galaxies_with_hats",
         "record_keys": ["gwh_mass_med_photoz"], "input_size": 1, "patch_size": 1,
         "token_ids": [26, 27, 28]},
    ])
    catalogs = [
        {"name": "legacy", "repo": "offline/legacy", "decoder": "legacy_image", "record_key": "image"},
        {"name": "galaxies", "repo": "offline/galaxies", "decoder": "rgb_png", "record_key": "galaxy_image",
         "scalar_columns": {"gwh_mass_med_photoz": "mass"}},
    ]
    return config, _Star(catalogs, config.modality_registry()), ObjectSequencer(config)


def test_coverage_uses_all_emitted_rows_and_keeps_rejection_reasons(scalar_case):
    _, star, sequencer = scalar_case
    frame = pd.DataFrame([
        {"object_id": "a", "dr8_id_galaxies": "x", "mass_galaxies": -99.0},
        {"object_id": "b", "dr8_id_galaxies": None, "mass_galaxies": None},
        {"object_id": None, "dr8_id_galaxies": "y", "mass_galaxies": 10.0},
    ], dtype=object)
    report = audit.summarize(frame, star, sequencer)
    assert report["rows"] == 3
    assert report["anchor_rows"] == 2
    assert report["outer_only_rows"] == 1
    assert report["source_presence"] == {"legacy": 2, "galaxies": 2}
    scalar = report["scalars"]["gwh_mass_med_photoz"]
    assert scalar["counts"] == {"range_rejected": 1, "missing": 1, "usable": 1}
    assert scalar["raw_usable"]["median"] == 10


def test_scalar_zero_is_usable_and_nonfinite_is_separate(scalar_case):
    _, _, sequencer = scalar_case
    mod = sequencer.registry.get_config("gwh_mass_med_photoz")
    assert audit.scalar_status(mod, {mod.name: 0.0}, sequencer) == "usable"
    assert audit.scalar_status(mod, {mod.name: np.inf}, sequencer) == "nonfinite"
    assert audit.scalar_status(mod, {mod.name: -99}, sequencer) == "range_rejected"


def test_hash_sample_is_stable_when_rows_are_reordered(scalar_case):
    _, star, _ = scalar_case
    frame = pd.DataFrame([{"object_id": str(i), "dr8_id_galaxies": None} for i in range(10)])
    assert audit.select_rows(frame, star, 2) == audit.select_rows(frame.iloc[::-1], star, 2)
    assert len(audit.select_rows(frame, star, 2)) == 2
    assert audit.select_rows(frame, star, 0) == []


def test_spherical_aperture_wraps_ra():
    frame = pd.DataFrame({"_ra": [359.9999, 0.0001, 0.1], "_dec": [0., 0., 0.]})
    field = {"ra": 0., "dec": 0., "radius_arcsec": 1.}
    assert audit.in_cone(frame, field).index.tolist() == [0, 1]


@pytest.mark.parametrize('patch', [{'name': '../escape'}, {'radius_arcsec': 0}, {'ra': float('nan')}, {'dec': 91}])
def test_invalid_field_manifest_rejected(patch):
    field = {'name': 'field', 'ra': 10, 'dec': 0, 'radius_arcsec': 300} | patch
    with pytest.raises(ValueError):
        audit.validate_fields([field])


def test_real_margin_star_counts_outer_singletons(scalar_case):
    config, star, sequencer = scalar_case
    configured = CatalogStream({"anchors": [{**star.catalogs[0], "draw_weight": 1}],
                                "catalogs": [star.catalogs[1]], "how": "outer", "min_catalogs": 1}, config)

    def catalog(rows, name):
        frame = pa.Table.from_pylist(rows).to_pandas(types_mapper=pd.ArrowDtype)
        return lsdb.from_dataframe(frame, catalog_name=name, margin_threshold=10,
                                   lowest_order=0, highest_order=0, ra_column="ra", dec_column="dec")

    opened = {
        "legacy": catalog([{"object_id": "a", "ra": 10., "dec": 0.}], "legacy"),
        "galaxies": catalog([{"dr8_id": "x", "ra": 10.0001, "dec": 0., "mass": -99.},
                             {"dr8_id": "y", "ra": 10.01, "dec": 0., "mass": 10.}], "galaxies"),
    }
    stream, = configured._streams(opened, None, 42)
    with dask.config.set(scheduler="synchronous"):
        frames = [stream._delayed_partition(p).compute() for p in stream._pixels]
    frame = pd.concat(frames)
    report = audit.summarize(frame, star, sequencer)
    assert report["rows"] == 2
    assert report["source_combinations"] == {"galaxies": 1, "legacy+galaxies": 1}
    assert report["repeated_source_tuples"] == 0
    assert report["scalars"]["gwh_mass_med_photoz"]["counts"] == {"range_rejected": 1, "usable": 1}
    field = {'name': 'fixture', 'ra': 10., 'dec': 0., 'radius_arcsec': 60.}
    with dask.config.set(scheduler='synchronous'):
        checked = audit.verify_star(frame, star, opened, stream.footprint, field, 1.)
        assert checked['checked_anchor_partner_pairs'] == 1
        assert checked['checked_outer_rows'] == 1
        broken = frame.copy()
        broken.loc[broken['object_id'].notna(), 'dr8_id_galaxies'] = 'y'
        with pytest.raises(AssertionError, match='reference'):
            audit.verify_star(broken, star, opened, stream.footprint, field, 1.)
        with pytest.raises(AssertionError, match='core anchor rows'):
            audit.verify_star(frame[frame['object_id'].isna()], star, opened, stream.footprint, field, 1.)
        with pytest.raises(AssertionError, match='outer rows differ'):
            audit.verify_star(frame[frame['object_id'].notna()], star, opened, stream.footprint, field, 1.)


def test_primary_reference_checks_boundary_matches_and_two_radius_halo(scalar_case):
    config, star, _ = scalar_case
    configured = CatalogStream({'anchors': [{**star.catalogs[0], 'draw_weight': 1}],
                                'catalogs': [star.catalogs[1]], 'how': 'outer', 'min_catalogs': 1}, config)

    def catalog(rows, name):
        frame = pa.Table.from_pylist(rows).to_pandas(types_mapper=pd.ArrowDtype)
        return lsdb.from_dataframe(frame, catalog_name=name, margin_threshold=10,
                                   lowest_order=0, highest_order=0, ra_column='ra', dec_column='dec')

    opened = {
        'legacy': catalog([{'object_id': key, 'ra': 45 + offset / 3600, 'dec': 0.}
                           for key, offset in [('a', -.18), ('halo', 2.9), ('far', -360)]], 'legacy'),
        'galaxies': catalog([{'dr8_id': key, 'ra': 45 + offset / 3600, 'dec': 0., 'mass': 10.}
                             for key, offset in [('x', .18), ('y', .36), ('z', 1.95), ('w', 3.7),
                                                 ('far_right', -720)]], 'galaxies'),
    }
    # The core anchor and its nearest counterpart are in different order-0 pixels.
    assert audit.primary_pixel(opened['legacy'], 45 - .18/3600, 0).pixel != audit.primary_pixel(
        opened['galaxies'], 45 + .18/3600, 0).pixel
    stream, = configured._streams(opened, None, 42)
    field = {'name': 'boundary', 'ra': 45., 'dec': 0., 'radius_arcsec': 2.}
    with dask.config.set(scheduler='synchronous'):
        frame = audit.in_cone(pd.concat([stream._delayed_partition(p).compute() for p in stream._pixels]), field)
        result = audit.verify_star(frame, star, opened, stream.footprint, field, 1.)
    assert result['checked_anchor_rows'] == 1
    # y loses to x; z loses to w for an anchor outside the core. Both are outer.
    assert result['checked_outer_rows'] == 2
    assert set(frame.loc[frame['object_id'].isna(), 'dr8_id_galaxies']) == {'y', 'z'}


def test_capped_payload_selection_covers_sources_and_fields_stably(scalar_case):
    _, star, _ = scalar_case
    rows = [('f1', star, {'object_id': 'a', 'dr8_id_galaxies': 'x'}),
            ('f1', star, {'object_id': 'b', 'dr8_id_galaxies': None}),
            ('f2', star, {'object_id': 'c', 'dr8_id_galaxies': None}),
            ('f3', star, {'object_id': None, 'dr8_id_galaxies': 'z'})]
    selected = audit.bound_payload_selection(rows, 3)
    assert selected == audit.bound_payload_selection(list(reversed(rows)), 3)
    assert {item[0] for item in selected} == {'f1', 'f2', 'f3'}
    assert set().union(*(audit.source_ids(row, st).keys() for _, st, row in selected)) == {'legacy', 'galaxies'}
    assert audit.bound_payload_selection(rows, 0) == []


def test_invalid_png_is_reported_as_failed_payload(scalar_case):
    _, star, sequencer = scalar_case
    selected = [('fixture', star, {'object_id': None, 'dr8_id_galaxies': 'bad', 'mass_galaxies': 10.})]
    payloads = {'legacy': {}, 'galaxies': {'bad': {'image': {'bytes': b'not an image', 'path': 'bad.png'}}}}
    checks = audit.check_payloads(selected, payloads, sequencer, 42)
    assert checks[0]['status'] == 'decode_or_sequence_error'
    assert 'UnidentifiedImageError' in checks[0]['error']


def test_halo_ties_are_explicit_without_hiding_other_missing_rows(scalar_case):
    config, star, _ = scalar_case
    configured = CatalogStream({'anchors': [{**star.catalogs[0], 'draw_weight': 1}],
                                'catalogs': [star.catalogs[1]], 'how': 'outer', 'min_catalogs': 1}, config)

    def catalog(rows, name):
        frame = pa.Table.from_pylist(rows).to_pandas(types_mapper=pd.ArrowDtype)
        return lsdb.from_dataframe(frame, catalog_name=name, margin_threshold=10,
                                   lowest_order=0, highest_order=0, ra_column='ra', dec_column='dec')

    opened = {
        'legacy': catalog([{'object_id': key, 'ra': 10 + offset / 3600, 'dec': 0.}
                           for key, offset in [('a', 0), ('halo', 2.4)]], 'legacy'),
        'galaxies': catalog([{'dr8_id': key, 'ra': 10 + offset / 3600, 'dec': 0., 'mass': 10.}
                             for key, offset in [('x', 0), ('y', 1.9), ('z', 2.9), ('q', -1.5)]], 'galaxies'),
    }
    stream, = configured._streams(opened, None, 42)
    field = {'name': 'ties', 'ra': 10., 'dec': 0., 'radius_arcsec': 2.}
    with dask.config.set(scheduler='synchronous'):
        frame = audit.in_cone(pd.concat([stream._delayed_partition(p).compute() for p in stream._pixels]), field)
        result = audit.verify_star(frame, star, opened, stream.footprint, field, 1.)
        assert result['distance_ties'] == 1
        assert result['outer_ids_withheld_for_halo_ties'] == 1
        assert result['checked_outer_rows'] == 1
        broken = frame[frame['dr8_id_galaxies'] != 'q']
        with pytest.raises(AssertionError, match='outer rows differ'):
            audit.verify_star(broken, star, opened, stream.footprint, field, 1.)


def test_offline_verifier_rejects_tampered_recipe_without_executing_snapshot(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import verify_modality_audit as verifier

    snapshot = tmp_path / 'audit_script.py'
    snapshot.write_text("raise RuntimeError('Saved source must never be executed')\n")
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text('original: true\n')
    audit.save(tmp_path / 'manifest.json', {
        'status': 'complete', 'validation': 'passed', 'script_sha256': audit.fingerprint(snapshot),
        'recipe_sha256': audit.fingerprint(recipe),
    })
    recipe.write_text('tampered: true\n')
    with pytest.raises(ValueError, match='Recipe checksum differs'):
        verifier.verify_run(tmp_path)


def test_run_preserves_recipe_bytes_for_both_line_endings(tmp_path, monkeypatch):
    def stop_after_snapshot(*args):
        raise RuntimeError('stop before opening catalogues')

    monkeypatch.setenv(audit.telemetry.TELEMETRY_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(audit.telemetry, 'install_byte_probe', lambda: None)
    monkeypatch.setattr(audit.streaming, '_unregister_hf_array2d_extension', lambda: None)
    monkeypatch.setattr(audit.streaming, 'CatalogStream', stop_after_snapshot)
    monkeypatch.setattr(audit, 'AstroPT3Config', lambda **kwargs: object())
    recipe = audit.yaml.safe_dump({'data_stages': [{'data': {'dataset': {'catalog_stream': {}}}}],
                                  'model': {'model_config': {}}})
    for index, newline in enumerate(('\n', '\r\n')):
        source, output = tmp_path / f'recipe-{index}.yaml', tmp_path / f'run-{index}'
        source.write_bytes(recipe.replace('\n', newline).encode('utf-8'))
        with pytest.raises(RuntimeError, match='stop before opening catalogues'):
            audit.run(SimpleNamespace(output=output, recipe=source, seed=42))
        assert audit.fingerprint(output / 'recipe.yaml') == audit.fingerprint(source)


def test_offline_verifier_checks_joined_scalar_values_against_primary(scalar_case, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import verify_modality_audit as verifier

    _, star, _ = scalar_case
    row = {'object_id': None, 'dr8_id_galaxies': 'x', 'ra_galaxies': 10., 'dec_galaxies': 0., 'mass_galaxies': None}
    payloads = {'galaxies': {'x': {'ra': 10., 'dec': 0., 'mass': float('nan')}}}
    assert verifier.check_primary_values(row, star, payloads) == (2, 1)
    payloads['galaxies']['x']['mass'] = 10.
    with pytest.raises(ValueError, match='Joined value differs from primary'):
        verifier.check_primary_values(row, star, payloads)


def test_all_filled_image_is_distinguished_from_finite_training_tokens():
    config = AstroPT3Config(modalities=[
        {"name": "legacy_south_r", "family": "image", "source": "legacy_south", "band": "des-r",
         "record_keys": ["legacy_south_image"], "input_size": 64, "patch_size": 8,
         "image_crop": 96},
    ])
    record = {"object_id": "all-filled", "legacy_south_image": {
        "band": ["des-r"], "flux": np.zeros((1, 160, 160), dtype=np.float32),
        "fill_mask": np.ones((1, 160, 160), dtype=bool),
    }}
    report = audit.payload_quality(record, ObjectSequencer(config))
    image = report["modalities"]["legacy_south_r"]
    assert image["finite_tokens"]
    assert image["all_input_filled"]
    assert image["filled_values"] == 96 * 96
    assert image["tokens"] == 144


def test_payload_budget_is_checked_before_loading_large_columns(tmp_path, scalar_case):
    _, star, _ = scalar_case
    source = star.catalogs[1]
    # A real tiny Parquet file, accessed through a local stand-in filesystem.
    relative = "dataset/Norder=0/Dir=0/Npix=4.parquet"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    pq = audit.pq
    pq.write_table(pa.Table.from_pylist([{"dr8_id": "a", "ra": 10., "dec": 0., "mass": 10.,
                                         "image": {"bytes": b"payload", "path": "a.png"}}]), path)
    from types import SimpleNamespace
    from unittest.mock import patch
    fake = SimpleNamespace(open=lambda uri, **kwargs: open(uri, "rb"))
    pixel = SimpleNamespace(order=0, pixel=4)
    with patch.object(audit, "HfFileSystem", return_value=fake), patch.object(audit, "primary_pixel", return_value=pixel):
        found, log = audit.fetch_payloads(None, str(tmp_path), source, {"a": (10., 0.)}, tmp_path,
                                         max_group_mib=0, max_total_mib=10)
    assert found == {}
    assert log[0]["status"] == "skipped_budget"


def test_payload_lookup_reuses_projection_and_checks_source_row_order(tmp_path, scalar_case):
    _, star, _ = scalar_case
    source = star.catalogs[1]
    relative = 'dataset/Norder=0/Dir=0/Npix=4.parquet'
    primary, projected, output = (tmp_path / p for p in ('primary', 'projection', 'output'))
    for directory in (primary, projected):
        (directory / relative).parent.mkdir(parents=True)
    output.mkdir()
    rows = [{'dr8_id': name, 'ra': 10., 'dec': 0., 'mass': 10.,
             'image': {'bytes': name.encode(), 'path': name + '.png'}} for name in ('a', 'b')]
    audit.pq.write_table(pa.Table.from_pylist(rows), primary / relative, row_group_size=1)
    audit.pq.write_table(pa.table({'dr8_id': ['a', 'b']}), projected / relative)
    from types import SimpleNamespace
    from unittest.mock import patch
    fake = SimpleNamespace(open=lambda uri, **kwargs: open(uri, 'rb'))
    pixel = SimpleNamespace(order=0, pixel=4)
    with patch.object(audit, 'HfFileSystem', return_value=fake), patch.object(audit, 'primary_pixel', return_value=pixel):
        found, log = audit.fetch_payloads(None, str(primary), source, {'b': (10., 0.)}, output, 1, 1, projected)
        assert set(found) == {'b'}
        assert log[0]['row_group'] == 1
        audit.pq.write_table(pa.table({'dr8_id': ['b', 'a']}), projected / relative)
        with pytest.raises(ValueError, match='row order changed'):
            audit.fetch_payloads(None, str(primary), source, {'b': (10., 0.)}, output, 1, 1, projected)


def current_recipe():
    path = Path(__file__).resolve().parents[1] / 'configs/nanotron/astropt3-70m.yaml'
    recipe = audit.yaml.safe_load(path.read_text(encoding='utf-8'))
    config = AstroPT3Config(**{k: v for k, v in recipe['model']['model_config'].items()
                              if k != 'is_astropt3_config'})
    return config, CatalogStream(recipe['data_stages'][0]['data']['dataset']['catalog_stream'], config)


def test_current_recipe_accepts_pinned_scalar_labels_without_network():
    config, configured = current_recipe()
    audit.validate_catalogs(configured)
    labels = next(cat for cat in configured.catalogs if cat['name'] == 'provabgs')
    assert labels['decoder'] == 'scalars' and labels['how'] == 'left'
    assert set(labels['scalar_columns']) <= set(config.modality_registry().names())
    labels['how'] = 'outer'
    with pytest.raises(ValueError, match='left scalar-only joins'):
        audit.validate_catalogs(configured)


@pytest.mark.parametrize('flag', [None, np.nan, np.inf, -np.inf, 1., 0.])
@pytest.mark.parametrize('name', ['Z', 'sdss_vdisp', 'gwh_total_sfr_median', 'hsc_cmodel_mag'])
def test_quality_flags_agree_with_production_decoder(name, flag):
    config, configured = current_recipe()
    sequencer = ObjectSequencer(config)
    mod = sequencer.registry.get_config(name)
    values = {'Z': .1, 'sdss_vdisp': 200., 'gwh_total_sfr_median': 1., 'hsc_cmodel_mag': 20.}
    record = {key: values[name] for key in mod.record_keys}
    record.update({key: flag for key in audit.QUALITY_GATES[name]})
    star = configured.stars[0]
    row = {}
    for index, cat in enumerate(star.catalogs):
        suffix = '' if index == 0 else '_' + cat['name']
        row[audit.streaming._ID_COLUMNS.get(cat['decoder'], 'object_id') + suffix] = cat['name']
        for key, column in cat['scalar_columns'].items():
            row[column + suffix] = record.get(key)
    actual = audit.scalar_status(mod, audit.scalar_record(row, star), sequencer)
    decoded = star.decode_row(row)['record']
    assert (actual == 'usable') == (decoded is not None and sequencer._scalar_value(name, decoded) is not None)
    assert actual == ('usable' if flag == 0 else 'quality_rejected')


def test_mixed_star_checks_labels_without_emitting_label_only_rows(scalar_case):
    config, original, _ = scalar_case
    labels = {'name': 'provabgs', 'repo': 'offline/labels', 'decoder': 'scalars', 'how': 'left',
              'scalar_columns': {'provabgs_LOG_MSTAR': 'LOG_MSTAR'}}
    configured = CatalogStream({'anchors': [{**original.catalogs[0], 'draw_weight': 1}],
                                'catalogs': [original.catalogs[1], labels], 'how': 'outer', 'min_catalogs': 1}, config)
    star = configured.stars[0]

    def catalog(rows, name):
        frame = pa.Table.from_pylist(rows).to_pandas(types_mapper=pd.ArrowDtype)
        return lsdb.from_dataframe(frame, catalog_name=name, margin_threshold=10,
                                   lowest_order=0, highest_order=0, ra_column='ra', dec_column='dec')

    opened = {
        'legacy': catalog([{'object_id': 'a', 'ra': 10., 'dec': 0.}], 'legacy'),
        'galaxies': catalog([{'dr8_id': 'x', 'ra': 10.0001, 'dec': 0., 'mass': 10.},
                             {'dr8_id': 'y', 'ra': 10.01, 'dec': 0., 'mass': 11.}], 'galaxies'),
        'provabgs': catalog([{'object_id': key, 'ra': ra, 'dec': 0., 'LOG_MSTAR': 10.}
                             for key, ra in [('matched', 10.0002), ('near_outer', 10.01),
                                             ('isolated', 10.015)]], 'provabgs'),
    }
    stream, = configured._streams(opened, None, 42)
    field = {'name': 'mixed', 'ra': 10., 'dec': 0., 'radius_arcsec': 60.}
    with dask.config.set(scheduler='synchronous'):
        frame = pd.concat([stream._delayed_partition(p).compute() for p in stream._pixels])
        checked = audit.verify_star(frame, star, opened, stream.footprint, field, 1.)
        assert len(frame) == 2
        assert checked['checked_anchor_partner_pairs'] == 2
        assert checked['checked_outer_rows'] == 1
        assert frame['object_id_provabgs'].dropna().tolist() == ['matched']
        broken = frame.copy()
        broken.loc[broken['object_id'].notna(), 'object_id_provabgs'] = 'isolated'
        with pytest.raises(AssertionError, match='reference'):
            audit.verify_star(broken, star, opened, stream.footprint, field, 1.)
        extra = {column: None for column in frame.columns}
        extra.update(object_id_provabgs='isolated', ra_provabgs=10.015, dec_provabgs=0.,
                     _ra=10.015, _dec=0.)
        broken = pd.concat([frame, pd.DataFrame([extra])])
        with pytest.raises(AssertionError, match='left-joined labels'):
            audit.verify_star(broken, star, opened, stream.footprint, field, 1.)


def test_scalar_only_primary_rows_are_saved_decoded_and_verified(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import verify_modality_audit as verifier

    config, configured = current_recipe()
    star = configured.stars[0]
    source = next(cat for cat in star.catalogs if cat['name'] == 'provabgs')
    relative = 'dataset/Norder=0/Dir=0/Npix=4.parquet'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    primary = dict(object_id='label', ra=10., dec=0., LOG_MSTAR=10., Z_MW=.01,
                   TAGE_MW=5., AVG_SFR=1., MAG_W1=18.)
    audit.pq.write_table(pa.Table.from_pylist([primary]), path)
    monkeypatch.setattr(audit, 'HfFileSystem', lambda **kwargs: SimpleNamespace(
        open=lambda uri, **options: open(uri, 'rb')))
    monkeypatch.setattr(audit, 'primary_pixel', lambda *args: SimpleNamespace(order=0, pixel=4))
    found, log = audit.fetch_payloads(None, str(tmp_path), source, {'label': (10., 0.)}, tmp_path, 1, 1)
    assert found == {'label': primary} and log[0]['status'] == 'read'
    row = {'object_id': 'anchor', **{key + '_provabgs': value for key, value in primary.items()}}
    payloads = {cat['name']: {} for cat in star.catalogs}
    payloads['legacy']['anchor'] = {'ra': 10., 'dec': 0.}
    payloads['provabgs'] = found
    # No image data is needed to decode a matched scalar-only partner.
    record = star.decode_row(row)['record']
    sequencer = ObjectSequencer(config)
    assert set(sequencer.build(record).values) == set(source['scalar_columns'])
    row.update(ra=10., dec=0.)
    assert verifier.check_primary_values(row, star, payloads)[1] >= len(source['scalar_columns'])
    row['LOG_MSTAR_provabgs'] = 99.
    with pytest.raises(ValueError, match='Joined value differs from primary'):
        verifier.check_primary_values(row, star, payloads)


@pytest.mark.parametrize('published_mask', [True, False])
def test_fill_counts_follow_image_resize(published_mask):
    config = AstroPT3Config(modalities=[{
        'name': 'legacy_south_r', 'family': 'image', 'source': 'legacy_south', 'band': 'des-r',
        'record_keys': ['image'], 'input_size': 4, 'patch_size': 2, 'image_resize': 4,
    }])
    flux = np.ones((1, 8, 8), dtype=np.float32)
    flux[:, :2, :2] = 0
    image = {'band': ['des-r'], 'flux': flux}
    if published_mask:
        image['fill_mask'] = np.zeros_like(flux, dtype=bool)
        image['fill_mask'][:, 7, 7] = True
    result = audit.payload_quality({'object_id': 'a', 'image': image}, ObjectSequencer(config))
    details = result['modalities']['legacy_south_r']
    assert details['input_values'] == details['values'] == 16
    assert details['filled_values'] == 1
    assert details['finite_tokens']


def test_spectrum_fill_counts_include_zeros_and_padding():
    config = AstroPT3Config(modalities=[{
        'name': 'sdss', 'family': 'spectrum', 'source': 'sdss', 'record_keys': ['spectrum'],
        'input_size': 4, 'patch_size': 4,
    }])
    record = {'object_id': 'a', 'spectrum': {
        'flux': np.array([1., 0., 2., 3., 4.]), 'lambda': 10 ** (3.6 + np.arange(5) * 1e-4),
        'mask': np.array([False, False, False, True, False]),
    }}
    result = audit.payload_quality(record, ObjectSequencer(config))['modalities']['sdss']
    assert result['input_values'] == result['values'] == 8
    assert result['padding_values'] == 3
    assert result['filled_values'] == 5
    assert result['finite_tokens']
