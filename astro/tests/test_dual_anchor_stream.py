"""Two real in-memory LSDB stars, no Hub or distributed worker needed."""

from contextlib import closing, nullcontext
from itertools import islice

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from lsdb.loaders.dataframe.from_dataframe import from_dataframe

from astropt3.configuration_astropt3 import DEFAULT_MODALITIES, AstroPT3Config
from astropt3.data import catalog_stream as stream


def survey(name, key, weight=None):
    entry = {'name': name, 'repo': f'offline/{name}', 'decoder': 'scalars', 'scalar_columns': {key: 'value'}}
    if weight is not None:
        entry['draw_weight'] = weight
    return entry


def test_dual_ownership_and_realized_cycle(monkeypatch):
    spec = {'anchors': [survey('north', 'ebv', 1), survey('south', 'south_value', 2)],
            'catalogs': [survey('partner', 'Z')], 'min_catalogs': 2, 'partitions_per_chunk': 64}
    spec['catalogs'][0]['scalar_columns']['ZWARN'] = 'flag'
    config = AstroPT3Config(modalities=[m for m in DEFAULT_MODALITIES if m['name'] in ('ebv', 'Z')])
    # The near pair shares a coarse sky cell but is separated by >1 arcsec:
    # an outer-only partner remains a singleton; the other partner attaches.
    locations = {'north': [10., 10.001, 70.], 'south': [10., 110.], 'partner': [10., 10.002, 110., 110.002]}
    tables = {}
    for name, ra in locations.items():
        rows = pd.DataFrame({'ra': ra, 'dec': 0., 'value': 0.25, 'flag': 0,
                             'object_id': [str(i) for i in range(len(ra))]})
        tables['hf://datasets/offline/' + name] = from_dataframe(rows, catalog_name=name)
    monkeypatch.setattr(stream, 'open_catalog', lambda url, **kw: tables[url])
    monkeypatch.setattr(stream, 'LocalCluster', lambda **kw: nullcontext())
    monkeypatch.setattr(stream, 'Client', lambda c: nullcontext())
    catalog = stream.CatalogStream(spec, config)
    with closing(catalog.draws(seed=7, rank=0, worker_id=0)) as draws:
        frames = [list(rs) for _, rs in islice(draws, 3)]
    paths = [frame[0]['anchor_path'] for frame in frames]
    assert sorted(paths) == ['north', 'south', 'south']
    north, south = ({r['object_id']: r for r in frames[paths.index(path)]} for path in ('north', 'south'))
    assert set(north) == {'north:0', 'north:1', 'partner:1'}
    assert set(south) == {'south:1', 'partner:3'}
    assert north['north:0']['source_ids'] == {'north': '0', 'partner': '0'}
    assert north['north:1']['source_ids'] == {'north': '1'}  # min_catalogs is sky, not rows
    assert south['south:1']['source_ids'] == {'south': '1', 'partner': '2'}
    assert north['partner:1']['anchor'] is south['partner:3']['anchor'] is None


def test_holdout_is_disjoint_and_buffered():
    train, validation = (stream._split_sky(name) for name in ('train', 'validation'))
    assert train.intersection(validation.to_order(10).extended()).empty()
    # Neighbouring centers / duplicate cutouts never straddle train and validation.
    ras = u.Quantity(np.arange(360.), u.deg)
    decs = u.Quantity(np.zeros(360), u.deg)
    held = validation.contains_lonlat(ras, decs)
    assert 0 < held.sum() < len(held)
    assert not train.contains_lonlat(ras[held] + 1*u.arcsec, decs[held]).any()


@pytest.mark.parametrize('weight', [0, -1, True, 1.5, None])
def test_draw_weights_are_explicit_and_positive(weight):
    spec = {'anchors': [survey('north', 'ebv', weight)], 'catalogs': []}
    with pytest.raises(ValueError, match='draw_weight'):
        stream.CatalogStream(spec, AstroPT3Config())
