"""Two real in-memory LSDB stars, no Hub or distributed worker needed."""

from contextlib import closing, nullcontext
from itertools import islice, repeat
from types import SimpleNamespace

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


def test_sky_shards_partition_the_unsharded_dual_anchor_rows(monkeypatch):
    # The North guard must come from North's whole footprint, not its shard; otherwise
    # South shards re-claim sky other consumers' North shards own and rows train twice.
    rng = np.random.default_rng(0)
    tables = {}
    for name, ra_max in (('north', 90.), ('south', 360.)):
        rows = pd.DataFrame({'ra': rng.uniform(0, ra_max, 2000), 'dec': rng.uniform(-20, 20, 2000),
                             'value': 0.25, 'object_id': [f'{name}:{i}' for i in range(2000)]})
        tables[name] = from_dataframe(rows, catalog_name=name, lowest_order=2, partition_rows=100)
    assert min(t.npartitions for t in tables.values()) > 4
    monkeypatch.setattr(stream, 'InfiniteStream', lambda cat, **kw: SimpleNamespace(npartitions=1, cat=cat))
    spec = {'anchors': [survey('north', 'ebv', 1), survey('south', 'south_value', 1)], 'catalogs': []}
    catalog = stream.CatalogStream(spec, AstroPT3Config(modalities=[m for m in DEFAULT_MODALITIES if m['name'] == 'ebv']))

    def owned(shard):
        return [ids for s in catalog._streams(tables, None, 0, shard) for ids in s.cat.compute()['object_id']]

    whole = owned((0, 1))
    shards = [ids for i in range(4) for ids in owned((i, 4))]
    assert len(shards) == len(set(shards)) and set(shards) == set(whole)


def test_healthy_anchor_does_not_reset_south_retries(monkeypatch):
    spec = {'anchors': [survey('north', 'ebv', 1), survey('south', 'south_value', 2)], 'catalogs': []}
    config = AstroPT3Config(modalities=[m for m in DEFAULT_MODALITIES if m['name'] == 'ebv'])
    catalog = stream.CatalogStream(spec, config)
    waits = []

    class South:
        npartitions = 1

        def __iter__(self):
            raise OSError('South unavailable')

    class North:
        npartitions = 1

        def __iter__(self):
            return repeat(SimpleNamespace(empty=True))

    def plan(*args):
        assert len(waits) <= 2, 'retry budget did not stop the reader'
        return [North(), South()]

    monkeypatch.setattr(stream, 'open_catalog', lambda *_args, **_kwargs: object())
    monkeypatch.setattr(stream, 'LocalCluster', lambda **kw: nullcontext())
    monkeypatch.setattr(stream, 'Client', lambda c: nullcontext())
    monkeypatch.setattr(stream.loader, 'consumer_seed', lambda *_args: 0)
    monkeypatch.setattr(stream.loader, '_MAX_NET_RETRIES', 2)
    monkeypatch.setattr(catalog, '_streams', plan)
    monkeypatch.setattr(stream.time, 'sleep', waits.append)
    with closing(catalog.draws(seed=0, rank=0, worker_id=0)) as draws, pytest.raises(OSError, match='South unavailable'):
        next(draws)
    assert waits == [5, 10]


def test_explicit_anchor_required():
    with pytest.raises(ValueError, match='anchors'):
        stream.CatalogStream({'catalogs': [survey('north', 'ebv')]}, AstroPT3Config())


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
