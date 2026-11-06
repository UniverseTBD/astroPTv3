"""Bounded live checks of the shared configured CatalogStream.

Requires hub access; exclude with ``pytest -m 'not network'`` for offline runs.
"""

from contextlib import closing
from copy import deepcopy
from itertools import islice

import pytest
import torch
from legacy_fixture import TINY_CATALOG_STREAM

from astropt3.configuration_astropt3 import AstroPT3Config
from astropt3.data.catalog_stream import CatalogStream
from astropt3.data.packing import ObjectSequencer

pytestmark = pytest.mark.network

ROWS_TO_CHECK = 5


@pytest.mark.parametrize("catalog_count", [1, 2])
def test_live_catalog_stream_decodes_and_selects(tiny_config, catalog_count):
    spec = deepcopy(TINY_CATALOG_STREAM)
    spec["catalogs"] = spec["catalogs"][:catalog_count]
    spec["partitions_per_chunk"] = 1
    modalities = tiny_config.modalities
    if catalog_count == 1:
        modalities = [m for m in modalities if m["source"] == "legacy"]
    config = AstroPT3Config(modalities=modalities)
    stream = CatalogStream(spec, config)
    sequencer = ObjectSequencer(config)

    with closing(stream.draws(seed=0, rank=0, worker_id=0)) as draws:
        _, records = next(draws)
        checked = 0
        for record in islice(records, ROWS_TO_CHECK):
            assert "image" in record or "spectrum" in record
            if "image" in record:
                flux = torch.as_tensor(record["image"]["flux"])
                assert flux.shape == (3, 152, 152)
                assert torch.isfinite(flux).all()
                assert record["image"]["band"] == ["des-g", "des-r", "des-z"]
            if "spectrum" in record:
                assert record["spectrum"]["flux"].shape == (7781,)
            obj = sequencer.build(record)
            assert obj.input_ids[0] == 1
            assert all(torch.isfinite(values).all() for values in obj.values.values())
            checked += 1
        assert checked == ROWS_TO_CHECK
        # No fixed matched/image-only mix is guaranteed for a small sky draw.
        for _ in range(2):
            _, more = next(draws)
            assert next(more, None) is not None
