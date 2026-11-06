"""CPU tests for the configured LSDB -> nanotron micro-batch adapter.

Only external LSDB/Dask I/O is faked; shared decoding and packing stay real.
"""

from contextlib import closing, nullcontext
from itertools import islice, repeat
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from legacy_fixture import TINY_CATALOG_STREAM, legacy_row, make_record, nested_frame

from astropt3.data import catalog_stream, nanotron_loader
from astropt3.data.nanotron_loader import (
    PackedMicroBatches,
    build_astropt3_dataloader,
    consumer_seed,
    decode_legacy_row,
)
from astropt3.data.nanotron_loader import (
    regroup_micro_batch as regroup,
)
from astropt3.tokenization import BOS_ID, modality_token_ids

MBS = 2
SEQ_LEN = 896


def _frame(image_only=False):
    rows = []
    for i in range(6):
        record = make_record(i, image_only_fraction=0.0)
        rows.append({
            **legacy_row(i),
            "object_id_desi": f"desi-{i}",
            "spectrum_desi": record["spectrum"],
            "Z_desi": record["Z"],
            "ZWARN_desi": record["ZWARN"],
        })
    frame = nested_frame(rows, ("image", "spectrum_desi"))
    if image_only:
        # Preserve the Arrow nested dtype for an entirely unmatched right side.
        for column in ("object_id_desi", "spectrum_desi", "Z_desi", "ZWARN_desi"):
            frame[column] = frame[column].mask(np.ones(len(frame), dtype=bool))
    return frame


@pytest.fixture(autouse=True)
def fake_lsdb(monkeypatch):
    state = SimpleNamespace(frames=lambda: repeat(_frame()), streams=[], sleeps=[])

    class FakeStream:
        npartitions = 3

        def __init__(self, anchor, *rights, min_surveys=None, **kwargs):
            self.seed = kwargs["seed"]
            state.streams.append(self)

        def __iter__(self):
            return iter(state.frames())

    monkeypatch.setattr(catalog_stream, "open_catalog", lambda *a, **k: object())
    monkeypatch.setattr(catalog_stream, "LocalCluster", lambda **k: nullcontext())
    monkeypatch.setattr(catalog_stream, "Client", lambda *a: nullcontext())
    monkeypatch.setattr(catalog_stream, "CrossMatchStream", FakeStream)
    monkeypatch.setattr(catalog_stream.time, "sleep", state.sleeps.append)
    monkeypatch.delenv("ASTROPT3_TELEMETRY_DIR", raising=False)
    return state


def _stream(tiny_config, **kwargs):
    return PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, catalog_stream=TINY_CATALOG_STREAM, **kwargs
    )


def test_micro_batch_contract(tiny_config):
    registry = tiny_config.modality_registry()
    stream = _stream(tiny_config)
    assert isinstance(stream.catalog_stream, catalog_stream.CatalogStream)
    with closing(iter(stream)) as batches:
        for flat in islice(batches, 3):
            assert flat["input_ids"].shape == (MBS, SEQ_LEN)
            assert flat["position_ids"].shape == (MBS, SEQ_LEN)
            assert flat["input_ids"].dtype == torch.long
            for name in registry.names():
                mod = registry.get_config(name)
                mask = flat[f"{name}_mask"]
                values = flat[f"{name}_values"]
                positions = flat[f"{name}_positions"]
                assert mask.shape == (MBS, SEQ_LEN) and mask.dtype == torch.bool
                assert mask.any(), name
                assert values.shape == (int(mask.sum()), mod.input_size)
                assert values.dtype == torch.float32
                assert torch.isfinite(values).all()
                assert len(positions) == len(values)
                _, placeholder_id, _ = modality_token_ids(name)
                assert torch.equal(mask, flat["input_ids"] == placeholder_id)
                assert not mask[:, 0].any()
            assert (flat["position_ids"][:, 0] == 0).all()
            assert (flat["input_ids"][:, 0] == BOS_ID).all()


def test_batches_feed_hf_model(tiny_config, tiny_model):
    stream = _stream(tiny_config)
    names = tiny_config.modality_registry().names()
    with closing(iter(stream)) as batches, torch.no_grad():
        for flat in islice(batches, 3):
            out = tiny_model(**regroup(flat, names))
            assert torch.isfinite(out.loss)
            assert set(out.modality_losses) == set(names)


def test_image_only_records_ship_empty_spectra_tensors(tiny_config, tiny_model, fake_lsdb):
    fake_lsdb.frames = lambda: repeat(_frame(image_only=True))
    with closing(iter(_stream(tiny_config))) as batches:
        flat = next(batches)
    assert not flat["spectra_mask"].any()
    assert flat["spectra_values"].shape == (0, 256)
    assert flat["spectra_values"].dtype == torch.float32
    assert flat["spectra_positions"].shape == (0, 1)
    assert flat["Z_values"].shape == (0, 1)
    assert flat["Z_positions"].dtype == torch.long
    with torch.no_grad():
        out = tiny_model(**regroup(flat, tiny_config.modality_registry().names()))
    assert torch.isfinite(out.loss)
    assert set(out.modality_losses) == {"images", "ebv", "photometry"}


def test_open_draws_delegates_identity_and_epoch(tiny_config, fake_lsdb):
    stream = _stream(tiny_config, rank=2, seed=19)
    with closing(stream._open_draws(SimpleNamespace(id=3))) as draws:
        for epoch in range(2):
            records = list(next(draws))
            assert len(records) == 6
            assert records[0]["object_id"] == "synth_00000000"
            assert stream._epoch == epoch
    assert len(fake_lsdb.streams) == 1
    assert fake_lsdb.streams[0].seed == consumer_seed(19, 2, 3, 0)


def test_seeds_differ_across_rank_worker_and_retry():
    base = consumer_seed(42, 0, 0, 0)
    assert base != consumer_seed(42, 1, 0, 0)
    assert base != consumer_seed(42, 0, 1, 0)
    assert base != consumer_seed(42, 0, 0, 1)
    assert base == consumer_seed(42, 0, 0, 0)


def test_decoder_recovers_record_fields():
    record = decode_legacy_row(legacy_row(3))
    assert record["image"]["flux"].shape == (3, 152, 152)
    assert record["image"]["flux"].dtype == np.float32
    assert record["image"]["band"] == ["des-g", "des-r", "des-z"]
    for key in ("ebv", "flux_g", "flux_r", "flux_z", "fiberflux_g", "psfdepth_z", "z_spec"):
        assert isinstance(record[key], float), key
    for band in ("des-g", "des-r", "des-z"):
        assert isinstance(record[f"psf_fwhm_{band}"], float)


def test_decoder_rejects_bad_shape_and_id():
    row = legacy_row(0)
    row["image"] = {"band": ["des-g"], "flux": [[0.0]]}
    with pytest.raises(ValueError, match="flux has shape"):
        decode_legacy_row(row)
    row = legacy_row(0)
    del row["object_id"]
    with pytest.raises(ValueError, match="object_id"):
        decode_legacy_row(row)


@pytest.mark.parametrize("dataframe", [False, True])
def test_spectrum_decoder_preserves_calibrated_payload(dataframe):
    spectrum = make_record(3, image_only_fraction=0.0)["spectrum"]
    payload = pd.DataFrame(spectrum) if dataframe else spectrum
    decoded = nanotron_loader._decode_spectrum(payload)
    assert set(decoded) == {"flux", "lambda", "mask"}
    for key, values in decoded.items():
        np.testing.assert_array_equal(values, spectrum[key])
        assert values.shape == (7781,)
        assert values.dtype == (bool if key == "mask" else np.float32)


class _AmbiguousBoolMapping:
    """A nested-pandas struct scalar must not be tested for truthiness."""

    def __init__(self, mapping):
        self._mapping = mapping

    def __bool__(self):
        raise ValueError("The truth value of a DataFrame is ambiguous.")

    def as_py(self):
        return self._mapping


def test_decoder_never_bool_checks_nested_image_scalar():
    row = legacy_row(3)
    row["image"] = _AmbiguousBoolMapping(row["image"])
    record = decode_legacy_row(row)
    assert record["image"]["flux"].shape == (3, 152, 152)


@pytest.mark.parametrize("error", [
    OSError("simulated storage blip"),
    TypeError("can't concat HfHubHTTPError to bytes"),
    TypeError("'HfHubHTTPError' object is not subscriptable"),
])
def test_transient_error_reopens_configured_stream(tiny_config, fake_lsdb, error):
    frame = _frame()

    def frames():
        yield frame
        if len(fake_lsdb.streams) == 1:
            raise error
        yield from repeat(frame)

    fake_lsdb.frames = frames
    with closing(iter(_stream(tiny_config))) as iterator:
        batches = list(islice(iterator, 3))
    assert len(fake_lsdb.streams) == 2
    assert fake_lsdb.streams[0].seed != fake_lsdb.streams[1].seed
    assert fake_lsdb.sleeps == [5]
    assert len(batches) == 3
    assert all(batch["input_ids"].shape == (MBS, SEQ_LEN) for batch in batches)


def test_swallowed_hub_error_signature_matches_real_fsspec_message():
    """Canary for exception objects swallowed into fsspec parquet block data."""
    import httpx
    from fsspec.caching import KnownPartsOfAFile
    from huggingface_hub.errors import HfHubHTTPError

    error = HfHubHTTPError(
        "transient 503",
        response=httpx.Response(
            503, request=httpx.Request("GET", "https://hf.co/f.parquet")
        ),
    )
    with pytest.raises(TypeError) as raised:
        KnownPartsOfAFile(
            blocksize=4,
            fetcher=lambda a, b: b"",
            size=8,
            data={(0, 4): b"PAR1", (4, 8): error},  # type: ignore[arg-type] -- regression: exception stored as block data
        )
    assert nanotron_loader._retryable(raised.value)
    assert nanotron_loader._retryable(
        TypeError("'HfHubHTTPError' object is not subscriptable")
    )
    assert not nanotron_loader._retryable(TypeError("ordinary"))
    assert not nanotron_loader._retryable(TypeError("'str' object is not subscriptable"))


def test_non_retryable_error_fails_immediately(tiny_config, fake_lsdb):
    def frames():
        yield _frame()
        raise ValueError("decode blew up")

    fake_lsdb.frames = frames
    with pytest.raises(ValueError, match="decode blew up"):
        next(iter(_stream(tiny_config)))
    assert len(fake_lsdb.streams) == 1
    assert not fake_lsdb.sleeps


def test_packing_keeps_each_record_once_and_preserves_base_shuffle(tiny_config):
    record = make_record(1, image_only_fraction=0.0)
    records = [{**record, "object_id": str(i)} for i in range(5)]
    sequencer = nanotron_loader.ObjectSequencer(tiny_config)
    objects = [sequencer.build(item, epoch=7) for item in records]
    stream = PackedMicroBatches(
        tiny_config, 2, len(objects[0]), catalog_stream=TINY_CATALOG_STREAM
    )
    stream._epoch = 7
    batches = list(stream._pack_records(iter(records)))
    assert len(batches) == 2
    for index, batch in enumerate(batches):
        expected = stream.collator.collate_rows(
            [[objects[2 * index]], [objects[2 * index + 1]]]
        )
        expected = nanotron_loader.flatten_packed_batch(expected, tiny_config, stream.seq_len)
        for key in batch:
            assert torch.equal(batch[key], expected[key])
    assert len({tuple(obj.input_ids.tolist()) for obj in objects}) > 1


def test_unequal_objects_keep_emptiest_first_placement_across_flushes(tiny_config):
    full = make_record(1, image_only_fraction=0.0)
    kinds = ["image", "spectrum", "image", "both", "spectrum", "both",
             "image", "spectrum", "both", "image", "spectrum"]
    records = [
        {"object_id": str(i), **{key: full[key] for key in ("image", "spectrum")
                                if kind == "both" or key == kind}}
        for i, kind in enumerate(kinds)
    ]
    stream = PackedMicroBatches(tiny_config, 2, 360, catalog_stream=TINY_CATALOG_STREAM)
    objects = [stream.sequencer.build(record, epoch=7) for record in records]
    assert [len(obj) for obj in objects[:4]] == [147, 34, 147, 180]
    stream._epoch = 7
    # Frozen factor-1 placements from the removed first-fitting/emptiest-row path.
    expected_rows = [[[0, 3], [1, 2, 4]], [[5, 8], [6, 7, 9]]]
    batches = list(stream._pack_records(iter(records)))
    assert len(batches) == len(expected_rows)
    for batch, rows in zip(batches, expected_rows, strict=True):
        packed = stream.collator.collate_rows([[objects[i] for i in row] for row in rows])
        expected = nanotron_loader.flatten_packed_batch(packed, tiny_config, 360)
        for key in batch:
            assert torch.equal(batch[key], expected[key])


def test_catalog_stream_argument_is_required(tiny_config):
    with pytest.raises(TypeError, match="catalog_stream"):
        PackedMicroBatches(tiny_config, MBS, SEQ_LEN)  # pyright: ignore[reportCallIssue] -- deliberately missing required spec


@pytest.mark.parametrize("spec", [None, {}, {"catalogs": []}, {"catalogs": "legacy"}])
def test_missing_or_invalid_catalogs_are_rejected(tiny_config, spec):
    with pytest.raises(ValueError, match="catalog_stream"):
        PackedMicroBatches(tiny_config, MBS, SEQ_LEN, catalog_stream=spec)


def test_builder_rejects_missing_catalogs_and_old_selection(tiny_config):
    for args in (SimpleNamespace(), SimpleNamespace(crossmatch_desi=True)):
        with pytest.raises(ValueError, match="catalog_stream"):
            build_astropt3_dataloader(args, tiny_config, MBS, SEQ_LEN, 0, 1)


@pytest.mark.parametrize("obsolete", [
    {"ar_replicas": 2}, {"replica_placement": 2},
    {"crossmatch_desi": False}, {"crossmatch_desi": True},
])
def test_removed_selection_and_replica_options_are_rejected(tiny_config, obsolete):
    with pytest.raises(TypeError, match=next(iter(obsolete))):
        PackedMicroBatches(
            tiny_config, MBS, SEQ_LEN, catalog_stream=TINY_CATALOG_STREAM, **obsolete
        )  # pyright: ignore[reportArgumentType] -- deliberately invalid kwargs
