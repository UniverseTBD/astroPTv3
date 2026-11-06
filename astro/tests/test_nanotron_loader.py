"""CPU tests for the nanotron LSDB micro-batch adapter.

Contract honored by the nanotron fork's ``AstroPT3ForTraining.forward``;
sources are faked — only the network-marked live check touches the hub.
"""

from itertools import islice

import numpy as np
import pytest
import torch
from legacy_fixture import (
    crossmatch_row,
    legacy_only_crossmatch_row,
    legacy_row,
    make_record,
    nested_frame,
)

from astropt3.data import nanotron_loader
from astropt3.data.nanotron_loader import (
    PackedMicroBatches,
    consumer_seed,
    decode_crossmatch_row,
    decode_legacy_row,
)
from astropt3.data.nanotron_loader import (
    regroup_micro_batch as regroup,
)
from astropt3.tokenization import BOS_ID, modality_token_ids

MBS = 2
SEQ_LEN = 896


class _FakeCatalog:
    npartitions = 3


def fake_stream(frames_per_epoch=2, rows_per_frame=4):
    """Cheap ``InfiniteStream`` replacement yielding unconsumed DataFrames."""

    class FakeStream:
        seed = None

        def __init__(self, catalog, client, partitions_per_chunk, seed):
            FakeStream.seed = seed

        def __iter__(self):
            for _ in range(frames_per_epoch):
                yield nested_frame(
                    [legacy_row(i) for i in range(rows_per_frame)], ("image",)
                )

    return FakeStream


@pytest.fixture
def fake_lsdb(monkeypatch):
    monkeypatch.setattr(
        nanotron_loader, "open_catalog", lambda *a, **k: _FakeCatalog()
    )
    monkeypatch.setattr(nanotron_loader, "_log_provenance", lambda *a, **k: None)

    def install(stream_cls):
        monkeypatch.setattr(nanotron_loader, "InfiniteStream", stream_cls)

    return install


def _stream(tiny_config, fake_lsdb, **kwargs):
    fake_lsdb(fake_stream())
    return PackedMicroBatches(tiny_config, MBS, SEQ_LEN, **kwargs)


def test_micro_batch_contract(tiny_config, fake_lsdb):
    registry = tiny_config.modality_registry()
    stream = _stream(tiny_config, fake_lsdb)
    for flat in islice(iter(stream), 3):
        assert flat["input_ids"].shape == (MBS, SEQ_LEN)
        assert flat["position_ids"].shape == (MBS, SEQ_LEN)
        assert flat["input_ids"].dtype == torch.long
        for name in registry.names():
            mod = registry.get_config(name)
            mask = flat[f"{name}_mask"]
            values = flat[f"{name}_values"]
            positions = flat[f"{name}_positions"]
            assert mask.shape == (MBS, SEQ_LEN) and mask.dtype == torch.bool
            assert values.shape == (int(mask.sum()), mod.input_size)
            assert values.dtype == torch.float32
            assert len(positions) == len(values)
            _, placeholder_id, _ = modality_token_ids(name)
            assert (flat["input_ids"][mask] == placeholder_id).all()
            assert not mask[:, 0].any()
        assert (flat["position_ids"][:, 0] == 0).all()
        assert (flat["input_ids"][:, 0] == BOS_ID).all()


def test_batches_feed_hf_model(tiny_config, tiny_model, fake_lsdb):
    stream = _stream(tiny_config, fake_lsdb)
    names = tiny_config.modality_registry().names()
    for flat in islice(iter(stream), 3):
        out = tiny_model(**regroup(flat, names))
        assert torch.isfinite(out.loss)


def test_image_only_records_ship_empty_spectra_tensors(
    tiny_config, tiny_model, fake_lsdb
):
    stream = _stream(tiny_config, fake_lsdb)
    flat = next(iter(stream))
    # uncrossmatched LegacySurvey rows carry image + legacy scalars only
    assert not flat["spectra_mask"].any()
    assert flat["spectra_values"].shape == (0, 256)
    assert flat["spectra_values"].dtype == torch.float32
    assert flat["spectra_positions"].shape == (0, 1)
    assert flat["Z_values"].shape == (0, 1)
    assert flat["Z_positions"].dtype == torch.long
    out = tiny_model(**regroup(flat, tiny_config.modality_registry().names()))
    assert torch.isfinite(out.loss)
    assert set(out.modality_losses) == {"images", "ebv", "photometry"}


def test_seeds_differ_across_rank_worker_and_retry(tiny_config):
    base = consumer_seed(42, 0, 0, 0)
    assert base != consumer_seed(42, 1, 0, 0)
    assert base != consumer_seed(42, 0, 1, 0)
    assert base != consumer_seed(42, 0, 0, 1)
    # two consumers with identical identity derive the same seed
    assert base == consumer_seed(42, 0, 0, 0)


def test_decoder_recovers_record_fields():
    record = decode_legacy_row(legacy_row(3))
    assert record["image"]["flux"].shape == (3, 152, 152)
    assert record["image"]["flux"].dtype == np.float32
    assert record["image"]["band"] == ["des-g", "des-r", "des-z"]
    for key in (
        "ebv",
        "flux_g",
        "flux_r",
        "flux_z",
        "fiberflux_g",
        "psfdepth_z",
        "z_spec",
    ):
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


def test_crossmatch_decoder_recovers_matched_row():
    record = decode_crossmatch_row(crossmatch_row(3, matched=True))
    assert record["spectrum"]["flux"].shape == (7781,)
    assert record["spectrum"]["flux"].dtype == np.float32
    assert record["spectrum"]["mask"].dtype == bool
    assert record["image"]["flux"].shape == (3, 152, 152)
    assert record["image"]["band"] == ["des-g", "des-r", "des-z"]
    assert isinstance(record["Z"], float)
    assert isinstance(record["ebv"], float)
    for band in ("des-g", "des-r", "des-z"):
        assert isinstance(record[f"psf_fwhm_{band}"], float)


def test_crossmatch_decoder_handles_unmatched_row():
    record = decode_crossmatch_row(crossmatch_row(3, matched=False))
    assert "spectrum" in record
    assert "image" not in record
    assert "ebv" not in record


def test_crossmatch_decoder_falls_back_to_legacy_id_when_desi_id_missing():
    """OuterKdTreeCrossmatch rows can be Legacy-only (no DESI match at all,
    so "object_id" itself is null) -- object_id falls back to
    "object_id_legacy" rather than raising."""
    row = crossmatch_row(0, matched=True)
    del row["object_id"]
    record = decode_crossmatch_row(row)
    assert record["object_id"] == row["object_id_legacy"]


def test_crossmatch_decoder_rejects_missing_id():
    row = crossmatch_row(0, matched=True)
    del row["object_id"]
    del row["object_id_legacy"]
    with pytest.raises(ValueError, match="object_id"):
        decode_crossmatch_row(row)


def test_crossmatch_decoder_handles_legacy_only_row(tiny_config):
    """OuterKdTreeCrossmatch rows recovered from an unmatched right side:
    image present, no spectrum, object_id from the Legacy side."""
    row = legacy_only_crossmatch_row(7)
    record = decode_crossmatch_row(row)
    assert record["object_id"] == row["object_id_legacy"]
    assert "spectrum" not in record
    assert record["image"]["flux"].shape == (3, 152, 152)

    from astropt3.data.packing import ObjectSequencer

    obj = ObjectSequencer(tiny_config).build(record)
    assert obj.input_ids[0] == 1


class _AmbiguousBoolMapping:
    """Mimics a real nested-pandas struct scalar: like a live ``image_legacy``
    cell, ``bool(...)`` raises pandas' own ambiguity error rather than
    falling back to truthiness."""

    def __init__(self, mapping):
        self._mapping = mapping

    def __bool__(self):
        raise ValueError(
            "The truth value of a DataFrame is ambiguous. "
            "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
        )

    def as_py(self):
        return self._mapping


def test_crossmatch_decoder_never_bool_checks_nested_image_scalar():
    """Regression: a live run crashed because decode did ``image_value and
    ...`` on a nested-pandas struct scalar, whose ``bool()`` raises instead
    of returning True/False like a plain dict would."""
    row = crossmatch_row(3, matched=True)
    row["image_legacy"] = _AmbiguousBoolMapping(row["image_legacy"])
    record = decode_crossmatch_row(row)
    assert record["image"]["flux"].shape == (3, 152, 152)


def test_transient_error_reopens_fresh_stream(tiny_config, monkeypatch):
    """A mid-iteration transport blip discards the iterator and retries."""
    builds = {"n": 0}

    class FlakyStream:
        def __init__(self, *a, **k):
            builds["n"] += 1
            self.built = builds["n"]

        def __iter__(self):
            if self.built == 1:
                yield nested_frame([legacy_row(i) for i in range(6)], ("image",))
                raise OSError("simulated storage blip")
            while True:
                yield nested_frame([legacy_row(i) for i in range(6)], ("image",))

    monkeypatch.setattr(
        nanotron_loader, "open_catalog", lambda *a, **k: _FakeCatalog()
    )
    monkeypatch.setattr(nanotron_loader, "_log_provenance", lambda *a, **k: None)
    monkeypatch.setattr(nanotron_loader, "InfiniteStream", FlakyStream)
    monkeypatch.setattr(nanotron_loader.time, "sleep", lambda *_: None)

    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN)
    batches = list(islice(iter(stream), 3))
    assert builds["n"] >= 2, "no fresh stream was opened after the blip"
    assert len(batches) == 3
    for flat in batches:
        assert flat["input_ids"].shape == (MBS, SEQ_LEN)


def test_swallowed_hub_error_reopens_fresh_stream(tiny_config, monkeypatch):
    """Regression: fsspec's parquet reader stores cat_ranges() results --
    gathered with return_exceptions=True -- as KnownPartsOfAFile block data
    without exception checks, so a transient HfHubHTTPError surfaces as
    ``TypeError: can't concat HfHubHTTPError to bytes`` (killed the v3 run
    at step 46,767, 2026-09-04). It must retry like any transport blip."""
    builds = {"n": 0}

    class SwallowedErrorStream:
        def __init__(self, *a, **k):
            builds["n"] += 1
            self.built = builds["n"]

        def __iter__(self):
            if self.built == 1:
                yield nested_frame([legacy_row(i) for i in range(6)], ("image",))
                raise TypeError("can't concat HfHubHTTPError to bytes")
            while True:
                yield nested_frame([legacy_row(i) for i in range(6)], ("image",))

    monkeypatch.setattr(
        nanotron_loader, "open_catalog", lambda *a, **k: _FakeCatalog()
    )
    monkeypatch.setattr(nanotron_loader, "_log_provenance", lambda *a, **k: None)
    monkeypatch.setattr(nanotron_loader, "InfiniteStream", SwallowedErrorStream)
    monkeypatch.setattr(nanotron_loader.time, "sleep", lambda *_: None)

    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN)
    batches = list(islice(iter(stream), 3))
    assert builds["n"] >= 2, "no fresh stream was opened after the swallow"
    assert len(batches) == 3


def test_swallowed_hub_error_signature_matches_real_fsspec_message():
    """Canary: the TypeErrors _retryable matches are the ones fsspec's
    KnownPartsOfAFile actually raises when a cat_ranges() result is an
    exception object -- if fsspec rewords them, fail here instead of
    silently un-matching real hub blips at 3am. Two shapes seen in the
    wild: block merge (concat) and footer slicing (subscript)."""
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
            data={(0, 4): b"PAR1", (4, 8): error},  # type: ignore[arg-type]  # the bug: an exception stored as block data
        )
    assert nanotron_loader._retryable(raised.value)
    # footer-slicing shape (killed the bench arm at step 302, 2026-09-24)
    assert nanotron_loader._retryable(
        TypeError("'HfHubHTTPError' object is not subscriptable")
    )
    # ordinary TypeErrors stay fatal -- including subscript errors on
    # non-exception types
    assert not nanotron_loader._retryable(TypeError("ordinary"))
    assert not nanotron_loader._retryable(
        TypeError("'str' object is not subscriptable")
    )


def test_non_retryable_error_fails_immediately(tiny_config, monkeypatch):
    """Decode/validation errors are not retried."""

    class BadStream:
        def __init__(self, *a, **k):
            pass

        def __iter__(self):
            yield nested_frame([legacy_row(0)], ("image",))
            raise ValueError("decode blew up")

    monkeypatch.setattr(
        nanotron_loader, "open_catalog", lambda *a, **k: _FakeCatalog()
    )
    monkeypatch.setattr(nanotron_loader, "_log_provenance", lambda *a, **k: None)
    monkeypatch.setattr(nanotron_loader, "InfiniteStream", BadStream)
    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN)
    with pytest.raises(ValueError, match="decode blew up"):
        next(iter(stream))


def test_packing_keeps_each_record_once_and_preserves_base_shuffle(tiny_config, monkeypatch):
    record = make_record(1, image_only_fraction=0.0)
    records = [{**record, "object_id": str(i)} for i in range(5)]
    sequencer = nanotron_loader.ObjectSequencer(tiny_config)
    objects = [sequencer.build(item, epoch=7) for item in records]
    stream = PackedMicroBatches(tiny_config, 2, len(objects[0]))
    stream._epoch = 7
    monkeypatch.setattr(stream, "_open_records", lambda worker: iter(records))
    batches = list(stream)
    assert len(batches) == 2
    for index, batch in enumerate(batches):
        expected = stream.collator.collate_rows(
            [[objects[2 * index]], [objects[2 * index + 1]]]
        )
        expected = nanotron_loader.flatten_packed_batch(expected, tiny_config, stream.seq_len)
        for key in batch:
            assert torch.equal(batch[key], expected[key])
    assert len({tuple(obj.input_ids.tolist()) for obj in objects}) > 1


def test_unequal_objects_keep_emptiest_first_placement_across_flushes(tiny_config, monkeypatch):
    full = make_record(1, image_only_fraction=0.0)
    kinds = ["image", "spectrum", "image", "both", "spectrum", "both",
             "image", "spectrum", "both", "image", "spectrum"]
    records = [
        {"object_id": str(i), **{key: full[key] for key in ("image", "spectrum")
                                if kind == "both" or key == kind}}
        for i, kind in enumerate(kinds)
    ]
    stream = PackedMicroBatches(tiny_config, 2, 360)
    stream._epoch = 7
    objects = [stream.sequencer.build(record, epoch=7) for record in records]
    assert [len(obj) for obj in objects[:4]] == [147, 34, 147, 180]
    monkeypatch.setattr(stream, "_open_records", lambda worker: iter(records))
    # Frozen factor-1 placements from the removed first-fitting/emptiest-row path.
    expected_rows = [[[0, 3], [1, 2, 4]], [[5, 8], [6, 7, 9]]]
    batches = list(stream)
    assert len(batches) == len(expected_rows)
    for batch, rows in zip(batches, expected_rows, strict=True):
        packed = stream.collator.collate_rows([[objects[i] for i in row] for row in rows])
        expected = nanotron_loader.flatten_packed_batch(packed, tiny_config, 360)
        for key in batch:
            assert torch.equal(batch[key], expected[key])


@pytest.mark.parametrize("obsolete", ["ar_replicas", "replica_placement"])
def test_removed_replica_options_are_rejected(tiny_config, obsolete):
    with pytest.raises(TypeError, match=obsolete):
        PackedMicroBatches(tiny_config, MBS, SEQ_LEN, **{obsolete: 2})  # pyright: ignore[reportArgumentType] -- deliberately invalid kwargs
