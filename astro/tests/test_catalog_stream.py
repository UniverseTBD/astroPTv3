"""Offline contracts for configured catalogs, independent of the training runner."""

from contextlib import closing, nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from legacy_fixture import make_record, nested_frame

from astropt3.configuration_astropt3 import DEFAULT_MODALITIES, AstroPT3Config
from astropt3.data import catalog_stream as streaming
from astropt3.data import nanotron_loader as loader
from astropt3.data.packing import ObjectSequencer


@pytest.mark.parametrize("count", [1, 4])
def test_real_lsdb_stream_decodes_local_catalogs(recipe, monkeypatch, count):
    from lsdb.loaders.dataframe.from_dataframe import from_dataframe
    from lsdb.streams.catalog_streams import CrossMatchStream, InfiniteStream

    spec, config = _case(recipe, count)
    spec["partitions_per_chunk"] = 1
    catalogs = {}
    for catalog in spec["catalogs"]:
        payload = "image" if catalog["decoder"].endswith("image") else "spectrum"
        row = {"ra": 10.0, "dec": 0.0, **_row({"catalogs": [catalog]})}
        catalogs[f"hf://datasets/{catalog['repo']}"] = from_dataframe(
            nested_frame([row], (payload,)), catalog_name=catalog["name"],
        )
    monkeypatch.setattr(streaming, "open_catalog", lambda url, **kwargs: catalogs[url])
    monkeypatch.setattr(streaming, "Client", lambda cluster: nullcontext())
    monkeypatch.setattr(streaming, "CrossMatchStream", CrossMatchStream)
    monkeypatch.setattr(streaming, "InfiniteStream", InfiniteStream)
    stream = streaming.CatalogStream(spec, config)
    with closing(stream.draws(seed=0, rank=0, worker_id=0)) as draws:
        _, records = next(draws)
        record, = records
    assert record["object_id"] == f"{spec['catalogs'][0]['name']}-1"
    assert len(ObjectSequencer(config).build(record).values) == count


@pytest.fixture
def recipe():
    path = Path(__file__).resolve().parents[1] / "configs/nanotron/astropt3-70m-four-modal.yaml"
    return yaml.safe_load(path.read_text())


def _case(recipe, count=4):
    spec = deepcopy(recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"])
    modalities = deepcopy(recipe["model"]["model_config"]["modalities"])
    if count < 4:
        names = ("sdss", "desi")[:count]
        spec["catalogs"] = [next(c for c in spec["catalogs"] if c["name"] == n) for n in names]
        keys = {c["record_key"] for c in spec["catalogs"]}
        modalities = [m for m in modalities if m["record_keys"][0] in keys]
    if count == 5:
        spec["catalogs"][2]["scalar_columns"] = {"Z": "redshift", "ZWARN": "quality"}
        spec["catalogs"].append({
            "name": "dust", "repo": "offline/dust@v1", "decoder": "scalars",
            "scalar_columns": {"ebv": "dust_amount"},
        })
        modalities += deepcopy([m for m in DEFAULT_MODALITIES if m["name"] in ("Z", "ebv")])
    spec["min_catalogs"] = min(count, 3)
    return spec, AstroPT3Config(modalities=modalities)


def _row(spec, index=1):
    source = make_record(index, image_only_fraction=0.0)
    # Keep these tiny payloads local: importing runner tests imports the runner.
    payloads = {
        "legacy_image": {k: source["image"][k] for k in ("band", "flux", "psf_fwhm")},
        "desi_spectrum": {k: source["spectrum"][k] for k in ("flux", "lambda", "mask")},
        "sdss_spectrum": {
            "flux": np.linspace(1, 2, 512, dtype=np.float32),
            "lambda": (4000 * 10 ** (np.arange(512) * 1e-4)).astype(np.float32),
            "mask": np.arange(512) % 7 == 0,
        },
        "hsc_image": {
            "band": [f"hsc-{b}" for b in "grizy"],
            "flux": np.broadcast_to(np.linspace(-0.2, 1, 160), (5, 160, 160)),
            "scale": np.full(5, 0.168),
        },
    }
    row = {}
    for i, catalog in enumerate(spec["catalogs"]):
        suffix = "" if i == 0 else f"_{catalog['name']}"
        row[f"object_id{suffix}"] = f"{catalog['name']}-{index}"
        decoder = catalog["decoder"]
        if decoder != "scalars":
            payload = "image" if decoder.endswith("image") else "spectrum"
            row[f"{payload}{suffix}"] = {
                k: np.asarray(v).tolist() for k, v in payloads[decoder].items()
            }
        for key, column in catalog.get("scalar_columns", {}).items():
            row[f"{column}{suffix}"] = source[key]
    return row


def _decode(stream, row):
    mapped = {k: v for k, v in row.items() if k not in stream.nested}
    for column, fields in stream.nested.items():
        value = row.get(column) or {}
        mapped.update({f"{column}.{field}": np.asarray(value.get(field, [])) for field in fields})
    return stream.decode_row(mapped)["record"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    state = SimpleNamespace(opened=[], streams=[], closed=[], sleeps=[], clusters=[],
                            frames=list, npartitions=1)

    class Resource(nullcontext):
        def __exit__(self, *args):
            state.closed.append(self.enter_result)
            return super().__exit__(*args)

    def cluster(**kwargs):
        state.clusters.append(kwargs)
        return Resource("cluster")

    def open_catalog(url, *, columns):
        state.opened.append((url, columns))
        return url

    class FakeStream:
        def __init__(self, anchor, *rights, min_catalogs=None, **kwargs):
            self.anchor, self.rights, self.minimum = anchor, rights, min_catalogs
            self.kwargs = kwargs
            self.npartitions = state.npartitions
            state.streams.append(self)

        def __iter__(self):
            return iter(state.frames())

    class SingleStream(FakeStream):
        pass

    monkeypatch.setattr(streaming, "LocalCluster", cluster)
    monkeypatch.setattr(streaming, "Client", lambda cluster: Resource("client"))
    monkeypatch.setattr(streaming, "open_catalog", open_catalog)
    monkeypatch.setattr(streaming, "InfiniteStream", SingleStream)
    monkeypatch.setattr(streaming, "CrossMatchStream", FakeStream)
    monkeypatch.setattr(streaming.time, "sleep", state.sleeps.append)
    monkeypatch.setattr(streaming.telemetry, "_REPO_TO_SOURCE", {})
    monkeypatch.delenv("ASTROPT3_TELEMETRY_DIR", raising=False)
    return state


@pytest.mark.parametrize("count", [1, 2, 4, 5])
def test_catalog_count_projection_and_finite_draws(recipe, offline, count):
    spec, config = _case(recipe, count)
    before = deepcopy(spec)
    stream = streaming.CatalogStream(spec, config)
    assert spec == before
    frame = nested_frame([_row(spec)], tuple(stream.nested))
    offline.frames = lambda: [frame, frame]
    with closing(stream.draws(seed=42, rank=2, worker_id=3)) as draws:
        for expected_epoch in range(2):
            epoch, records = next(draws)
            assert epoch == expected_epoch
            assert len(list(records)) == 1
            assert list(records) == []
    assert offline.closed == ["client", "cluster"]
    assert len(offline.streams) == 1
    opened = offline.streams[0]
    assert type(opened).__name__ == ("SingleStream" if count == 1 else "FakeStream")
    assert opened.anchor == f"hf://datasets/{spec['catalogs'][0]['repo']}"
    assert opened.kwargs == {
        "client": "client", "seed": loader.consumer_seed(42, 2, 3, 0),
        "partitions_per_chunk": spec["partitions_per_chunk"],
    }
    assert opened.minimum == (spec["min_catalogs"] if count > 1 else None)
    assert list(opened.rights) == [
        {"other": f"hf://datasets/{c['repo']}", "how": spec["how"],
         "suffixes": ("", f"_{c['name']}"), "radius_arcsec": spec["radius_arcsec"]}
        for c in spec["catalogs"][1:]
    ]
    expected_nested = {}
    for i, (catalog, (url, columns)) in enumerate(zip(spec["catalogs"], offline.opened, strict=True)):
        decoder = catalog["decoder"]
        payload = [] if decoder == "scalars" else ["image" if decoder.endswith("image") else "spectrum"]
        assert url == f"hf://datasets/{catalog['repo']}"
        assert columns == ["object_id", "ra", "dec", *payload, *catalog.get("scalar_columns", {}).values()]
        if payload:
            suffix = "" if i == 0 else f"_{catalog['name']}"
            fields = (("band", "flux") if decoder == "legacy_image" else
                      ("band", "flux", "scale") if decoder == "hsc_image" else
                      ("flux", "lambda", "mask"))
            expected_nested[payload[0] + suffix] = fields
    assert stream.nested == expected_nested
    assert offline.clusters[0]["threads_per_worker"] == spec["partitions_per_chunk"]
    assert offline.clusters[0]["memory_limit"] == spec["worker_memory_limit"]


def test_configured_loader_preserves_alignment_across_echoes(recipe, offline):
    # Five catalogs exercise all four payload decoders plus scalar-only outputs.
    spec, config = _case(recipe, 5)
    dataset_args = {**recipe["data_stages"][0]["data"]["dataset"], "catalog_stream": spec}
    batches = loader.build_astropt3_dataloader(
        SimpleNamespace(**dataset_args), config,
        micro_batch_size=1, sequence_length=1024, dp_rank=0, dp_size=1, num_workers=0,
    )
    stream = batches.dataset.catalog_stream
    assert stream.spec == streaming.CatalogStream(spec, config).spec
    assert batches.dataset.echo_factor == 4
    row = _row(spec)
    expected = ObjectSequencer(config).build(_decode(stream, row))
    assert len(expected.values) == 6
    frame = nested_frame([row], tuple(stream.nested))
    offline.frames = lambda: [frame]
    with closing(iter(batches)) as iterator:
        for exposure in range(4):
            flat = next(iterator)
            assert batches.echo_index == exposure
            assert batches.echo_factor == 4
            assert "_astropt3_echo" not in flat
            assert flat["input_ids"].device.type == "cpu"
            assert flat["hsc_images_values"].shape == (144, 320)
            for modality in config.modalities:
                name = modality["name"]
                mask = flat[f"{name}_mask"]
                assert torch.equal(mask, flat["input_ids"] == modality["token_ids"][1])
                assert int(mask.sum()) == len(expected.values[name])
                torch.testing.assert_close(flat[f"{name}_values"], expected.values[name])
                torch.testing.assert_close(flat[f"{name}_positions"], expected.positions[name])
    assert len(offline.streams) == 1
    assert offline.closed == ["client", "cluster"]


def test_renamed_sdss_modality_and_padding(recipe, offline):
    spec, config = _case(recipe, 1)
    spec["catalogs"][0]["record_key"] = "observed_spectrum"
    modality = {**config.modalities[0], "name": "optical_flux", "record_keys": ["observed_spectrum"],
                "token_ids": [35, 36, 37]}
    config = AstroPT3Config(modalities=[modality])
    dataset = loader.PackedMicroBatches(config, 1, 16, catalog_stream=spec, echo_factor=2)
    row = _row(spec)
    original = deepcopy(row["spectrum"])
    row["spectrum"] = {k: np.pad(v, (2, 1)).tolist() for k, v in original.items()}
    row["spectrum"]["lambda"][0] = -1
    decoded = _decode(dataset.catalog_stream, row)
    assert decoded is not None
    assert set(decoded) == {"object_id", "observed_spectrum"}
    for key, values in original.items():
        np.testing.assert_array_equal(decoded["observed_spectrum"][key], values)
    assert decoded["observed_spectrum"]["flux"].dtype == np.float32
    assert decoded["observed_spectrum"]["mask"].dtype == bool
    frame = nested_frame([row], ("spectrum",))
    offline.frames = lambda: [frame]
    with closing(iter(dataset)) as batches:
        batch = next(batches)
    assert batch["input_ids"][0, :5].tolist() == [1, 35, 36, 36, 37]
    assert batch["optical_flux_values"].shape == (2, 256)
    assert not batch["optical_flux_values"][0, 0]  # masked bin remains aligned


@pytest.mark.parametrize("catalog_index", range(4))
@pytest.mark.parametrize("missing", [None, pd.NA, np.nan])
def test_optional_catalogs_and_right_only_identity(recipe, catalog_index, missing):
    spec, config = _case(recipe)
    stream = streaming.CatalogStream(spec, config)
    catalog = spec["catalogs"][catalog_index]
    suffix = "" if catalog_index == 0 else f"_{catalog['name']}"
    payload = "image" if catalog["decoder"].endswith("image") else "spectrum"
    row = _row(spec)
    selected = {key: missing for key in row if key.startswith("object_id")}
    selected.update({k: row[k] for k in (f"object_id{suffix}", f"{payload}{suffix}")})
    record = _decode(stream, selected)
    assert record is not None
    expected_id = row[f"object_id{suffix}"]
    if catalog_index:
        expected_id = f"{catalog['name']}:{expected_id}"
    assert record["object_id"] == expected_id
    assert set(record) == {"object_id", catalog["record_key"]}
    np.testing.assert_array_equal(record[catalog["record_key"]]["flux"],
                                  np.asarray(row[f"{payload}{suffix}"]["flux"], dtype=np.float32))
    assert len(ObjectSequencer(config).build(record).values) == 1
    selected[f"object_id{suffix}"] = missing
    with pytest.raises(ValueError, match="object_id"):
        _decode(stream, selected)


def test_stream_retains_matched_and_unpaired_records(recipe, offline):
    from astropt3.tokenization import antispiralise, patchify_image

    spec, config = _case(recipe)
    dataset = loader.PackedMicroBatches(config, 1, 4096, catalog_stream=spec)
    row = _row(spec)
    rows = [row]
    for index, catalog in enumerate(spec["catalogs"]):
        suffix = "" if index == 0 else f"_{catalog['name']}"
        row[f"object_id{suffix}"] = "same-id"
        payload = "image" if catalog["decoder"].endswith("image") else "spectrum"
        rows.append({key: row[key] for key in (f"object_id{suffix}", f"{payload}{suffix}")})
    offline.frames = lambda: [nested_frame(rows, tuple(dataset.catalog_stream.nested))]
    with closing(dataset._open_records(None)) as records:
        decoded = [next(records) for _ in rows]
    assert [record["object_id"] for record in decoded] == [
        "same-id", "same-id", "sdss:same-id", "desi:same-id", "hsc:same-id",
    ]
    sequences = [dataset.sequencer.build(record) for record in decoded]
    assert [set(sequence.values) for sequence in sequences] == [
        {"images", "spectra", "sdss_spectra", "hsc_images"},
        {"images"}, {"sdss_spectra"}, {"spectra"}, {"hsc_images"},
    ]
    raw = torch.tensor(row["image_hsc"]["flux"], dtype=torch.float32)[:, 32:128, 32:128]
    factor = 10 ** ((22.5 - 27.0) / 2.5) * (0.262 / 0.168) ** 2
    expected = patchify_image(torch.asinh(raw * factor / 0.01), 8)
    torch.testing.assert_close(antispiralise(sequences[-1].values["hsc_images"]), expected)
    flat = loader.flatten_packed_batch(dataset.collator(sequences), config, 4096)
    assert flat["hsc_images_values"].shape == (288, 320)
    assert int(flat["hsc_images_mask"].sum()) == 288
    assert all(flat[f"{name}_mask"].any() for name in config.modality_registry().names())
    assert offline.closed == ["client", "cluster"]


@pytest.mark.parametrize("with_spectrum", [False, True])
@pytest.mark.parametrize("flag", [0, 4, None, np.nan, pd.NA, "absent"])
def test_redshift_quality_gating(recipe, with_spectrum, flag):
    spec, config = _case(recipe, 1)
    catalog = spec["catalogs"][0]
    catalog["scalar_columns"] = {"Z": "redshift", "ZWARN": "quality"}
    modalities = config.modalities if with_spectrum else []
    if not with_spectrum:
        catalog["decoder"] = "scalars"
        del catalog["record_key"]
    config = AstroPT3Config(modalities=[*modalities, DEFAULT_MODALITIES[2]])
    stream = streaming.CatalogStream(spec, config)
    row = _row(spec)
    if isinstance(flag, str):
        del row["quality"]
    else:
        row["quality"] = flag
    record = _decode(stream, row)
    good = isinstance(flag, int) and flag == 0
    if not with_spectrum and not good:
        assert record is None
    else:
        assert record is not None
        values = ObjectSequencer(config).build(record).values
        assert ("Z" in values) == good
        if with_spectrum:
            assert "sdss_spectra" in values
        if good:
            torch.testing.assert_close(values["Z"], torch.tensor([[np.log1p(row["redshift"])]], dtype=torch.float32))


def test_missing_redshift_catalog_does_not_claim_right_only_row(recipe):
    spec, config = _case(recipe, 5)
    stream = streaming.CatalogStream(spec, config)
    row = _row(spec)
    record = _decode(stream, {"object_id_hsc": "right-only", "image_hsc": row["image_hsc"]})
    assert record is not None
    assert record["object_id"] == "hsc:right-only"
    assert set(ObjectSequencer(config).build(record).values) == {"hsc_images"}


def test_empty_and_no_active_records(recipe, offline):
    spec, config = _case(recipe)
    config = AstroPT3Config(modalities=[config.modalities[2]])
    stream = streaming.CatalogStream(spec, config)
    row = _row(spec)
    inactive = {"object_id": "inactive", "image": row["image"]}
    assert _decode(stream, inactive) is None
    assert _decode(stream, {"object_id": "empty"}) is None
    with pytest.raises(ValueError, match="no supported data or object_id"):
        _decode(stream, {})
    frames = [pd.DataFrame(), nested_frame([inactive, row], tuple(stream.nested))]
    offline.frames = lambda: frames
    with closing(stream.draws(seed=0, rank=0, worker_id=0)) as draws:
        epoch, records = next(draws)
        assert epoch == 1
        assert [r["object_id"] for r in records] == [row["object_id"]]


@pytest.mark.parametrize("target,patch,match", [
    ("catalog", {"record_key": "spectrum"}, "unique"),
    ("catalog", {"decoder": "unknown"}, "decoder"),
    ("catalog", {"name": "sdss"}, "unique"),
    ("catalog", {"repo": "https://example.invalid/data"}, "namespace"),
    ("catalog", {"scalar_columns": {"image": "flat"}}, "unique"),
    ("catalog", {"scalar_columns": {"ebv": "image.ebv"}}, "flat"),
    ("model", {"source": "desi"}, "source"),
    ("model", {"record_keys": ["unmapped"]}, "compatible catalog output"),
    ("stream", {"min_catalogs": 0}, "min_catalogs"),
    ("stream", {"min_catalogs": 5}, "min_catalogs"),
    ("stream", {"min_catalogs": True}, "min_catalogs"),
    ("stream", {"how": "inner"}, "how"),
    ("stream", {"radius_arcsec": 0}, "radius_arcsec"),
    ("stream", {"radius_arcsec": np.nan}, "radius_arcsec"),
    ("stream", {"radius_arcsec": np.inf}, "radius_arcsec"),
    ("stream", {"radius_arcsec": True}, "radius_arcsec"),
    ("stream", {"radius_arcsec": "1"}, "radius_arcsec"),
    ("stream", {"partitions_per_chunk": 0}, "partitions_per_chunk"),
    ("stream", {"partitions_per_chunk": True}, "partitions_per_chunk"),
    ("stream", {"worker_memory_limit": ""}, "worker_memory_limit"),
    ("stream", {"worker_memory_limit": 123}, "worker_memory_limit"),
    ("stream", {"min_surveys": 1}, "unknown settings"),
    ("stream", {"count_fraction_threshold": 0.0}, "unknown settings"),
    ("stream", {"count_fraction_threshold": 0.5}, "unknown settings"),
])
def test_invalid_configuration_fails_before_opening_catalogs(recipe, offline, target, patch, match):
    spec, config = _case(recipe)
    if target == "model":
        config = AstroPT3Config(modalities=[{**config.modalities[2], **patch}])
    else:
        (spec if target == "stream" else spec["catalogs"][0]).update(patch)
    with pytest.raises(ValueError, match=match):
        streaming.CatalogStream(spec, config)
    assert not offline.opened


@pytest.mark.parametrize("separate_flag", [False, True])
def test_redshift_requires_quality_from_same_catalog(separate_flag):
    catalogs = [{"name": "redshifts", "repo": "offline/redshifts", "decoder": "scalars",
                 "scalar_columns": {"Z": "redshift"}}]
    if separate_flag:
        catalogs.append({"name": "flags", "repo": "offline/flags", "decoder": "scalars",
                         "scalar_columns": {"ZWARN": "quality"}})
    with pytest.raises(ValueError, match="ZWARN"):
        streaming.CatalogStream({"catalogs": catalogs}, AstroPT3Config(modalities=[DEFAULT_MODALITIES[2]]))


def test_retry_reopens_with_new_seed_and_cleans_resources(recipe, offline):
    spec, config = _case(recipe, 1)
    stream = streaming.CatalogStream(spec, config)
    frame = nested_frame([_row(spec)], tuple(stream.nested))

    def frames():
        yield frame
        raise OSError("transient read")

    offline.frames = frames
    with closing(stream.draws(seed=42, rank=1, worker_id=2)) as draws:
        for epoch in range(2):
            actual, records = next(draws)
            assert actual == epoch
            assert len(list(records)) == 1
    assert offline.sleeps == [5]
    assert [s.kwargs["seed"] for s in offline.streams] == [loader.consumer_seed(42, 1, 2, g) for g in range(2)]
    assert len(offline.opened) == 2
    assert offline.closed == ["client", "cluster"] * 2


def test_transport_retries_are_bounded(recipe, offline, monkeypatch):
    spec, config = _case(recipe, 1)
    monkeypatch.setattr(loader, "_MAX_NET_RETRIES", 2)
    monkeypatch.setattr(loader, "_MAX_NET_RETRY_WAIT", 6)

    def frames():
        raise TimeoutError("unavailable")

    offline.frames = frames
    with (
        closing(streaming.CatalogStream(spec, config).draws(seed=0, rank=0, worker_id=0)) as draws,
        pytest.raises(TimeoutError, match="unavailable"),
    ):
        next(draws)
    assert len(offline.streams) == 3
    assert offline.sleeps == [5, 6]
    assert offline.closed == ["client", "cluster"] * 3


@pytest.mark.parametrize("failure", ["decode", "unknown", "empty-footprint", "ended"])
def test_nonretryable_errors_close_resources(recipe, offline, failure):
    spec, config = _case(recipe)
    stream = streaming.CatalogStream(spec, config)
    error, message = RuntimeError, "ended unexpectedly"
    if failure == "decode":
        row = _row(spec)
        row["image_hsc"]["flux"][0][80][80] = np.nan
        frame = nested_frame([row], tuple(stream.nested))
        offline.frames = lambda: [frame]
        error, message = ValueError, "HSC Wide"
    elif failure == "empty-footprint":
        offline.npartitions = 0
        error, message = ValueError, "no partitions"
    elif failure == "unknown":
        def frames():
            raise RuntimeError("unexpected bug")
        offline.frames = frames
        message = "unexpected bug"
    with (
        closing(stream.draws(seed=0, rank=0, worker_id=0)) as draws,
        pytest.raises(error, match=message),
    ):
        next(draws)
    assert len(offline.streams) == 1
    assert not offline.sleeps
    assert offline.closed == ["client", "cluster"]
