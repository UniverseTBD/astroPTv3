"""Offline contracts for configured catalogs, independent of the training runner."""

from contextlib import closing, nullcontext
from copy import deepcopy
from inspect import signature
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


def test_installed_lsdb_has_required_multi_catalog_api():
    from lsdb.streams.catalog_streams import CrossMatchStream

    assert "min_surveys" in signature(CrossMatchStream).parameters


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
    spec["min_surveys"] = min(count, 3)
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
        def __init__(self, anchor, *rights, min_surveys=None, **kwargs):
            self.anchor, self.rights, self.minimum = anchor, rights, min_surveys
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
        **({"count_fraction_threshold": spec["count_fraction_threshold"]} if count > 1 else {}),
    }
    assert opened.minimum == (spec["min_surveys"] if count > 1 else None)
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
            fields = (("band", "flux", "psf_fwhm") if decoder == "legacy_image" else
                      ("band", "flux", "scale") if decoder == "hsc_image" else
                      ("flux", "lambda", "mask"))
            expected_nested[payload[0] + suffix] = fields
    assert stream.nested == expected_nested
    assert offline.clusters[0]["threads_per_worker"] == spec["partitions_per_chunk"]
    assert offline.clusters[0]["memory_limit"] == spec["worker_memory_limit"]


@pytest.mark.parametrize("count", [1, 2, 4, 5])
def test_packed_modalities_and_alignment(recipe, offline, count):
    spec, config = _case(recipe, count)
    dataset = loader.PackedMicroBatches(config, 1, 1024, catalog_stream=spec, echo_factor=2)
    row = _row(spec)
    assert dataset.catalog_stream is not None
    record = _decode(dataset.catalog_stream, row)
    assert record is not None
    expected = ObjectSequencer(config).build(record)
    assert len(expected.values) == (6 if count == 5 else count)
    frame = nested_frame([row], tuple(dataset.catalog_stream.nested))
    offline.frames = lambda: [frame]
    with closing(iter(dataset)) as batches:
        for _ in range(2):
            flat = next(batches)
            assert flat["input_ids"].shape == (1, 1024)
            assert flat["input_ids"].device.type == "cpu"
            for modality in config.modalities:
                name = modality["name"]
                mask = flat[f"{name}_mask"]
                assert mask.dtype == torch.bool
                assert torch.equal(mask, flat["input_ids"] == modality["token_ids"][1])
                assert int(mask.sum()) == len(expected.values[name])
                torch.testing.assert_close(flat[f"{name}_values"], expected.values[name])
                torch.testing.assert_close(flat[f"{name}_positions"], expected.positions[name])
                assert torch.isfinite(flat[f"{name}_values"]).all()
    assert offline.closed == ["client", "cluster"]


def test_recipe_and_shared_builder(recipe, offline):
    dataset_args = recipe["data_stages"][0]["data"]["dataset"]
    assert dataset_args["echo_factor"] == 4
    assert len(recipe["model"]["model_config"]["modalities"]) == 4
    batches = loader.build_astropt3_dataloader(
        SimpleNamespace(**dataset_args), SimpleNamespace(**recipe["model"]["model_config"]),
        micro_batch_size=1, sequence_length=1024, dp_rank=0, dp_size=1, num_workers=0,
    )
    expected = streaming.CatalogStream(dataset_args["catalog_stream"], batches.dataset.config).spec
    assert batches.dataset.catalog_stream.spec == expected
    assert batches.dataset.echo_factor == 4
    stream = batches.dataset.catalog_stream
    frame = nested_frame([_row(stream.spec)], tuple(stream.nested))
    offline.frames = lambda: [frame]
    with closing(iter(batches)) as iterator:
        for exposure in range(4):
            batch = next(iterator)
            assert batches.echo_index == exposure
            assert batches.echo_factor == 4
            assert "_astropt3_echo" not in batch
            assert batch["hsc_images_values"].shape == (144, 320)
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
def test_optional_catalogs_and_right_only_identity(recipe, catalog_index):
    spec, config = _case(recipe)
    stream = streaming.CatalogStream(spec, config)
    catalog = spec["catalogs"][catalog_index]
    suffix = "" if catalog_index == 0 else f"_{catalog['name']}"
    payload = "image" if catalog["decoder"].endswith("image") else "spectrum"
    row = _row(spec)
    record = _decode(stream, {k: row[k] for k in (f"object_id{suffix}", f"{payload}{suffix}")})
    assert record is not None
    expected_id = row[f"object_id{suffix}"]
    if catalog_index:
        expected_id = f"{catalog['name']}:{expected_id}"
    assert record["object_id"] == expected_id
    assert set(record) == {"object_id", catalog["record_key"]}
    np.testing.assert_array_equal(record[catalog["record_key"]]["flux"],
                                  np.asarray(row[f"{payload}{suffix}"]["flux"], dtype=np.float32))
    assert len(ObjectSequencer(config).build(record).values) == 1


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
    with pytest.raises(ValueError, match="object_id"):
        _decode(stream, {})
    frames = [pd.DataFrame(), nested_frame([inactive, row], tuple(stream.nested))]
    offline.frames = lambda: frames
    with closing(stream.draws(seed=0, rank=0, worker_id=0)) as draws:
        epoch, records = next(draws)
        assert epoch == 1
        assert [r["object_id"] for r in records] == [row["object_id"]]


@pytest.mark.parametrize("patch,match", [
    ({"record_key": "spectrum"}, "unique"),
    ({"decoder": "unknown"}, "decoder"),
    ({"name": "sdss"}, "unique"),
    ({"repo": "https://example.invalid/data"}, "namespace"),
    ({"scalar_columns": {"image": "flat"}}, "unique"),
    ({"scalar_columns": {"ebv": "image.ebv"}}, "flat"),
])
def test_invalid_catalogs(recipe, offline, patch, match):
    spec, config = _case(recipe)
    spec["catalogs"][0].update(patch)
    with pytest.raises(ValueError, match=match):
        streaming.CatalogStream(spec, config)
    assert not offline.opened


@pytest.mark.parametrize("patch,match", [
    ({"source": "desi"}, "source"),
    ({"record_keys": ["unmapped"]}, "compatible catalog output"),
])
def test_model_fields_must_match_decoder(recipe, patch, match):
    spec, config = _case(recipe, 1)
    config = AstroPT3Config(modalities=[{**config.modalities[0], **patch}])
    with pytest.raises(ValueError, match=match):
        streaming.CatalogStream(spec, config)


@pytest.mark.parametrize("separate_flag", [False, True])
def test_redshift_requires_quality_from_same_catalog(separate_flag):
    catalogs = [{"name": "redshifts", "repo": "offline/redshifts", "decoder": "scalars",
                 "scalar_columns": {"Z": "redshift"}}]
    if separate_flag:
        catalogs.append({"name": "flags", "repo": "offline/flags", "decoder": "scalars",
                         "scalar_columns": {"ZWARN": "quality"}})
    with pytest.raises(ValueError, match="ZWARN"):
        streaming.CatalogStream({"catalogs": catalogs}, AstroPT3Config(modalities=[DEFAULT_MODALITIES[2]]))


@pytest.mark.parametrize("key,value", [
    ("min_surveys", 0), ("min_surveys", 5), ("min_surveys", True),
    ("how", "inner"), ("radius_arcsec", 0), ("radius_arcsec", np.nan),
    ("count_fraction_threshold", -0.1), ("count_fraction_threshold", 1.1),
    ("count_fraction_threshold", np.inf), ("partitions_per_chunk", 0),
    ("partitions_per_chunk", True), ("worker_memory_limit", ""),
    ("worker_memory_limit", 123),
])
def test_invalid_stream_bounds(recipe, offline, key, value):
    spec, config = _case(recipe)
    spec[key] = value
    with pytest.raises(ValueError):
        streaming.CatalogStream(spec, config)
    assert not offline.opened


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
