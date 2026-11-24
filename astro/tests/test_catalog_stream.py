"""Offline contracts for configured catalogs, independent of the training runner."""

import io
from contextlib import closing, nullcontext, suppress
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from legacy_fixture import make_record, nested_frame
from PIL import Image

from astropt3.configuration_astropt3 import (
    DEFAULT_MODALITIES,
    LEGACY_SOUTH_MODALITIES,
    AstroPT3Config,
)
from astropt3.data import catalog_stream as streaming
from astropt3.data import nanotron_loader as loader
from astropt3.data.packing import ObjectSequencer
from astropt3.tokenization import antispiralise, patchify_image

NORTH_BANDS = {f"legacy_north_{band}" for band in "grz"}
HSC_BANDS = {f"hsc_{band}" for band in "grizy"}


@pytest.mark.parametrize("count", [1, 4])
def test_real_lsdb_stream_decodes_local_catalogs(recipe, monkeypatch, tmp_path, count):
    from lsdb.loaders.dataframe.from_dataframe import from_dataframe
    from lsdb.loaders.hats.read_hats import open_catalog
    from lsdb.streams.catalog_streams import (
        CrossMatchStream,
        InfiniteStream,
        _restrict_to_footprint,
    )

    spec, config = _case(recipe, count)
    spec["partitions_per_chunk"] = 1
    catalogs = {}
    for catalog in _catalogs(spec):
        payload = "image" if catalog["decoder"].endswith("image") else "spectrum"
        row = {"ra": 10.0, "dec": 0.0, **_row({"anchors": [catalog], "catalogs": []})}
        if catalog["decoder"] == "hsc_image":
            row["image"]["ivar"] = row["image"]["flux"]  # Make the unused leaf present on disk.
        elif catalog["decoder"] in ("desi_spectrum", "sdss_spectrum"):
            row["spectrum"]["ivar"] = row["spectrum"]["flux"]  # Unused spectrum leaf on disk.
        source = from_dataframe(nested_frame([row], (payload,)), catalog_name=catalog["name"])
        url = f"hf://datasets/{catalog['repo']}"
        if catalog["decoder"] in ("hsc_image", "desi_spectrum", "sdss_spectrum"):
            # Read a real Parquet projection for image and spectrum partners (left join needs no margin).
            if catalog is not spec["anchors"][0]:
                catalog["how"] = "left"
                spec["min_catalogs"] = 1
            path = tmp_path / catalog["name"]
            source.write_catalog(path, as_collection=False, progress_bar=False, should_write_skymap=False)
            catalogs[url] = path
        else:
            catalogs[url] = source

    def open_local(url, **kwargs):
        source = catalogs[url]
        if isinstance(source, Path):
            catalog = open_catalog(source, **kwargs)
            if source.name == "hsc":
                assert catalog.meta.get_subcolumns() == ["image.band", "image.flux", "image.scale"]
            else:
                assert catalog.meta.get_subcolumns() == ["spectrum.flux", "spectrum.lambda", "spectrum.mask"]
            return catalog
        return source

    monkeypatch.setattr(streaming, "open_catalog", open_local)
    monkeypatch.setattr(streaming, "Client", lambda cluster: nullcontext())
    monkeypatch.setattr(streaming, "CrossMatchStream", CrossMatchStream)
    monkeypatch.setattr(streaming, "InfiniteStream", InfiniteStream)
    monkeypatch.setattr(streaming, "_restrict_to_footprint", _restrict_to_footprint)
    stream = streaming.CatalogStream(spec, config)
    with closing(stream.draws(seed=0, rank=0, worker_id=0)) as draws:
        _, records = next(draws)
        record, = records
    name = spec['anchors'][0]['name']
    assert record["object_id"] == f"{name}:{name}-1"
    assert set(ObjectSequencer(config).build(record).values) == {m["name"] for m in config.modalities}


@pytest.fixture
def recipe():
    path = Path(__file__).resolve().parents[1] / "configs/controls/astropt3-70m-north.yaml"
    control = yaml.safe_load(path.read_text())
    control['data_stages'][0]['data']['dataset']['catalog_stream'].pop('split', None)
    return control


def _catalogs(spec):
    return spec["anchors"] + spec["catalogs"]


def _case(recipe, count=4):
    spec = deepcopy(recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"])
    modalities = deepcopy(recipe["model"]["model_config"]["modalities"])
    # the four survey payloads without their scalars; galaxies-with-hats and
    # the recipe's scalar wiring have their own tests
    spec["catalogs"] = [c for c in spec["catalogs"] if c["decoder"] not in ("rgb_png", "scalars")]
    for catalog in _catalogs(spec):
        catalog.pop("scalar_columns", None)
    modalities = [m for m in modalities if m["family"] != "scalar" and m["source"] != "galaxies_with_hats"]
    if count < 4:
        names = ("sdss", "desi")[:count]
        root, *partners = [next(c for c in spec["catalogs"] if c["name"] == n) for n in names]
        spec["anchors"] = [{**root, "draw_weight": 1}]
        spec["catalogs"] = partners
        keys = {c["record_key"] for c in _catalogs(spec)}
        modalities = [m for m in modalities if m["record_keys"][0] in keys]
    if count == 5:
        spec["catalogs"][1]["scalar_columns"] = {"Z": "redshift", "ZWARN": "quality"}
        spec["catalogs"].append({
            "name": "dust", "repo": "offline/dust@v1", "decoder": "scalars",
            "scalar_columns": {"ebv": "dust_amount"},
        })
        modalities += deepcopy([m for m in DEFAULT_MODALITIES if m["name"] in ("Z", "ebv")])
    spec["min_catalogs"] = min(count, 3)
    return spec, AstroPT3Config(modalities=modalities)


@pytest.mark.parametrize("count, override, expected", [(1, None, 1), (4, None, 4), (1, 2, 2)])
def test_partition_chunk_defaults(recipe, count, override, expected):
    spec, config = _case(recipe, count)
    spec.pop("partitions_per_chunk", None)
    if override is not None:
        spec["partitions_per_chunk"] = override
    assert streaming.CatalogStream(spec, config).spec["partitions_per_chunk"] == expected


def _png(index, size=256):
    pixels = (np.arange(size * size * 3, dtype=np.uint32).reshape(size, size, 3) * (index + 1)) % 256
    buffer = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


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
            "flux": np.broadcast_to(np.linspace(-0.2, 1, 160), (5, 160, 160)) + np.arange(5)[:, None, None],
            "scale": np.full(5, 0.168),
        },
        "rgb_png": {"bytes": _png(index), "path": f"{index}.png"},
    }
    row = {}
    for i, catalog in enumerate(_catalogs(spec)):
        suffix = "" if i == 0 else f"_{catalog['name']}"
        decoder = catalog["decoder"]
        row[f"{streaming._ID_COLUMNS.get(decoder, 'object_id')}{suffix}"] = f"{catalog['name']}-{index}"
        if decoder == "rgb_png":
            row[f"image{suffix}"] = payloads[decoder]
        elif decoder != "scalars":
            payload = "image" if decoder.endswith("image") else "spectrum"
            row[f"{payload}{suffix}"] = {
                k: np.asarray(v).tolist() for k, v in payloads[decoder].items()
            }
        for key, column in catalog.get("scalar_columns", {}).items():
            row[f"{column}{suffix}"] = source.get(key, 0.5)  # gwh_* scalars are not in the legacy fixture
    return row


def _decode(stream, row):
    stream, = stream.stars
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
    monkeypatch.setattr(streaming, "_restrict_to_footprint",
                        lambda anchor, footprint, rights, minimum: (anchor, ("footprint", anchor, tuple(rights), minimum)))
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
    frame = nested_frame([_row(spec)], tuple(stream.stars[0].nested))
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
    assert opened.anchor == f"hf://datasets/{spec['anchors'][0]['repo']}"
    assert opened.kwargs == {
        "client": "client", "seed": loader.consumer_seed(42, 2, 3, 0),
        "partitions_per_chunk": spec["partitions_per_chunk"],
    }
    assert opened.minimum == (spec["min_catalogs"] if count > 1 else None)
    assert list(opened.rights) == [
        {"other": f"hf://datasets/{c['repo']}", "how": spec["how"],
         "suffixes": ("", f"_{c['name']}"), "radius_arcsec": spec["radius_arcsec"]}
        for c in spec["catalogs"]
    ]
    expected_nested = {}
    for i, (catalog, (url, columns)) in enumerate(zip(_catalogs(spec), offline.opened, strict=True)):
        decoder = catalog["decoder"]
        payload = [] if decoder == "scalars" else ["image" if decoder.endswith("image") else "spectrum"]
        assert url == f"hf://datasets/{catalog['repo']}"
        read_payload = {
            "legacy_image": ["image.band", "image.flux"],
            "hsc_image": ["image.band", "image.flux", "image.scale"],
            "desi_spectrum": ["spectrum.flux", "spectrum.lambda", "spectrum.mask"],
            "sdss_spectrum": ["spectrum.flux", "spectrum.lambda", "spectrum.mask"],
        }.get(decoder, payload)
        assert columns == ["object_id", "ra", "dec", *read_payload, *catalog.get("scalar_columns", {}).values()]
        if payload:
            suffix = "" if i == 0 else f"_{catalog['name']}"
            fields = (("band", "flux") if decoder == "legacy_image" else
                      ("band", "flux", "scale") if decoder == "hsc_image" else
                      ("flux", "lambda", "mask"))
            expected_nested[payload[0] + suffix] = fields
    assert stream.stars[0].nested == expected_nested
    assert offline.clusters[0]["threads_per_worker"] == spec["partitions_per_chunk"]
    assert offline.clusters[0]["memory_limit"] == spec["worker_memory_limit"]


def test_south_draw_requests_only_band_and_flux(offline):
    repo = "offline/south"
    config = AstroPT3Config(modalities=LEGACY_SOUTH_MODALITIES)
    stream = streaming.CatalogStream({"anchors": [{
        "name": "south", "repo": repo, "decoder": "legacy_south_image",
        "record_key": "legacy_south_image", "draw_weight": 1,
    }], "catalogs": []}, config)
    frame = nested_frame([{"object_id": "1", "image": {
        "band": [f"des-{band}" for band in "griz"],
        "flux": np.ones((4, 160, 160), dtype=np.float32).tolist(),
    }}], ("image",))
    offline.frames = lambda: [frame]
    with closing(stream.draws(seed=0, rank=0, worker_id=0)) as draws:
        _, records = next(draws)
        record, = records
    assert offline.opened == [(f"hf://datasets/{repo}", [
        "object_id", "ra", "dec", "image.band", "image.flux",
    ])]
    assert record["legacy_south_image"]["band"] == ["des-g", "des-r", "des-i", "des-z"]


def test_draws_drop_hf_array2d_extension_before_opening(recipe, offline):
    """HF datasets' Array2D registration makes HSC's schema unreadable in-process."""
    import pyarrow as pa

    class Array2D(pa.ExtensionType):  # stand-in registered under the datasets name
        def __init__(self):
            super().__init__(pa.list_(pa.float32()), "datasets.features.features.Array2DExtensionType")

        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

    schema = pa.schema([pa.field("image", pa.struct([pa.field("flux", pa.list_(Array2D()))]))]).serialize()
    pa.register_extension_type(Array2D())
    try:
        with pytest.raises(pa.ArrowNotImplementedError, match="extension"):
            pa.ipc.read_schema(pa.BufferReader(schema)).empty_table()
        spec, config = _case(recipe, 1)
        stream = streaming.CatalogStream(spec, config)
        offline.frames = lambda: [nested_frame([_row(spec)], tuple(stream.stars[0].nested))]
        with closing(stream.draws(seed=0, rank=0, worker_id=0)) as draws:
            next(draws)
        pa.ipc.read_schema(pa.BufferReader(schema)).empty_table()
    finally:
        with suppress(pa.ArrowKeyError):
            pa.unregister_extension_type("datasets.features.features.Array2DExtensionType")


def test_configured_loader_preserves_alignment_across_echoes(recipe, offline):
    # Five catalogs exercise all four payload decoders plus scalar-only outputs.
    spec, config = _case(recipe, 5)
    dataset_args = {**recipe["data_stages"][0]["data"]["dataset"], "catalog_stream": spec}
    batches = loader.build_astropt3_dataloader(
        SimpleNamespace(**dataset_args), config,
        micro_batch_size=1, sequence_length=1536, dp_rank=0, dp_size=1, num_workers=0,
    )
    stream = batches.dataset.catalog_stream
    assert stream.spec == streaming.CatalogStream(spec, config).spec
    assert batches.dataset.echo_factor == 4
    row = _row(spec)
    expected = ObjectSequencer(config).build(_decode(stream, row))
    assert set(expected.values) == NORTH_BANDS | HSC_BANDS | {"spectra", "sdss_spectra", "Z", "ebv"}
    frame = nested_frame([row], tuple(stream.stars[0].nested))
    offline.frames = lambda: [frame]
    with closing(iter(batches)) as iterator:
        for exposure in range(4):
            flat = next(iterator)
            assert batches.echo_index == exposure
            assert batches.echo_factor == 4
            assert "_astropt3_echo" not in flat
            assert flat["input_ids"].device.type == "cpu"
            for name in NORTH_BANDS | HSC_BANDS:
                assert flat[f"{name}_values"].shape == (144, 64)
            for modality in config.modalities:
                name = modality["name"]
                mask = flat[f"{name}_mask"]
                assert torch.equal(mask, flat["input_ids"] == modality["token_ids"][1])
                assert int(mask.sum()) == len(expected.values[name])
                torch.testing.assert_close(flat[f"{name}_values"], expected.values[name])
                torch.testing.assert_close(flat[f"{name}_positions"], expected.positions[name])
                assert torch.isfinite(flat[f"{name}_values"]).all()
    assert offline.closed == ["client", "cluster"]


def test_recipe_and_shared_builder(recipe, offline):
    dataset_args = recipe["data_stages"][0]["data"]["dataset"]
    assert dataset_args["echo_factor"] == 4
    assert len(recipe["model"]["model_config"]["modalities"]) == 51
    batches = loader.build_astropt3_dataloader(
        SimpleNamespace(**dataset_args), SimpleNamespace(**recipe["model"]["model_config"]),
        micro_batch_size=1, sequence_length=2048, dp_rank=0, dp_size=1, num_workers=0,
    )
    expected = streaming.CatalogStream(dataset_args["catalog_stream"], batches.dataset.config).spec
    assert batches.dataset.catalog_stream.spec == expected
    assert batches.dataset.echo_factor == 4
    stream = batches.dataset.catalog_stream
    frame = nested_frame([_row(stream.spec)], tuple(stream.stars[0].nested))
    offline.frames = lambda: [frame]
    with closing(iter(batches)) as iterator:
        for exposure in range(4):
            batch = next(iterator)
            assert batches.echo_index == exposure
            assert batches.echo_factor == 4
            assert "_astropt3_echo" not in batch
            for name in NORTH_BANDS | HSC_BANDS:
                assert batch[f"{name}_values"].shape == (144, 64)
            assert batch["galaxies_values"].shape == (256, 768)
            assert batch["gwh_redshift_values"].shape == (1, 1)
    assert len(offline.streams) == 1
    assert offline.closed == ["client", "cluster"]


def test_renamed_sdss_modality_and_padding(recipe, offline):
    spec, config = _case(recipe, 1)
    spec["anchors"][0]["record_key"] = "observed_spectrum"
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
    assert decoded['source_ids'] == {'sdss': 'sdss-1'}
    assert decoded['anchor'] == decoded['anchor_path'] == 'sdss'
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
    from astropt3.data.spectral import spectral_normalize

    values = batch["optical_flux_values"].flatten()[:512]
    expected = spectral_normalize(
        torch.as_tensor(np.asarray(original["flux"], np.float32)),
        torch.as_tensor(np.asarray(original["lambda"], np.float32)), source="sdss")
    keep = ~torch.as_tensor(np.asarray(original["mask"], bool))
    torch.testing.assert_close(values[keep].float(), expected[keep])  # real bins stay aligned
    assert values[0] != 0  # masked bin 0 is fill noise, not an exact zero


@pytest.mark.parametrize("catalog_index", range(4))
@pytest.mark.parametrize("missing", [None, pd.NA, np.nan])
def test_optional_catalogs_and_right_only_identity(recipe, catalog_index, missing):
    spec, config = _case(recipe)
    stream = streaming.CatalogStream(spec, config)
    catalog = _catalogs(spec)[catalog_index]
    suffix = "" if catalog_index == 0 else f"_{catalog['name']}"
    payload = "image" if catalog["decoder"].endswith("image") else "spectrum"
    row = _row(spec)
    selected = {key: missing for key in row if key.startswith("object_id")}
    selected.update({k: row[k] for k in (f"object_id{suffix}", f"{payload}{suffix}")})
    record = _decode(stream, selected)
    assert record is not None
    expected_id = row[f"object_id{suffix}"]
    assert record["object_id"] == f"{catalog['name']}:{expected_id}"
    assert record['source_ids'] == {catalog['name']: expected_id}
    assert record['anchor'] == (None if catalog_index else catalog['name'])
    np.testing.assert_array_equal(record[catalog["record_key"]]["flux"],
                                  np.asarray(row[f"{payload}{suffix}"]["flux"], dtype=np.float32))
    assert set(ObjectSequencer(config).build(record).values) == {
        m["name"] for m in config.modalities if m["record_keys"] == [catalog["record_key"]]
    }
    selected[f"object_id{suffix}"] = missing
    with pytest.raises(ValueError, match="object_id"):
        _decode(stream, selected)


def test_stream_retains_matched_and_unpaired_records(recipe, offline):
    spec, config = _case(recipe)
    dataset = loader.PackedMicroBatches(config, 1, 4096, catalog_stream=spec)
    row = _row(spec)
    rows = [row]
    for index, catalog in enumerate(_catalogs(spec)):
        suffix = "" if index == 0 else f"_{catalog['name']}"
        row[f"object_id{suffix}"] = "same-id"
        payload = "image" if catalog["decoder"].endswith("image") else "spectrum"
        rows.append({key: row[key] for key in (f"object_id{suffix}", f"{payload}{suffix}")})
    offline.frames = lambda: [nested_frame(rows, tuple(dataset.catalog_stream.stars[0].nested))]
    with closing(dataset._open_records(None)) as records:
        decoded = [next(records) for _ in rows]
    assert [record["object_id"] for record in decoded] == [
        "legacy:same-id", "legacy:same-id", "sdss:same-id", "desi:same-id", "hsc:same-id",
    ]
    sequences = [dataset.sequencer.build(record) for record in decoded]
    assert [set(sequence.values) for sequence in sequences] == [
        NORTH_BANDS | HSC_BANDS | {"spectra", "sdss_spectra"},
        NORTH_BANDS, {"sdss_spectra"}, {"spectra"}, HSC_BANDS,
    ]
    raw = torch.tensor(row["image_hsc"]["flux"], dtype=torch.float32)[:, 32:128, 32:128]
    factor = 10 ** ((22.5 - 27.0) / 2.5) * (0.262 / 0.168) ** 2
    flat = loader.flatten_packed_batch(dataset.collator(sequences), config, 4096)
    for band, plane in zip("grizy", raw, strict=True):
        name = f"hsc_{band}"
        expected = patchify_image(torch.asinh(plane[None] * factor / 0.01), 8)
        torch.testing.assert_close(antispiralise(sequences[-1].values[name]), expected)
        assert flat[f"{name}_values"].shape == (288, 64)
        assert int(flat[f"{name}_mask"].sum()) == 288
    assert all(flat[f"{name}_mask"].any() for name in config.modality_registry().names())
    assert offline.closed == ["client", "cluster"]


@pytest.mark.parametrize("with_spectrum", [False, True])
@pytest.mark.parametrize("flag", [0, 4, None, np.nan, pd.NA, "absent"])
def test_redshift_quality_gating(recipe, with_spectrum, flag):
    spec, config = _case(recipe, 1)
    catalog = spec["anchors"][0]
    catalog["scalar_columns"] = {"Z": "redshift", "ZWARN": "quality"}
    modalities = config.modalities if with_spectrum else []
    if not with_spectrum:
        catalog["decoder"] = "scalars"
        del catalog["record_key"]
    config = AstroPT3Config(modalities=[*modalities, *(m for m in DEFAULT_MODALITIES if m["name"] == "Z")])
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
    assert set(ObjectSequencer(config).build(record).values) == HSC_BANDS


def test_empty_and_no_active_records(recipe, offline):
    spec, config = _case(recipe)
    config = AstroPT3Config(modalities=[m for m in config.modalities if m["name"] == "sdss_spectra"])
    stream = streaming.CatalogStream(spec, config)
    row = _row(spec)
    inactive = {"object_id": "inactive", "image": row["image"]}
    assert _decode(stream, inactive) is None
    assert _decode(stream, {"object_id": "empty"}) is None
    with pytest.raises(ValueError, match="no supported data or object_id"):
        _decode(stream, {})
    frames = [pd.DataFrame(), nested_frame([inactive, row], tuple(stream.stars[0].nested))]
    offline.frames = lambda: frames
    with closing(stream.draws(seed=0, rank=0, worker_id=0)) as draws:
        epoch, records = next(draws)
        assert epoch == 1
        assert [r["object_id"] for r in records] == ['legacy:' + row['object_id']]


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
        sdss = next(m for m in config.modalities if m["name"] == "sdss_spectra")
        config = AstroPT3Config(modalities=[{**sdss, **patch}])
    else:
        (spec if target == "stream" else spec["anchors"][0]).update(patch)
    with pytest.raises(ValueError, match=match):
        streaming.CatalogStream(spec, config)
    assert not offline.opened


def test_old_lsdb_rejected_at_config_time(recipe, offline, monkeypatch):
    def old_crossmatch_stream(left, *others, min_surveys=1, **kwargs):
        raise AssertionError("must not open a stream")

    monkeypatch.setattr(streaming, "CrossMatchStream", old_crossmatch_stream)
    with pytest.raises(ValueError, match="min_catalogs API"):
        streaming.CatalogStream(*_case(recipe))
    assert not offline.opened


@pytest.mark.parametrize("separate_flag", [False, True])
def test_redshift_requires_quality_from_same_catalog(separate_flag):
    catalogs = [{"name": "redshifts", "repo": "offline/redshifts", "decoder": "scalars",
                 "scalar_columns": {"Z": "redshift"}}]
    if separate_flag:
        catalogs.append({"name": "flags", "repo": "offline/flags", "decoder": "scalars",
                         "scalar_columns": {"ZWARN": "quality"}})
    config = AstroPT3Config(modalities=[m for m in DEFAULT_MODALITIES if m["name"] == "Z"])
    root, *partners = catalogs
    with pytest.raises(ValueError, match="ZWARN"):
        streaming.CatalogStream({"anchors": [{**root, "draw_weight": 1}], "catalogs": partners}, config)


def test_per_catalog_join_overrides_stream_join(recipe, offline):
    spec = deepcopy(recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"])
    hsc = next(c for c in spec["catalogs"] if c["name"] == "hsc")
    hsc["how"] = "left"
    config = AstroPT3Config(modalities=recipe["model"]["model_config"]["modalities"])
    stream = streaming.CatalogStream(spec, config)
    frame = nested_frame([_row(stream.spec)], tuple(stream.stars[0].nested))
    offline.frames = lambda: [frame]
    with closing(stream.draws(seed=0, rank=0, worker_id=0)) as draws:
        next(draws)
    opened = offline.streams[0]
    joins = {right["suffixes"][1]: right["how"] for right in opened.rights}
    assert joins == {"_sdss": "outer", "_desi": "outer", "_hsc": "left", "_galaxies": "outer", "_provabgs": "left"}
    # sky is selected from the anchor and outer partners only; left joins cannot widen it
    surveys = tuple(f"hf://datasets/{c['repo']}" for c in spec["catalogs"] if c.get("how", "outer") == "outer")
    assert opened.kwargs["footprint"] == ("footprint", f"hf://datasets/{spec['anchors'][0]['repo']}", surveys, 3)
    hsc["how"] = "inner"
    with pytest.raises(ValueError, match="left or outer"):
        streaming.CatalogStream(spec, config)


def test_retry_reopens_with_new_seed_and_cleans_resources(recipe, offline):
    spec, config = _case(recipe, 1)
    stream = streaming.CatalogStream(spec, config)
    frame = nested_frame([_row(spec)], tuple(stream.stars[0].nested))

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
        frame = nested_frame([row], tuple(stream.stars[0].nested))
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


def test_galaxies_with_hats_png_scalars_and_sentinels(recipe):
    spec = deepcopy(recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"])
    spec["anchors"] = [{**next(c for c in spec["catalogs"] if c["decoder"] == "rgb_png"), "draw_weight": 1}]
    spec["catalogs"] = []
    spec["min_catalogs"] = 1
    modalities = [m for m in recipe["model"]["model_config"]["modalities"] if m["source"] == "galaxies_with_hats"]
    config = AstroPT3Config(modalities=modalities)
    stream = streaming.CatalogStream(spec, config)
    star, = stream.stars
    assert star.columns[0] == "dr8_id" and "image" in star.columns and not star.nested
    row = _row(spec, index=3)
    scalars = spec["anchors"][0]["scalar_columns"]
    row[scalars["gwh_redshift"]] = 0.1
    row[scalars["gwh_mass_med_photoz"]] = -99.0  # photo-z fill value means "not measured"
    row[scalars["gwh_total_sfr_flag"]] = 0.0  # MPA-JHU flag 0 = reliable
    row[scalars["gwh_total_ssfr_flag"]] = 2.0  # any other flag drops the span
    for key in ("gwh_mag_f", "gwh_mag_n", "gwh_mag_u", "gwh_mag_i"):
        row[scalars[key]] = 18.0
    row[scalars["gwh_bar_strong_fraction"]] = 0.25
    record = _decode(stream, row)
    assert record is not None
    assert record["object_id"] == "galaxies:galaxies-3"
    image = record["galaxy_image"]
    assert image["band"] == ["rgb-r", "rgb-g", "rgb-b"]
    assert image["flux"].shape == (3, 256, 256)
    # dequantized: each value lies uniformly inside its own 8-bit bin [k, k + 1) / 255
    pixels = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")).transpose(2, 0, 1)
    offsets = image["flux"].astype(np.float64) * 255 - pixels
    assert offsets.min() > -1e-4 and offsets.max() < 1 + 1e-4  # float32 rounding at the bin top
    assert 0.45 < offsets.mean() < 0.55 and offsets.std() > 0.25
    values = ObjectSequencer(config).build(record).values
    tokens = values["galaxies"]
    assert tokens.shape == (256, 768)
    torch.testing.assert_close(antispiralise(tokens), patchify_image(torch.from_numpy(image["flux"]), 16))
    torch.testing.assert_close(values["gwh_redshift"], torch.log1p(torch.tensor([[0.1]])))
    assert values["gwh_gz_bar"].shape == (1, 2)  # one joint span per Galaxy Zoo question
    torch.testing.assert_close(values["gwh_gz_bar"], torch.tensor([[-0.5, 0.0]]))
    assert set(values) == {m["name"] for m in modalities} - {"gwh_mass_med_photoz", "gwh_total_ssfr_median"}


def test_unmatched_anchor_has_no_galaxy_modalities(recipe):
    spec = deepcopy(recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"])
    config = AstroPT3Config(modalities=recipe["model"]["model_config"]["modalities"])
    stream = streaming.CatalogStream(spec, config)
    row = _row(spec)
    row["dr8_id_galaxies"] = pd.NA
    row["image_galaxies"] = pd.NA  # no galaxy within the match radius
    galaxies = next(c for c in spec["catalogs"] if c["name"] == "galaxies")
    for column in galaxies["scalar_columns"].values():
        row[f"{column}_galaxies"] = pd.NA
    record = _decode(stream, row)
    assert record is not None
    assert record["object_id"] == "legacy:legacy-1"
    assert "galaxy_image" not in record and not any(key.startswith("gwh_") for key in record)
    assert "galaxies" not in ObjectSequencer(config).build(record).values


def test_unmatched_galaxy_keeps_its_identity_and_modalities(recipe):
    spec = recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    config = AstroPT3Config(modalities=recipe["model"]["model_config"]["modalities"])
    stream = streaming.CatalogStream(spec, config)
    record = _decode(stream, {k: v for k, v in _row(spec).items() if k.endswith("_galaxies")})
    assert record is not None
    assert record["source_ids"] == {"galaxies": "galaxies-1"}
    assert record["object_id"] == "galaxies:galaxies-1" and record["anchor"] is None
    # same galaxies modalities as when matched; the fixture's placeholder scalars fail some gates either way
    matched = ObjectSequencer(config).build(_decode(stream, _row(spec))).values
    gwh = {m["name"] for m in config.modalities if m["source"] == "galaxies_with_hats"}
    assert "galaxies" in gwh & set(matched)
    assert set(ObjectSequencer(config).build(record).values) == gwh & set(matched)


def test_project_anchor_margin_strips_payload_columns_on_disk(tmp_path, monkeypatch):
    """Left-anchor margins fetch only coordinate columns while preserving outer-join output."""
    import lsdb
    import nested_pandas as npd
    import pyarrow as pa
    from lsdb.loaders.hats import read_hats as lsdb_rh
    from lsdb.streams.catalog_streams import CrossMatchStream

    rng = np.random.default_rng(7)
    n = 120
    ra, dec = rng.uniform(10.0, 20.0, n), rng.uniform(10.0, 20.0, n)
    left_tbl = pa.table({
        "object_id": [f"L{i}" for i in range(n)],
        "ra": ra,
        "dec": dec,
        "ebv": np.linspace(0.01, 0.05, n).astype(np.float32),
        "image": pa.StructArray.from_arrays(
            [
                pa.array([["des-g", "des-r"]] * n),
                pa.array([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]] * n),
            ],
            names=["band", "flux"],
        ),
    })
    right_tbl = pa.table({
        "object_id": [f"R{i}" for i in range(n)],
        "ra": np.concatenate([ra[:90] + 1e-5, rng.uniform(10.0, 22.0, 30)]),
        "dec": np.concatenate([dec[:90] + 1e-5, rng.uniform(10.0, 22.0, 30)]),
        "z": np.linspace(0.1, 0.5, n).astype(np.float32),
    })
    lsdb.from_dataframe(
        npd.from_pyarrow(left_tbl), catalog_name="left", margin_threshold=300.0, highest_order=5, partition_rows=30
    ).write_catalog(tmp_path / "left", as_collection=True)
    lsdb.from_dataframe(
        npd.from_pyarrow(right_tbl), catalog_name="right", margin_threshold=300.0, highest_order=5, partition_rows=30
    ).write_catalog(tmp_path / "right", as_collection=True)

    left_full = lsdb.open_catalog(tmp_path / "left", columns=["object_id", "ra", "dec", "image.band", "image.flux", "ebv"])
    right_full = lsdb.open_catalog(tmp_path / "right", columns=["object_id", "ra", "dec", "z"])
    fp = left_full.hc_structure.pixel_tree.to_moc()
    ref = next(iter(CrossMatchStream(
        left_full, {"other": right_full, "how": "outer", "suffixes": ("", "_right"), "radius_arcsec": 1.0},
        footprint=fp, partitions_per_chunk=50, seed=0,
    )))

    calls: list[tuple[str, tuple[str, ...]]] = []
    orig_read_pixel = lsdb_rh.read_pixel

    def spy_read_pixel(*args, **kwargs):
        calls.append((str(kwargs["catalog_base_dir"]), tuple(kwargs.get("columns") or ())))
        return orig_read_pixel(*args, **kwargs)

    monkeypatch.setattr(lsdb_rh, "read_pixel", spy_read_pixel)
    left_proj = streaming._project_anchor_margin(
        lsdb.open_catalog(tmp_path / "left", columns=["object_id", "ra", "dec", "image.band", "image.flux", "ebv"])
    )
    right_proj = lsdb.open_catalog(tmp_path / "right", columns=["object_id", "ra", "dec", "z"])
    out = next(iter(CrossMatchStream(
        left_proj, {"other": right_proj, "how": "outer", "suffixes": ("", "_right"), "radius_arcsec": 1.0},
        footprint=fp, partitions_per_chunk=50, seed=0,
    )))
    assert ref.sort_index().to_pyarrow().equals(out.sort_index().to_pyarrow())
    margin_cols = {cols for path, cols in calls if "left_300arcs" in path}
    assert margin_cols == {("ra", "dec")}
    right_margin_cols = {cols for path, cols in calls if "right_300arcs" in path}
    assert right_margin_cols == {("object_id", "ra", "dec", "z")}
