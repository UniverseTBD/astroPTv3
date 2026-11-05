"""Offline checks for the sustained four-survey run adapter."""

import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from legacy_fixture import make_record, nested_frame

from astropt3.data import catalog_stream
from astropt3.data.echo import echo_draw
from astropt3.tokenization import BOS_ID, PAD_ID

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_three_catalogs as run  # pyright: ignore[reportMissingImports] -- scripts added above


@pytest.fixture
def default_recipe():
    return yaml.safe_load(run.DEFAULT_CONFIG.read_text())


@pytest.fixture
def default_stream(default_recipe):
    hf = run.loader.hf_config_from_modalities(
        default_recipe["model"]["model_config"]["modalities"]
    )
    spec = default_recipe["data_stages"][0]["data"]["dataset"]["catalog_stream"]
    return catalog_stream.CatalogStream(spec, hf)


def test_training_config_keeps_batch_shape_and_checkpoint_schedule(tmp_path, default_recipe):
    baseline = default_recipe
    before = copy.deepcopy(baseline)
    config = run.prepare_config(
        baseline, tmp_path / "new-run", tmp_path / "old-checkpoints"
    )
    assert baseline == before
    assert config["tokens"]["train_steps"] == 30_000
    assert config["tokens"]["micro_batch_size"] == 16
    assert config["tokens"]["sequence_length"] == 4096
    assert config["optimizer"] == baseline["optimizer"]
    assert config["checkpoints"]["checkpoint_schedule"] == "pythia"
    assert config["checkpoints"]["checkpoint_interval"] == 1000
    assert config["checkpoints"]["save_final_state"]
    assert config["checkpoints"]["resume_checkpoint_path"] == str(
        tmp_path / "old-checkpoints"
    )
    modalities = config["model"]["model_config"]["modalities"]
    assert [modality["name"] for modality in modalities] == [
        "images",
        "spectra",
        "sdss_spectra",
        "hsc_images",
    ]
    assert modalities[2]["token_ids"] == [17, 18, 19]
    assert modalities[-1]["token_ids"] == [20, 21, 22]
    assert modalities[-1]["input_size"] == 320
    assert modalities[-1]["record_keys"] == ["hsc_image"]
    assert modalities[-1]["family"] == "image"
    hf = run.loader.hf_config_from_modalities(modalities)
    ids = [token for modality in hf.modalities for token in modality["token_ids"]]
    assert len(ids) == len(set(ids))
    dataset = config["data_stages"][0]["data"]["dataset"]
    assert dataset["echo_factor"] == 4
    assert config["data_stages"][0]["data"]["num_loading_workers"] == 4
    assert dataset["catalog_stream"]["min_surveys"] == 3
    catalogs = {cat["name"]: cat["repo"] for cat in dataset["catalog_stream"]["catalogs"]}
    assert set(catalogs) == {"legacy", "sdss", "desi", "hsc"}
    assert catalogs["hsc"] == (
        "hugging-science/mmu_hsc_pdr3_wide_21@7de6b92fd99097207a00b9c42f2d2862d62662bf"
    )


@pytest.mark.parametrize("count", [1, 2, 4])
def test_prepare_config_preserves_recipe_modalities_horizon_and_workers(tmp_path, default_recipe, count):
    modalities = default_recipe["model"]["model_config"]["modalities"][:count]
    for index, modality in enumerate(modalities):
        modality["name"] = f"survey_{index}"
        modality["token_ids"] = list(range(23 + 3 * index, 26 + 3 * index))
    default_recipe["model"]["model_config"]["modalities"] = modalities
    default_recipe["tokens"]["train_steps"] = 123
    default_recipe["data_stages"][0]["data"]["num_loading_workers"] = 2
    before = copy.deepcopy(default_recipe)
    config = run.prepare_config(default_recipe, tmp_path / "custom")
    assert default_recipe == before
    assert config["model"] == before["model"]
    assert config["tokens"] == before["tokens"]
    assert config["optimizer"] == before["optimizer"]
    assert config["data_stages"][0]["data"]["num_loading_workers"] == 2


@pytest.mark.parametrize(
    "configured,override,expected",
    [(None, None, 1), (4, None, 4), (4, 2, 2), (2, None, 2), (2, 3, 3), (3, 1, 1)],
)
def test_echo_config_has_one_saved_source_of_truth(tmp_path, default_recipe, configured, override, expected):
    baseline = default_recipe
    dataset = baseline["data_stages"][0]["data"]["dataset"]
    if configured is None:
        dataset.pop("echo_factor")
    else:
        dataset["echo_factor"] = configured
    before = copy.deepcopy(baseline)
    config = run.prepare_config(baseline, tmp_path / "output", echo_factor=override)
    assert baseline == before
    restored = yaml.safe_load(yaml.safe_dump(config))
    assert restored["data_stages"][0]["data"]["dataset"]["echo_factor"] == expected
    assert run.wandb_config(restored, None)["echo_factor"] == expected


@pytest.mark.parametrize("factor", [0, -1, True, 1.5, "2"])
def test_echo_config_rejects_invalid_yaml_values(tmp_path, default_recipe, factor):
    baseline = default_recipe
    baseline["data_stages"][0]["data"]["dataset"]["echo_factor"] = factor
    with pytest.raises(ValueError, match="positive integer"):
        run.prepare_config(baseline, tmp_path / "output")


def test_wandb_config_keeps_training_settings_without_local_provenance(tmp_path, default_recipe):
    baseline = default_recipe
    config = run.prepare_config(baseline, tmp_path / "output", tmp_path / "resume")
    config["general"]["benchmark_csv_path"] = str(tmp_path / "benchmark.csv")
    config["provenance"] = {"commit": "private-commit", "source": str(tmp_path)}
    config["model"]["init_method"] = {"path": str(tmp_path / "init-checkpoint")}
    before = copy.deepcopy(config)
    payload = run.wandb_config(config, 1)
    assert config == before
    assert set(payload) == {"training", "stream", "max_steps", "echo_factor", "echo_order", "throughput_policy"}
    assert payload["echo_factor"] == 4
    configured = run.prepare_config(baseline, tmp_path / "configured", echo_factor=3)
    assert run.wandb_config(configured, 1)["echo_factor"] == 3
    assert configured["data_stages"][0]["data"]["dataset"]["echo_factor"] == 3
    assert str(tmp_path) not in json.dumps(payload)
    assert "private-commit" not in json.dumps(payload)
    assert "provenance" not in json.dumps(payload)
    assert payload["training"]["model"] == {
        key: value for key, value in config["model"].items() if key != "init_method"
    }
    assert payload["training"]["optimizer"] == config["optimizer"]
    assert payload["training"]["tokens"]["train_steps"] == 30_000
    assert payload["training"]["seed"] == config["general"]["seed"]
    assert payload["training"]["checkpoints"]["checkpoint_interval"] == 1000
    assert payload["stream"]["min_surveys"] == 3
    assert payload["stream"]["how"] == "outer"
    serialized = json.dumps(payload)
    assert "catalog_stream" not in serialized
    assert "catalogs" not in payload["stream"]
    for catalog in config["data_stages"][0]["data"]["dataset"]["catalog_stream"]["catalogs"]:
        assert catalog["repo"] not in serialized
        assert catalog["repo"].split("@")[0] not in serialized
    assert payload["max_steps"] == 1
    payload["training"]["model"]["dtype"] = "changed"
    assert config == before


@pytest.mark.parametrize("include_sdss,padding", [(False, 0), (True, 0), (True, 2)])
def test_catalog_decoder_preserves_optional_spectra(default_stream, include_sdss, padding):
    source = make_record(1, image_only_fraction=0.0)
    nested = {"image": source["image"], "spectrum_desi": source["spectrum"]}
    if include_sdss:
        nested["spectrum_sdss"] = {
            "flux": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "lambda": (4000 * 10 ** (np.arange(3) * 1e-4)).astype(np.float32),
            "mask": np.array([False, True, False]),
        }
        if padding:
            nested["spectrum_sdss"] = {
                name: np.pad(values, padding)
                for name, values in nested["spectrum_sdss"].items()
            }
    mapped = {"object_id": source["object_id"]}
    for name, fields in default_stream.nested.items():
        for field in fields:
            mapped[f"{name}.{field}"] = (
                np.asarray(nested[name][field]) if name in nested else np.array([])
            )
    decoded = default_stream.decode_row(mapped)["record"]
    assert decoded["object_id"] == source["object_id"]
    np.testing.assert_array_equal(decoded["image"]["flux"], source["image"]["flux"])
    np.testing.assert_array_equal(
        decoded["spectrum"]["flux"], source["spectrum"]["flux"]
    )
    assert ("spectrum_sdss" in decoded) == include_sdss
    if include_sdss:
        np.testing.assert_array_equal(
            decoded["spectrum_sdss"]["mask"], [False, True, False]
        )
        np.testing.assert_array_equal(decoded["spectrum_sdss"]["flux"], [1.0, 2.0, 3.0])
        from astropt3.data.spectral import spectral_normalize

        assert torch.isfinite(
            spectral_normalize(
                torch.from_numpy(decoded["spectrum_sdss"]["flux"]),
                torch.from_numpy(decoded["spectrum_sdss"]["lambda"]),
                source="sdss",
            )
        ).all()


def sdss_spectrum():
    return {
        "flux": np.ones(512, dtype=np.float32),
        "lambda": (4000 * 10 ** (np.arange(512) * 1e-4)).astype(np.float32),
        "mask": np.zeros(512, dtype=bool),
    }


def hsc_image():
    flux = np.broadcast_to(
        np.linspace(-0.2, 1.0, 160, dtype=np.float32), (5, 160, 160)
    ).copy()
    return {
        "band": [f"hsc-{band}" for band in "grizy"],
        "flux": flux,
        "scale": np.full(5, 0.168, dtype=np.float32),
    }


class OfflineCatalogBatches(run.loader.PackedMicroBatches):
    """Keep real worker serialization/packing; replace only external records."""

    def _open_draws(self, worker):
        assert worker is not None and worker.id == 0
        assert self.catalog_stream is not None
        record = make_record(1, image_only_fraction=0.0)
        record["spectrum_sdss"] = sdss_spectrum()
        record["hsc_image"] = hsc_image()
        while True:
            yield iter([record])


@pytest.mark.parametrize("factor", [1, 4])
def test_spawned_worker_includes_hsc(monkeypatch, tmp_path, default_recipe, factor):
    monkeypatch.delenv("ASTROPT3_TELEMETRY_DIR", raising=False)
    monkeypatch.setattr(run.loader, "PackedMicroBatches", OfflineCatalogBatches)
    baseline = default_recipe
    config = run.prepare_config(baseline, tmp_path / "output")
    batches = run.loader.build_astropt3_dataloader(
        dataset_args=SimpleNamespace(**{
            **config["data_stages"][0]["data"]["dataset"], "echo_factor": factor,
        }),
        model_config=SimpleNamespace(**config["model"]["model_config"]),
        micro_batch_size=1,
        sequence_length=512,
        dp_rank=0,
        dp_size=1,
        num_workers=1,
        multiprocessing_context="spawn",
    )
    iterator = iter(batches)
    try:
        batch = next(iterator)
        assert batch["input_ids"].shape == (1, 512)
        assert batch["hsc_images_values"].shape == (144, 320)
        assert int((batch["input_ids"] == 21).sum()) == 144
        assert all(
            torch.isfinite(value).all()
            for key, value in batch.items()
            if key.endswith("_values")
        )
        assert all(isinstance(value, torch.Tensor) for value in batch.values())
        if factor > 1:
            repeated = next(iterator)
            assert batches.echo_index == 1
            torch.testing.assert_close(repeated["hsc_images_values"], batch["hsc_images_values"])
    finally:
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        else:
            iterator.close()


@pytest.mark.parametrize("source", ["sdss", "desi", "hsc"])
@pytest.mark.parametrize("missing", [None, pd.NA, np.nan])
def test_decoder_keeps_right_only_rows(default_stream, source, missing):
    if source == "hsc":
        payload, column, key = hsc_image(), "image_hsc", "hsc_image"
    else:
        payload = (
            sdss_spectrum()
            if source == "sdss"
            else make_record(1, image_only_fraction=0.0)["spectrum"]
        )
        column = f"spectrum_{source}"
        key = "spectrum_sdss" if source == "sdss" else "spectrum"
    mapped = {
        "object_id": missing,
        "object_id_sdss": missing,
        "object_id_desi": missing,
    }
    mapped[f"object_id_{source}"] = "same-id"
    for name, fields in default_stream.nested.items():
        for field in fields:
            mapped[f"{name}.{field}"] = (
                np.asarray(payload[field]) if name == column else np.array([])
            )
    record = default_stream.decode_row(mapped)["record"]
    assert set(record) == {"object_id", key}
    assert record["object_id"] == f"{source}:same-id"
    np.testing.assert_array_equal(record[key]["flux"], payload["flux"])
    mapped[f"object_id_{source}"] = missing
    with pytest.raises(ValueError, match="object_id"):
        default_stream.decode_row(mapped)


def test_stream_retains_matched_and_unpaired_records(monkeypatch, default_recipe, default_stream):
    source = make_record(1, image_only_fraction=0.0)
    image = {
        field: np.asarray(source["image"][field]).tolist()
        for field in default_stream.nested["image"]
    }
    rows = [
        {"object_id": "image-only", "image": image},
        {
            "object_id": "matched",
            "image": image,
            "object_id_sdss": "matched-sdss",
            "spectrum_sdss": sdss_spectrum(),
            "object_id_desi": "matched-desi",
            "spectrum_desi": source["spectrum"],
        },
        {"object_id_sdss": "same-id", "spectrum_sdss": sdss_spectrum()},
        {"object_id_desi": "same-id", "spectrum_desi": source["spectrum"]},
    ]
    hsc = {field: np.asarray(values).tolist() for field, values in hsc_image().items()}
    rows[1].update(object_id_hsc="matched-hsc", image_hsc=hsc)
    rows.append({"object_id_hsc": "same-id", "image_hsc": hsc})
    frame = nested_frame(rows, tuple(default_stream.nested))
    catalogs = {cat["name"]: cat["repo"] for cat in default_stream.catalogs}

    class FakeStream:
        npartitions = 1

        def __init__(self, anchor, *rights, min_surveys, **kwargs):
            assert anchor == f"hf://datasets/{catalogs['legacy']}"
            names = ["sdss", "desi", "hsc"]
            assert [right["other"] for right in rights] == [
                f"hf://datasets/{catalogs[name]}" for name in names
            ]
            assert [right["how"] for right in rights] == ["outer"] * len(names)
            assert [right["suffixes"] for right in rights] == [
                ("", f"_{name}") for name in names
            ]
            assert all(right["radius_arcsec"] == 1.0 for right in rights)
            assert min_surveys == 3
            assert kwargs["count_fraction_threshold"] == 0.0

        def __iter__(self):
            yield frame

    def open_catalog(url, *, columns):
        is_image = "hsc" in url or "legacysurvey" in url
        assert columns == [
            "object_id",
            "ra",
            "dec",
            "image" if is_image else "spectrum",
        ]
        return url

    monkeypatch.setattr(catalog_stream, "open_catalog", open_catalog)
    monkeypatch.setattr(catalog_stream, "CrossMatchStream", FakeStream)
    monkeypatch.setattr(catalog_stream, "LocalCluster", lambda **kwargs: nullcontext())
    monkeypatch.setattr(catalog_stream, "Client", lambda cluster: nullcontext())
    hf_config = run.loader.hf_config_from_modalities(
        default_recipe["model"]["model_config"]["modalities"]
    )
    dataset = run.loader.PackedMicroBatches(
        hf_config, micro_batch_size=1, seq_len=4096, catalog_stream=default_stream.spec,
    )
    iterator = dataset._open_records(None)
    try:
        records = list(islice(iterator, len(rows)))
    finally:
        iterator.close()
    assert [record["object_id"] for record in records] == [
        "image-only",
        "matched",
        "sdss:same-id",
        "desi:same-id",
        "hsc:same-id",
    ]
    sequences = [dataset.sequencer.build(record) for record in records]
    assert [set(sequence.values) for sequence in sequences] == [
        {"images"},
        {"images", "sdss_spectra", "spectra", "hsc_images"},
        {"sdss_spectra"},
        {"spectra"},
        {"hsc_images"},
    ]
    assert sequences[-1].values["hsc_images"].shape == (144, 320)
    from astropt3.tokenization import antispiralise, patchify_image

    raw = torch.from_numpy(hsc_image()["flux"][:, 32:128, 32:128])
    factor = 10 ** ((22.5 - 27.0) / 2.5) * (0.262 / 0.168) ** 2
    expected = patchify_image(torch.asinh(raw * factor / 0.01), 8)
    torch.testing.assert_close(
        antispiralise(sequences[-1].values["hsc_images"]), expected
    )
    batch = dataset.collator(sequences)
    flat = run.loader.flatten_packed_batch(batch, hf_config, 4096)
    assert flat["hsc_images_values"].shape == (288, 320)
    assert int(flat["hsc_images_mask"].sum()) == 288
    assert all(
        batch["modality_masks"][name].any()
        for name in hf_config.modality_registry().names()
    )


@pytest.mark.parametrize(
    "bad_field", ["shape", "bands", "scale", "missing_scale", "nan", "inf"]
)
def test_decoder_rejects_invalid_hsc_payloads(default_stream, bad_field):
    image = hsc_image()
    if bad_field == "shape":
        image["flux"] = image["flux"][:, :-1, :]
    elif bad_field == "bands":
        image["band"] = image["band"][::-1]
    elif bad_field == "scale":
        image["scale"][0] = 0.262
    elif bad_field == "missing_scale":
        del image["scale"]
    else:
        image["flux"][0, 80, 80] = float(bad_field)
    mapped: dict = {"object_id_hsc": "hsc-only"}
    mapped.update(
        {
            f"image_hsc.{field}": np.asarray(image.get(field))
            for field in default_stream.nested["image_hsc"]
        }
    )
    with pytest.raises(ValueError, match="HSC Wide"):
        default_stream.decode_row(mapped)


@pytest.mark.parametrize("hsc_only", [True, False])
def test_hsc_head_forward_backward(default_recipe, tiny_config, hsc_only):
    from astropt3 import AstroPT3Config, AstroPT3Model
    from astropt3.data.packing import ObjectSequencer, PackedCollator

    hf_config = AstroPT3Config(
        **{
            **tiny_config.to_dict(),
            "modalities": default_recipe["model"]["model_config"]["modalities"],
            "loss_aggregation": "family",
        }
    )
    record = {"object_id": "hsc-only", "hsc_image": hsc_image()}
    if not hsc_only:
        record.update(make_record(1, image_only_fraction=0.0))
        record["spectrum_sdss"] = sdss_spectrum()
    sequence = ObjectSequencer(hf_config).build(record)
    batch = PackedCollator(hf_config, seq_len=512)([sequence])
    model = AstroPT3Model(hf_config).train()
    out = model(**batch)
    expected_names = (
        {"hsc_images"}
        if hsc_only
        else {"images", "spectra", "sdss_spectra", "hsc_images"}
    )
    assert set(out.modality_losses) == expected_names
    assert torch.isfinite(out.loss)
    torch.testing.assert_close(
        out.loss, torch.stack(list(out.modality_losses.values())).mean()
    )
    out.loss.backward()
    for module in (model.flows["hsc_images"], model.decoders["hsc_images"]):
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        )


@pytest.mark.parametrize("flag", ["--hsc-wide", "--no-hsc-wide"])
def test_python_cli_rejects_removed_hsc_options(monkeypatch, tmp_path, flag):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_three_catalogs.py", "--output", str(tmp_path / "output"), flag],
    )
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def write_resume(path, model_config, *, echo_factor=None, step=64):
    checkpoint = path / str(step)
    checkpoint.mkdir(parents=True)
    (path / "latest.txt").write_text(f"{step}\n")
    (checkpoint / "model_config.json").write_text(json.dumps(model_config))
    dataset = {} if echo_factor is None else {"echo_factor": echo_factor}
    (checkpoint / "config.yaml").write_text(yaml.safe_dump({
        "data_stages": [{"data": {"dataset": dataset}}],
    }))


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("factor", [None, 4])
def test_resume_compares_normalized_modalities(tmp_path, default_recipe, count, factor):
    model = default_recipe["model"]["model_config"]
    model["modalities"] = model["modalities"][:count]
    model["modalities"][0].update(name="legacy_patches", token_ids=[23, 24, 25])
    saved = copy.deepcopy(model)
    registry = run.loader.hf_config_from_modalities(model["modalities"]).modality_registry()
    saved["modalities"] = list(reversed(registry.to_list()))
    write_resume(tmp_path, saved, echo_factor=factor)
    assert run.validate_resume(tmp_path, model) == (1 if factor is None else factor)


@pytest.mark.parametrize("field,value", [
    ("missing", None), ("name", "renamed"), ("token_ids", [23, 24, 25]),
    ("source", "different-survey"), ("input_size", 384), ("pos_input_size", 2),
])
def test_resume_rejects_generic_modality_mismatches(tmp_path, default_recipe, field, value):
    model = default_recipe["model"]["model_config"]
    saved = copy.deepcopy(model)
    if field == "missing":
        saved["modalities"].pop(0)
    else:
        saved["modalities"][2][field] = value
    write_resume(tmp_path, saved)
    with pytest.raises(ValueError, match="Checkpoint modalities differ"):
        run.validate_resume(tmp_path, model)


def test_decoder_rejects_rows_without_modalities(default_stream):
    mapped = {
        f"{name}.{field}": np.array([])
        for name, fields in default_stream.nested.items()
        for field in fields
    }
    with pytest.raises(ValueError, match="no supported data or object_id"):
        default_stream.decode_row(mapped)


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["-1"],
        ["0,1"],
        ["invalid"],
        ["1", "extra"],
        ["1", ""],
        ["1", "--hsc-wide"],
        ["1", "--no-hsc-wide"],
        ["1", "0"],
        ["1", "-2"],
        ["1", "1.5"],
        ["1", "2", "3"],
        ["1", "--config-file"],
        ["1", "--config-file", ""],
        ["1", "--config-file", "recipe.yaml", "2"],
    ],
)
def test_launcher_rejects_invalid_gpu_arguments(args):
    result = subprocess.run(
        ["bash", str(run.REPO / "astro/scripts/run_three_modal.bash"), *args],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "GPU_INDEX" in result.stderr


@pytest.mark.parametrize("echo_arg", [None, "2", "03"])
@pytest.mark.parametrize("custom_config", [False, True])
def test_launcher_selects_gpu_and_starts_full_run_without_offline_flags(tmp_path, echo_arg, custom_config):
    repo = tmp_path / "repo"
    script = repo / "astro/scripts/run_three_modal.bash"
    script.parent.mkdir(parents=True)
    script.write_text((run.REPO / "astro/scripts/run_three_modal.bash").read_text())
    config_file = repo / "astro/configs/nanotron" / run.DEFAULT_CONFIG.name
    config_file.parent.mkdir(parents=True)
    config_file.write_text(run.DEFAULT_CONFIG.read_text())
    config_args = []
    if custom_config:
        config_file = tmp_path / "recipes/custom two-modal.yaml"
        config_file.parent.mkdir()
        config_file.write_text(run.DEFAULT_CONFIG.read_text())
        config_args = ["--config-file", str(config_file.relative_to(tmp_path))]
    result = subprocess.run(["bash", str(script), "3"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Requires" in result.stderr

    python = repo / ".venv-astropt3/bin/python"
    python.parent.mkdir(parents=True)
    # Use a harmless installed binary; temporary filesystems may be mounted noexec.
    echo = shutil.which("echo")
    assert echo is not None
    python.symlink_to(echo)
    lsdb_src = tmp_path / "lsdb-pr-1584/src"
    module = lsdb_src / "lsdb/streams/catalog_streams.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    result = subprocess.run(
        ["bash", "-x", str(script), "3", *([echo_arg] if echo_arg else []), *config_args],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "NANOTRON_BENCHMARK": "1",
            "PYTHONPATH": "existing",
        },
    )
    lines = result.stdout.splitlines()
    assert lines[0] == "GPU: 3"
    output = lines[1].removeprefix("Output: ")
    trace = next(
        shlex.split(line[2:])
        for line in result.stderr.splitlines()
        if line.startswith("+ exec env ")
    )
    assert trace[:8] == [
        "exec",
        "env",
        "-u",
        "HF_HUB_OFFLINE",
        "-u",
        "HF_DATASETS_OFFLINE",
        "-u",
        "NANOTRON_BENCHMARK",
    ]
    assert "CUDA_VISIBLE_DEVICES=3" in trace
    assert f"PYTHONPATH={lsdb_src}:existing" in trace
    assert trace[trace.index(str(python)) + 1 :] == [
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=1",
        "--rdzv-backend=c10d",
        "--rdzv-endpoint=127.0.0.1:0",
        "--max-restarts=0",
        "astro/scripts/train_three_catalogs.py",
        "--config-file",
        str(config_file.resolve()),
        "--output",
        output,
        *(["--echo-factor", echo_arg] if echo_arg else []),
    ]
    assert Path(output).parent == tmp_path / "astroPTv3_telemetry"
    assert Path(output).name.startswith(f"{config_file.stem}-echo{echo_arg or 'config'}-")
    assert not Path(output).exists()


def echo_record(record, index, factor):
    """Make a hand-counted step record for the wall-time aggregation tests."""
    result = {**record, "echo_index": index, "echo_factor": factor}
    for prefix, include in (("fresh", index == 0), ("replayed", index != 0)):
        for key in ("micro_batches", "tokens_nonpad", "rows"):
            result[f"{prefix}_{key}"] = record[key] if include else 0
        for key in ("loss_tokens", "target_values"):
            result[f"{prefix}_{key}"] = {
                name: count if include else 0 for name, count in record[key].items()
            }
    return result


def echo_loader(batches, config, *, echo_factor=1, seed=42):
    return run.telemetry.TelemetryLoader(
        echo_draw(batches, config, echo_factor=echo_factor, seed=seed), enabled=False,
    )


def test_throughput_uses_wall_time_weighted_counts_and_excludes_warmup(default_recipe):
    meter = run.Throughput(
        [modality["name"] for modality in default_recipe["model"]["model_config"]["modalities"]],
        default_recipe["tokens"]["train_steps"],
    )
    record = {
        "step_seconds": 10.0,
        "tokens_nonpad": 100,
        "rows": 2,
        "loader_wait_s": 6.0,
        "utilisation_packing": 0.5,
        "loss_tokens": {"images": 70, "spectra": 20, "hsc_images": 10},
        "target_values": {},
        "micro_batches": 1,
    }
    record = echo_record(record, 0, 1)
    warmup = meter.observe(10, record)
    assert warmup["throughput/warmup"]
    assert "throughput/cumulative_nonpad_tokens_per_s" not in warmup
    first = meter.observe(11, record)
    second = meter.observe(
        12, {**record, "step_seconds": 2.0, "loader_wait_s": 1.0, "tokens_nonpad": 300}
    )
    assert first["data/stall_fraction"] == 0.6
    assert second["throughput/cumulative_nonpad_tokens_per_s"] == pytest.approx(
        400 / 12
    )
    assert second["throughput/rolling_nonpad_tokens_per_s"] == pytest.approx(400 / 12)
    assert second["throughput/cumulative_rows_per_s"] == pytest.approx(4 / 12)
    assert second["data/sdss_spectra_loss_tokens"] == 0
    assert second["data/hsc_images_loss_tokens"] == 10
    assert second["data/hsc_images_token_fraction"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="duration"):
        meter.observe(13, {**record, "step_seconds": 0.0})


@pytest.fixture
def echo_batch(default_recipe):
    """Multiple objects per row, unequal spans, four surveys and absent modalities."""
    from astropt3.data.packing import ObjectSequencer, PackedCollator

    hf = run.loader.hf_config_from_modalities(default_recipe["model"]["model_config"]["modalities"])
    full = make_record(1, image_only_fraction=0.0)
    full.update(spectrum_sdss=sdss_spectrum(), hsc_image=hsc_image())
    records = [
        full,
        {"object_id": "image-only", "image": full["image"]},
        {"object_id": "spectrum-only", "spectrum": full["spectrum"]},
        {**full, "object_id": "another-full"},
    ]
    objects = [ObjectSequencer(hf).build(record) for record in records]
    packed = PackedCollator(hf, seq_len=1024).collate_rows(
        [objects[:2], objects[2:], []]
    )
    return hf, run.loader.flatten_packed_batch(packed, hf, 1024)


@pytest.mark.parametrize("factor", [0, -1])
def test_echo_rejects_nonpositive_factors(echo_batch, factor):
    config, batch = echo_batch
    with pytest.raises(ValueError, match="echo-factor"):
        list(echo_loader([batch], config, echo_factor=factor))


def test_echo_shuffle_preserves_objects_patch_alignment_and_rng(echo_batch):
    import random

    config, batch = echo_batch
    before = {key: value.clone() for key, value in batch.items()}
    rng = random.getstate(), torch.get_rng_state()
    echoes = list(echo_loader([batch], config, echo_factor=3, seed=42))
    repeated = list(echo_loader([batch], config, echo_factor=3, seed=42))
    assert random.getstate() == rng[0]
    assert torch.equal(torch.get_rng_state(), rng[1])
    changed = False
    for echoed, duplicate in zip(echoes, repeated, strict=True):
        assert torch.equal(echoed["input_ids"], duplicate["input_ids"])
        for name in config.modality_registry().names():
            begin, placeholder, end = config.modality_registry().get_config(name).token_ids
            assert torch.equal(echoed[f"{name}_mask"], echoed["input_ids"] == placeholder)
            assert echoed[f"{name}_values"] is batch[f"{name}_values"]
            assert echoed[f"{name}_positions"] is batch[f"{name}_positions"]
        assert echoed["position_ids"] is batch["position_ids"]
        for old_row, new_row in zip(batch["input_ids"].tolist(), echoed["input_ids"].tolist(), strict=True):
            starts = [i for i, token in enumerate(old_row) if token == BOS_ID]
            ends = starts[1:] + [old_row.index(PAD_ID)] if starts else []
            for start, end in zip(starts, ends, strict=True):
                old_spans, new_spans = {}, {}
                for name in config.modality_registry().names():
                    begin, _, finish = config.modality_registry().get_config(name).token_ids
                    for row, spans in ((old_row, old_spans), (new_row, new_spans)):
                        if begin in row[start:end]:
                            a, b = row.index(begin, start, end), row.index(finish, start, end)
                            spans[name] = row[a:b + 1]
                assert old_spans == new_spans
                assert new_row[start] == BOS_ID
                if len(old_spans) == 1:
                    assert old_row[start:end] == new_row[start:end]
                changed |= old_row[start:end] != new_row[start:end]
            assert new_row[len(new_row) - new_row.count(PAD_ID):] == [PAD_ID] * new_row.count(PAD_ID)
    assert changed
    for key in before:
        assert torch.equal(batch[key], before[key])


@pytest.mark.parametrize("factor", [2, 3])
def test_echo_runs_independent_model_optimizer_updates(echo_batch, tiny_config, factor):
    from astropt3 import AstroPT3Config, AstroPT3Model

    config, batch = echo_batch
    model = AstroPT3Model(AstroPT3Config(**{
        **tiny_config.to_dict(), "modalities": config.modalities,
    })).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    snapshots = []
    original = {key: value.clone() for key, value in batch.items()}
    for echoed in echo_loader([batch], config, echo_factor=factor):
        # CPU counterpart of nanotron's non-mutating top-level device transfer.
        moved = {key: value.to("cpu", copy=True, memory_format=torch.contiguous_format,
                               non_blocking=True) for key, value in echoed.items()}
        optimizer.zero_grad()
        result = model(**run.loader.regroup_micro_batch(moved, config.modality_registry().names()))
        assert torch.isfinite(result.loss)
        result.loss.backward()
        optimizer.step()
        snapshots.append(next(model.parameters()).detach().clone())
    assert len(snapshots) == factor
    assert all(not torch.equal(a, b) for a, b in zip(snapshots[:-1], snapshots[1:], strict=True))
    for key in batch:
        assert torch.equal(batch[key], original[key])


@pytest.mark.gpu
def test_pinned_echo_cache_survives_nanotron_device_transfer(echo_batch):
    from nanotron.data.dataloader import sanity_check_dataloader

    config, batch = echo_batch
    pinned = {key: value.pin_memory() for key, value in batch.items()}
    before = {key: value.clone() for key, value in pinned.items()}
    echoed = echo_loader([pinned], config, echo_factor=3)
    for cpu_batch in echoed:
        assert all(value.is_pinned() for value in cpu_batch.values())
        transferred = next(sanity_check_dataloader(
            iter([cpu_batch]), None,  # pyright: ignore[reportArgumentType] -- no process groups used with checks disabled
            SimpleNamespace(general=SimpleNamespace(ignore_sanity_checks=True)),  # pyright: ignore[reportArgumentType] -- only this config field is read
        ))
        torch.cuda.synchronize()
        for key, value in transferred.items():
            assert isinstance(value, torch.Tensor)
            assert value.device.type == "cuda"
            torch.testing.assert_close(value.cpu(), cpu_batch[key])
        transferred_ids = transferred["input_ids"]
        assert isinstance(transferred_ids, torch.Tensor)
        transferred_ids.zero_()
        for key in before:
            assert torch.equal(pinned[key], before[key])


def test_echo_telemetry_counts_and_wait_attribution(echo_batch, monkeypatch, tmp_path):
    config, batch = echo_batch
    clock = [0.0]
    monkeypatch.setenv(run.telemetry.TELEMETRY_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(run.telemetry.time, "perf_counter", lambda: clock[0])

    def source():
        for _ in range(2):
            clock[0] += 5.0
            yield from echo_draw([batch], config, echo_factor=3, seed=42)

    wrapped = run.telemetry.instrument(source(), echo_factor=3)
    assert isinstance(wrapped, run.telemetry.TelemetryLoader)
    run.telemetry.drain_step()
    records = []
    for _ in wrapped:
        record = run.telemetry.drain_step()
        records.append(record)
        run.telemetry.write_step(len(records), record)
    assert [r["loader_wait_s"] for r in records] == [5, 0, 0, 5, 0, 0]
    assert [r["echo_index"] for r in records] == [0, 1, 2, 0, 1, 2]
    assert sum(r["fresh_micro_batches"] for r in records) == 2
    assert sum(r["replayed_micro_batches"] for r in records) == 4
    for record in records:
        for key in ("rows", "tokens_nonpad", "micro_batches"):
            assert record[key] == record[f"fresh_{key}"] + record[f"replayed_{key}"]
        for key in ("loss_tokens", "target_values"):
            for name, count in record[key].items():
                assert count == record[f"fresh_{key}"].get(name, 0) + record[f"replayed_{key}"].get(name, 0)
    assert sum(r["fresh_rows"] for r in records) == 8
    assert sum(r["replayed_rows"] for r in records) == 16
    saved = [json.loads(line) for line in (tmp_path / "steps.rank0.jsonl").read_text().splitlines()]
    assert [r["echo_index"] for r in saved] == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize("start_step", [0, 100])
def test_fresh_throughput_excludes_warmup_even_across_echo_boundary(start_step):
    meter = run.Throughput(["images"], train_steps=123, start_step=start_step)
    base = {"micro_batches": 1, "tokens_nonpad": 100, "rows": 2,
            "step_seconds": 10.0, "loader_wait_s": 8.0,
            "utilisation_packing": 0.8, "loss_tokens": {"images": 90}, "target_values": {"images": 90 * 192}}
    meter.observe(start_step + 10, echo_record(base, 0, 2))
    first = meter.observe(start_step + 11, echo_record(
        {**base, "step_seconds": 2.0, "loader_wait_s": 0.0}, 1, 2))
    assert first["throughput/cumulative_fresh_nonpad_tokens_per_s"] == 0
    assert first["data/fresh_images_loss_tokens"] == 0
    second = meter.observe(start_step + 12, echo_record(base, 0, 2))
    assert second["throughput/cumulative_nonpad_tokens_per_s"] == pytest.approx(200 / 12)
    assert second["throughput/cumulative_fresh_nonpad_tokens_per_s"] == pytest.approx(100 / 12)
    assert second["throughput/cumulative_fresh_rows_per_s"] == pytest.approx(2 / 12)
    assert second["throughput/cumulative_rows_per_s"] == pytest.approx(4 / 12)
    assert second["throughput/eta_hours"] == pytest.approx((123 - start_step - 12) * 6 / 3600)


@pytest.mark.parametrize("names", [("legacy_patches",), ("legacy_patches", "desi_flux")])
def test_throughput_uses_configured_names_and_horizon(names):
    meter = run.Throughput(names, train_steps=12)
    record = echo_record({
        "step_seconds": 2.0, "tokens_nonpad": 100, "rows": 1,
        "loader_wait_s": 0.0, "utilisation_packing": 1.0, "micro_batches": 1,
        "loss_tokens": {names[0]: 90}, "target_values": {},
    }, 0, 4)
    metrics = meter.observe(11, record)
    assert {key for key in metrics if key.endswith("_loss_tokens")} == {
        f"data/{prefix}{name}_loss_tokens"
        for name in names for prefix in ("", "fresh_", "replayed_")
    }
    assert metrics[f"data/{names[0]}_loss_tokens"] == 90
    assert metrics[f"data/{names[0]}_token_fraction"] == 1.0
    if len(names) > 1:
        assert metrics[f"data/{names[1]}_loss_tokens"] == 0
        assert metrics[f"data/{names[1]}_token_fraction"] == 0.0
    assert metrics["throughput/eta_hours"] == pytest.approx(2 / 3600)
    assert meter.observe(12, record)["throughput/eta_hours"] == 0
    assert meter.observe(13, record)["throughput/eta_hours"] == 0


@pytest.mark.parametrize("factor", ["0", "-1", "1.5", "abc"])
def test_python_cli_rejects_bad_echo_before_creating_output(monkeypatch, tmp_path, factor):
    monkeypatch.setenv("WORLD_SIZE", "1")
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["train_three_catalogs.py", "--output", str(output), "--echo-factor", factor])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("custom_config", [False, True])
def test_main_wires_echo_accounting_and_restarts_mid_group(monkeypatch, tmp_path, default_recipe, custom_config):
    source_record = make_record(1, image_only_fraction=0.0)
    source_record.update(spectrum_sdss=sdss_spectrum(), hsc_image=hsc_image())
    config_args = []
    if custom_config:
        modalities = default_recipe["model"]["model_config"]["modalities"][:2]
        for index, name in enumerate(("legacy_patches", "desi_flux")):
            modalities[index].update(name=name, token_ids=list(range(23 + 3 * index, 26 + 3 * index)))
        default_recipe["model"]["model_config"]["modalities"] = modalities
        default_recipe["tokens"]["train_steps"] = 123
        default_recipe["data_stages"][0]["data"]["num_loading_workers"] = 0
        config_file = tmp_path / "custom.yaml"
        config_file.write_text(yaml.safe_dump(default_recipe))
        config_args = ["--config-file", str(config_file)]
    names = [modality["name"] for modality in default_recipe["model"]["model_config"]["modalities"]]
    resume = tmp_path / "checkpoints"
    write_resume(resume, default_recipe["model"]["model_config"], echo_factor=3, step=2)

    class FakeTrainer:
        def __init__(self, effective, benchmark):
            config = yaml.safe_load(Path(effective).read_text())
            assert config["tokens"] == default_recipe["tokens"]
            assert config["data_stages"][0]["data"]["num_loading_workers"] == (
                default_recipe["data_stages"][0]["data"]["num_loading_workers"]
            )
            self.init_checkpoint_path = config["checkpoints"]["resume_checkpoint_path"]
            self.metadata = SimpleNamespace(last_train_step=2 if self.init_checkpoint_path else 0)
            self.parallel_context = None
            self.random_states = {}
            self.num_params = {"total": 1}
            self.dataset_args = SimpleNamespace(**config["data_stages"][0]["data"]["dataset"])
            self.model_config = SimpleNamespace(**config["model"]["model_config"])
            self.config = SimpleNamespace(save_as_yaml=lambda path: path.write_text(yaml.safe_dump(config)))

        def train(self, dataloaders, max_steps):
            self.current_base_dl = dataloaders["stage"]
            iterator = iter(self.current_base_dl)
            for _ in range(max_steps):
                next(iterator)
                self.metadata.last_train_step += 1
                record = {**run.telemetry.drain_step(), "step_seconds": 1.0}
                run.telemetry.write_step(self.metadata.last_train_step, record)
            iterator.close()

    def run_attempt(attempt, override, expected_factor):
        output = tmp_path / f"run-{attempt}"
        consumed, logged, payloads = [], [], []

        class Source(run.loader.PackedMicroBatches):
            def _open_draws(self, worker):
                assert self.catalog_stream is not None
                consumed.append("fresh")
                yield iter([source_record])
                raise AssertionError("fetched before the echo group ended")

        def get_dataloader(trainer):
            return {"stage": run.loader.build_astropt3_dataloader(
                trainer.dataset_args, trainer.model_config, 1, 512, 0, 1,
                num_workers=0, multiprocessing_context=None,
            )}

        tracker = SimpleNamespace(
            url="https://wandb.ai/test/echo", id="test", summary={},
            define_metric=lambda *args, **kwargs: None,
            log=lambda metrics, step: logged.append((step, metrics)),
        )

        def init(**kwargs):
            payloads.append(kwargs["config"])
            return nullcontext(tracker)

        with monkeypatch.context() as patch:
            for name, attributes in {
                "run_train": {"DistributedTrainer": FakeTrainer, "get_dataloader": get_dataloader},
                "benchmark_throughput": {"revision": lambda path: {}},
                "wandb": {"init": init, "Settings": lambda **kwargs: kwargs},
                "nanotron.random": {"RandomState": dict, "RandomStates": dict},
                "nanotron.serialize.random": {"load_random_states": lambda *args: {}},
            }.items():
                module = ModuleType(name)
                vars(module).update(attributes)
                patch.setitem(sys.modules, name, module)
            patch.setattr(run.loader, "PackedMicroBatches", Source)
            patch.setattr(run.telemetry, "write_step", run.telemetry.write_step)
            patch.setattr(run.torch, "set_num_threads", lambda _: None)
            patch.setattr(run.torch.cuda, "memory_allocated", lambda: 0)
            patch.setattr(run.torch.cuda, "max_memory_allocated", lambda: 0)
            patch.setattr(run.torch.distributed, "is_initialized", lambda: False)
            patch.setenv("WORLD_SIZE", "1")
            patch.delenv("NANOTRON_BENCHMARK", raising=False)
            patch.delenv("ASTROPT3_TELEMETRY_DIR", raising=False)
            patch.setattr(sys, "argv", ["train_three_catalogs.py", "--output", str(output),
                                       "--max-steps", "2", *config_args,
                                       *(["--echo-factor", str(override)] if override else []),
                                       *(["--resume-from", str(resume)] if attempt else [])])
            run.main()
        records = [json.loads(line) for line in (output / "telemetry/steps.rank0.jsonl").read_text().splitlines()]
        assert consumed == ["fresh"]
        assert [record["echo_index"] for record in records] == [0, 1]
        assert [record["fresh_micro_batches"] for record in records] == [1, 0]
        assert [step for step, _ in logged] == ([3, 4] if attempt else [1, 2])
        assert [metrics["data/echo_index"] for _, metrics in logged] == [0, 1]
        assert payloads[0]["echo_factor"] == expected_factor
        assert "catalog_stream" not in json.dumps(payloads[0])
        assert {key for key in logged[0][1] if key.endswith("_token_fraction")} == {
            f"data/{name}_token_fraction" for name in names
        }
        state = json.loads((output / "run.json").read_text())
        assert state["echo_factor"] == expected_factor
        assert state["target_steps"] == default_recipe["tokens"]["train_steps"]
        for filename in ("effective.yaml", "resolved.yaml"):
            effective = yaml.safe_load((output / filename).read_text())
            assert effective["data_stages"][0]["data"]["dataset"]["echo_factor"] == expected_factor
        assert state["status"] == "startup-check-passed"
        if attempt:
            assert state["rng_restored"]

    run_attempt(0, None, 4)
    run_attempt(1, None, 3)
    run_attempt(2, 2, 2)


@pytest.mark.parametrize("factor", [None, "3"])
def test_python_cli_forwards_echo_and_checks_accumulation_before_outputs(monkeypatch, tmp_path, capsys, factor):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("NANOTRON_BENCHMARK", raising=False)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["train_three_catalogs.py", "--output", str(output),
                                     *(["--echo-factor", factor] if factor else [])])
    expected = int(factor) if factor else None
    original = run.prepare_config

    def prepare(baseline, output, resume, echo_factor):
        assert echo_factor == expected
        baseline["tokens"]["batch_accumulation_per_replica"] = 2
        return original(baseline, output, resume, echo_factor)

    monkeypatch.setattr(run, "prepare_config", prepare)
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert "requires batch_accumulation_per_replica=1" in capsys.readouterr().err
    assert not output.exists()


def test_python_cli_checks_max_steps_against_selected_yaml(monkeypatch, tmp_path, default_recipe, capsys):
    default_recipe["tokens"]["train_steps"] = 123
    config_file = tmp_path / "short-run.yaml"
    config_file.write_text(yaml.safe_dump(default_recipe))
    output = tmp_path / "output"
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("NANOTRON_BENCHMARK", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "train_three_catalogs.py", "--config-file", str(config_file),
        "--output", str(output), "--max-steps", "124",
    ])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert "max-steps must be in 1..123" in capsys.readouterr().err
    assert not output.exists()
