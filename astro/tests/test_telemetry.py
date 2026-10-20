"""ADR 0014 §3 instrumentation: byte probe, step counters, loader wrapper.

Source attribution covers LegacySurvey and DESI, including duplicate fetches.
"""

import json

import pytest
import torch

from astropt3.data import telemetry
from astropt3.data.nanotron_loader import PackedMicroBatches
from astropt3.tokenization import BOS_ID, PAD_ID

MBS, SEQ_LEN = 2, 512


@pytest.fixture(autouse=True)
def _reset_counters(monkeypatch):
    from huggingface_hub.hf_file_system import HfFileSystemFile, HfFileSystemStreamFile

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(telemetry, "_probe_pid", None)
    monkeypatch.setattr(HfFileSystemFile, "_fetch_range", HfFileSystemFile._fetch_range)
    monkeypatch.setattr(HfFileSystemStreamFile, "read", HfFileSystemStreamFile.read)
    telemetry.drain_step()
    yield
    telemetry.drain_step()


def test_telemetry_is_off_without_the_env_var(monkeypatch):
    monkeypatch.delenv(telemetry.TELEMETRY_DIR_ENV, raising=False)
    assert telemetry.telemetry_dir() is None
    loader = object()
    assert telemetry.instrument(loader) is loader  # untouched, zero cost


def test_source_attribution_covers_the_legacy_anchor():
    anchor = "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north/x.parquet"
    assert telemetry.source_of_path(anchor) == "legacy"
    assert telemetry.source_of_path("datasets/UniverseTBD/mmu_desi_edr_sv3/y.parquet") == "desi"
    assert telemetry.source_of_path("s3://somewhere/else.parquet") == "unknown"


def test_step_counters_match_a_hand_computed_micro_batch():
    input_ids = torch.tensor([[1, 3, 3, PAD_ID], [1, 3, PAD_ID, PAD_ID]])
    flat = {
        "input_ids": input_ids,
        "images_mask": torch.tensor(
            [[False, True, True, False], [False, True, False, False]]
        ),
        "images_values": torch.zeros((3, 192)),
        "spectra_mask": torch.zeros((2, 4), dtype=torch.bool),
        "spectra_values": torch.empty((0, 256)),
    }
    telemetry.observe_micro_batch(flat, wait_s=0.25)
    record = telemetry.drain_step()

    assert record["micro_batches"] == 1
    assert record["loader_wait_s"] == pytest.approx(0.25)
    assert record["tokens_total"] == 8
    assert record["tokens_nonpad"] == 5  # three pads
    assert record["utilisation_packing"] == pytest.approx(5 / 8)
    assert record["loss_tokens"] == {"images": 3}  # absent modality contributes nothing
    assert record["target_values"] == {"images": 3 * 192}
    assert record["rows"] == 2  # one <|bos|> per packed row here
    assert record["rows_per_s"] == pytest.approx(2 / 0.25)

    # draining resets, so the next step starts clean
    drained = telemetry.drain_step()
    assert drained["tokens_total"] == 0
    assert drained["rows"] == 0
    assert drained["rows_per_s"] == 0.0


def test_counters_accumulate_across_the_micro_batches_of_one_step():
    flat = {
        "input_ids": torch.tensor([[1, 3, PAD_ID, PAD_ID]]),
        "images_mask": torch.tensor([[False, True, False, False]]),
        "images_values": torch.zeros((1, 192)),
    }
    telemetry.observe_micro_batch(flat, wait_s=0.1)
    telemetry.observe_micro_batch(flat, wait_s=0.4)
    record = telemetry.drain_step()
    assert record["micro_batches"] == 2
    assert record["loader_wait_s"] == pytest.approx(0.5)
    assert record["tokens_nonpad"] == 4
    assert record["loss_tokens"] == {"images": 2}
    assert record["rows"] == 2  # one row per micro-batch, accumulated
    assert record["rows_per_s"] == pytest.approx(2 / 0.5)


def test_rows_counts_galaxies_not_tensor_rows():
    """A packed row can hold several objects — ``rows`` must count objects."""
    input_ids = torch.tensor([[BOS_ID, 3, BOS_ID, 3, PAD_ID]])
    flat = {"input_ids": input_ids}
    telemetry.observe_micro_batch(flat, wait_s=1.0)
    record = telemetry.drain_step()
    assert record["rows"] == 2  # two packed objects in one tensor row
    assert record["rows_per_s"] == pytest.approx(2.0)


def test_loader_wrapper_times_batches_and_proxies(tiny_config, tmp_path, monkeypatch):
    monkeypatch.setenv(telemetry.TELEMETRY_DIR_ENV, str(tmp_path))
    # ADR 0015: no dataset checkpoint state to reach through the wrapper —
    # only the plain DataLoader surface (dataset, num_workers, iteration).
    dataset = PackedMicroBatches(tiny_config, MBS, SEQ_LEN)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None)
    wrapped = telemetry.instrument(loader)
    assert isinstance(wrapped, telemetry.TelemetryLoader)
    assert wrapped.dataset is dataset
    assert wrapped.num_workers == 0


def test_write_step_appends_one_line_per_step(tmp_path, monkeypatch):
    monkeypatch.setenv(telemetry.TELEMETRY_DIR_ENV, str(tmp_path))
    telemetry.write_step(1, {"tokens_nonpad": 7}, rank=1)
    telemetry.write_step(2, {"tokens_nonpad": 9}, rank=1)
    lines = (tmp_path / "steps.rank1.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]
    assert json.loads(lines[1])["tokens_nonpad"] == 9


def test_byte_probe_records_payload_bytes(tmp_path, monkeypatch):
    """The probe must count RETURNED payload, not requests (§11)."""
    monkeypatch.setenv(telemetry.TELEMETRY_DIR_ENV, str(tmp_path))
    monkeypatch.setenv("RANK", "7")

    from huggingface_hub.hf_file_system import HfFileSystemFile

    monkeypatch.setattr(
        HfFileSystemFile,
        "_fetch_range",
        lambda self, start, end: b"x" * (end - start),
        raising=False,
    )
    assert telemetry.install_byte_probe(rank=0, worker=3)

    fake = HfFileSystemFile.__new__(HfFileSystemFile)
    object.__setattr__(
        fake,
        "path",
        "datasets/UniverseTBD/mmu_ssl_legacysurvey_north/part.parquet",
    )
    payload = HfFileSystemFile._fetch_range(fake, 100, 356)
    assert len(payload) == 256

    entry = json.loads((tmp_path / "bytes.rank7.w3.jsonl").read_text().splitlines()[0])
    assert entry["rank"] == 7
    assert entry["dp_rank"] == 0
    assert entry["bytes"] == 256
    assert (entry["start"], entry["end"]) == (100, 356)
    assert entry["source"] == "legacy"
    assert entry["worker"] == 3
    assert entry["wait_s"] >= 0
    # The same range fetched twice costs bytes twice; never deduplicate it.
    HfFileSystemFile._fetch_range(fake, 100, 356)
    records = [json.loads(line) for line in (tmp_path / "bytes.rank7.w3.jsonl").read_text().splitlines()]
    assert sum(record["bytes"] for record in records) == 512


def test_stream_probe_counts_returned_payload_not_requested_length(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from huggingface_hub.hf_file_system import HfFileSystemStreamFile

    monkeypatch.setenv(telemetry.TELEMETRY_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(HfFileSystemStreamFile, "read", lambda self, length=-1: b"abc")
    assert telemetry.install_byte_probe(rank=0)
    assert telemetry.install_byte_probe(rank=0)  # idempotent, no doubled log
    fake = SimpleNamespace(loc=12, path="datasets/UniverseTBD/mmu_desi_edr_sv3/properties")
    assert HfFileSystemStreamFile.read(fake, 999) == b"abc"  # pyright: ignore[reportArgumentType] -- fake remote file
    records = (tmp_path / "bytes.rank0.main.jsonl").read_text().splitlines()
    assert len(records) == 1
    entry = json.loads(records[0])
    assert entry["bytes"] == 3
    assert entry["source"] == "desi"
    assert entry["worker"] is None
