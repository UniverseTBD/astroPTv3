"""Offline draw-local echoes: packing, scheduling, and worker/main accounting."""

import random
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from legacy_fixture import make_record

from astropt3.data import nanotron_loader, telemetry
from astropt3.data.echo import ECHO_METADATA, echo_draw
from astropt3.data.nanotron_loader import (
    PackedMicroBatches,
    build_astropt3_dataloader,
    consumer_seed,
    flatten_packed_batch,
)
from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.tokenization import BOS_ID, PAD_ID


class FiniteDraws(PackedMicroBatches):
    def __init__(self, *args, draws, **kwargs):
        super().__init__(*args, **kwargs)
        self.draws = draws
        self.fetched = []

    def _open_draws(self, worker):  # pyright: ignore[reportIncompatibleMethodOverride] -- deliberately finite test source
        for epoch, records in enumerate(self.draws):
            self._epoch = epoch
            self.fetched.append(epoch)
            yield iter(records)


def _no_io(*args, **kwargs):
    pytest.fail("an offline draw test tried to open live data")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv(telemetry.TELEMETRY_DIR_ENV, raising=False)
    monkeypatch.setattr(nanotron_loader, "open_catalog", _no_io)
    monkeypatch.setattr(nanotron_loader, "LocalCluster", _no_io)
    monkeypatch.setattr(nanotron_loader, "install_byte_probe", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    telemetry.drain_step()
    yield
    telemetry.drain_step()


@pytest.fixture
def records():
    # Drop catalog scalars explicitly: the shared tiny config enables them.
    return [
        {key: record[key] for key in ("object_id", "image", "spectrum")}
        for record in (make_record(i, image_only_fraction=0.0) for i in range(10))
    ]


def _only(record, modality):
    return {"object_id": record["object_id"], modality: record[modality]}


def _pack(config, rows, seq_len, epoch=0):
    sequencer = ObjectSequencer(config)
    objects = [[sequencer.build(record, epoch=epoch) for record in row] for row in rows]
    return flatten_packed_batch(
        PackedCollator(config, seq_len).collate_rows(objects), config, seq_len
    )


def _payload_key(batch):
    return tuple(
        (key, value.numpy().tobytes())
        for key, value in sorted(batch.items())
        if key.endswith("_values")
    )


def _assert_tensors_equal(actual, expected):
    assert set(actual) - {ECHO_METADATA} == set(expected) - {ECHO_METADATA}
    for key in expected:
        if key != ECHO_METADATA:
            assert torch.equal(actual[key], expected[key]), key


def test_draw_schedule_counts_first_exposures_and_keeps_payloads(tiny_config, records):
    bases = [
        _pack(tiny_config, [[_only(records[0], "image")]], 256),
        _pack(tiny_config, [[_only(records[1], "spectrum")]], 256),
        _pack(tiny_config, [[records[2]]], 256),
    ]
    snapshots = [{key: value.clone() for key, value in batch.items()} for batch in bases]
    python_rng, torch_rng = random.getstate(), torch.random.get_rng_state()
    emitted = list(echo_draw(bases, tiny_config, echo_factor=3, seed=0))
    assert random.getstate() == python_rng
    assert torch.equal(torch.random.get_rng_state(), torch_rng)

    keys = [_payload_key(batch) for batch in bases]
    order = [keys.index(_payload_key(batch)) for batch in emitted]
    expected_order = list(range(3)) * 3
    random.Random(0).shuffle(expected_order)
    assert order == expected_order
    assert Counter(order) == {0: 3, 1: 3, 2: 3}
    assert order != [index for index in dict.fromkeys(order) for _ in range(3)]

    uses = Counter()
    changed = False
    for batch, batch_index in zip(emitted, order, strict=True):
        base = snapshots[batch_index]
        assert batch[ECHO_METADATA] == (uses[batch_index], 3)
        if uses[batch_index] == 0 or batch_index < 2:
            _assert_tensors_equal(batch, base)
        else:
            changed |= not torch.equal(batch["input_ids"], base["input_ids"])
        uses[batch_index] += 1
        for key in base:
            if key == "position_ids" or key.endswith(("_values", "_positions")):
                assert torch.equal(batch[key], base[key]), key
        for name in tiny_config.modality_registry().names():
            placeholder = tiny_config.modality_registry().get_config(name).token_ids[1]
            assert torch.equal(batch[f"{name}_mask"], batch["input_ids"] == placeholder)
    assert changed, "the chosen multimodal replays should redraw span order"
    for base, snapshot in zip(bases, snapshots, strict=True):
        assert ECHO_METADATA not in base
        _assert_tensors_equal(base, snapshot)

    repeated = list(echo_draw(bases, tiny_config, echo_factor=3, seed=0))
    for batch, replay in zip(emitted, repeated, strict=True):
        assert batch[ECHO_METADATA] == replay[ECHO_METADATA]
        _assert_tensors_equal(batch, replay)
    different = list(echo_draw(bases, tiny_config, echo_factor=3, seed=43))
    assert [_payload_key(batch) for batch in different] != [_payload_key(batch) for batch in emitted]


@pytest.mark.parametrize("modality,seq_len", [("image", 200), ("spectrum", 40), (None, 200)])
def test_two_draws_never_mix_or_fetch_early_and_keep_tail(
    tiny_config, records, modality, seq_len, monkeypatch
):
    if modality:
        records = [_only(record, modality) for record in records]
    draws = [records[:5], records[5:]]
    dataset = FiniteDraws(tiny_config, 2, seq_len, draws=draws, echo_factor=3, seed=19, rank=2)
    worker = SimpleNamespace(id=3)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: worker)
    iterator = iter(dataset)
    for epoch, draw in enumerate(draws):
        bases = [
            _pack(tiny_config, [[draw[i]], [draw[i + 1]] if i + 1 < len(draw) else []], seq_len, epoch)
            for i in range(0, len(draw), 2)
        ]
        expected = list(echo_draw(
            bases, tiny_config, echo_factor=3, seed=consumer_seed(19, 2, 3, epoch)
        ))
        actual = []
        for reference in expected:
            batch = next(iterator)
            assert dataset.fetched == list(range(epoch + 1)), "fetched the next draw during echoes"
            assert batch[ECHO_METADATA] == reference[ECHO_METADATA]
            _assert_tensors_equal(batch, reference)
            actual.append(batch)
        assert Counter(_payload_key(batch) for batch in actual) == {
            _payload_key(base): 3 for base in bases
        }
        assert sum(int((batch["input_ids"] == BOS_ID).sum()) for batch in actual) == 15
    with pytest.raises(StopIteration):
        next(iterator)


@pytest.mark.parametrize("factor", [1, 3])
def test_empty_draw_helper(tiny_config, factor):
    assert list(echo_draw([], tiny_config, echo_factor=factor, seed=0)) == []


def test_empty_and_tiny_draws_emit_only_nonempty_padded_tails(tiny_config, records):
    draws = [[], [_only(records[0], "image")], [], [_only(records[1], "spectrum")], []]
    dataset = FiniteDraws(tiny_config, 2, 200, draws=draws, echo_factor=2)
    batches = list(dataset)
    assert len(batches) == 4
    assert dataset.fetched == list(range(5))
    for batch in batches:
        assert batch["input_ids"].shape == (2, 200)
        assert int((batch["input_ids"] == BOS_ID).sum()) == 1
        assert (batch["input_ids"][1] == PAD_ID).all()
        assert (batch["position_ids"][1] == 0).all()
    assert sum(bool(batch["images_mask"].any()) for batch in batches) == 2
    assert sum(bool(batch["spectra_mask"].any()) for batch in batches) == 2
    assert list(FiniteDraws(tiny_config, 2, 200, draws=[[], []], echo_factor=3)) == []


@pytest.mark.parametrize("kwargs", [{}, {"echo_factor": 1}])
def test_factor_one_preserves_streaming_across_draw_boundaries(tiny_config, records, kwargs):
    records = [_only(record, "image") for record in records[:3]]
    dataset = FiniteDraws(tiny_config, 1, 300, draws=[[records[0]], records[1:]], **kwargs)
    batches = list(dataset)
    assert len(batches) == 1  # Existing infinite-stream path does not flush a finite tail.
    assert ECHO_METADATA not in batches[0]
    _assert_tensors_equal(batches[0], _pack(tiny_config, [records[:2]], 300))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("num_workers", [0, 1])
def test_worker_metadata_is_stripped_with_telemetry_off_or_on(
    tiny_config, records, enabled, num_workers, monkeypatch, tmp_path
):
    if enabled:
        monkeypatch.setenv(telemetry.TELEMETRY_DIR_ENV, str(tmp_path))
    draw = [_only(records[0], "image"), _only(records[1], "spectrum"), records[2]]
    dataset = FiniteDraws(tiny_config, 1, 180, draws=[draw], echo_factor=3, seed=7)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers else None,
    )
    wrapped = telemetry.instrument(loader, echo_factor=3)
    assert isinstance(wrapped, telemetry.TelemetryLoader)
    assert wrapped.dataset is dataset
    assert wrapped.num_workers == num_workers
    seen = Counter()
    for batch in wrapped:
        assert ECHO_METADATA not in batch
        assert all(isinstance(value, torch.Tensor) and value.device.type == "cpu" for value in batch.values())
        key = _payload_key(batch)
        index = seen[key]
        seen[key] += 1
        record = telemetry.drain_step()
        if not enabled:
            assert record["micro_batches"] == record["tokens_total"] == 0
            continue
        assert record["echo_index"] == index
        assert record["echo_factor"] == 3
        assert record["micro_batches"] == 1
        assert record["tokens_total"] == 180
        assert record["tokens_nonpad"] == int((batch["input_ids"] != PAD_ID).sum())
        assert record["rows"] == 1
        assert record["loader_wait_s"] >= 0
        for prefix, include in (("fresh", index == 0), ("replayed", index > 0)):
            for field in ("micro_batches", "tokens_nonpad", "rows"):
                assert record[f"{prefix}_{field}"] == (record[field] if include else 0)
            for field, suffix in (("loss_tokens", "_mask"), ("target_values", "_values")):
                counts = {
                    name.removesuffix(suffix): int(value.sum()) if suffix == "_mask" else value.numel()
                    for name, value in batch.items() if name.endswith(suffix) and value.numel()
                }
                counts = {name: count for name, count in counts.items() if count}
                assert record[field] == counts
                assert {name: count for name, count in record[f"{prefix}_{field}"].items() if count} == (
                    counts if include else {}
                )
    assert len(seen) == 3
    assert set(seen.values()) == {3}


def test_telemetry_accumulates_actual_fresh_and_replayed_batches(tiny_config, records, monkeypatch, tmp_path):
    monkeypatch.setenv(telemetry.TELEMETRY_DIR_ENV, str(tmp_path))
    bases = [_pack(tiny_config, [[record]], 200) for record in records[:3]]
    batches = echo_draw(bases, tiny_config, echo_factor=3, seed=42)
    assert len(list(telemetry.instrument(batches, echo_factor=3))) == 9
    record = telemetry.drain_step()
    for field in ("micro_batches", "tokens_nonpad", "rows"):
        assert record[field] == record[f"fresh_{field}"] + record[f"replayed_{field}"]
        assert record[f"replayed_{field}"] == 2 * record[f"fresh_{field}"]
    assert record["fresh_micro_batches"] == 3
    for field in ("loss_tokens", "target_values"):
        for name, count in record[field].items():
            assert count == 3 * record[f"fresh_{field}"][name]
            assert record[f"replayed_{field}"][name] == 2 * record[f"fresh_{field}"][name]


@pytest.mark.parametrize("factor", [0, -1, True, False, 1.5, "2", None])
def test_invalid_factors_fail_before_data_open(tiny_config, factor):
    with pytest.raises(ValueError, match="positive integer"):
        list(echo_draw([], tiny_config, echo_factor=factor, seed=0))
    with pytest.raises(ValueError, match="positive integer"):
        PackedMicroBatches(tiny_config, 1, 200, echo_factor=factor)
    with pytest.raises(ValueError, match="positive integer"):
        build_astropt3_dataloader(SimpleNamespace(echo_factor=factor), object(), 1, 200, 0, 1)


@pytest.mark.parametrize("topology", [
    {"dp_size": 2}, {"world_size": 2}, {"batch_accumulation_per_replica": 2},
])
def test_unsupported_topology_rejected_before_constructing_dataset(monkeypatch, topology):
    monkeypatch.setattr(nanotron_loader, "PackedMicroBatches", _no_io)
    with pytest.raises(ValueError, match="echo-factor.*requires"):
        build_astropt3_dataloader(
            SimpleNamespace(echo_factor=3), object(), 1, 200, 0,
            **{"dp_size": 1, **topology},
        )


@pytest.mark.parametrize("factor", [None, 1, 3])
def test_builder_wires_echo_factor_and_preserves_factor_one_topologies(tiny_config, factor):
    args = SimpleNamespace() if factor is None else SimpleNamespace(echo_factor=factor)
    loader = build_astropt3_dataloader(
        args, tiny_config, 1, 200, 0, 1 if factor == 3 else 2,
        world_size=1 if factor == 3 else 4,
        batch_accumulation_per_replica=1 if factor == 3 else 2,
    )
    assert loader.dataset.echo_factor == (factor or 1)
    assert isinstance(loader, telemetry.TelemetryLoader) == (factor == 3)
