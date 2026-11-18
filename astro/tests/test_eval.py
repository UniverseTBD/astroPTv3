"""Model-side eval functions: mean-batch loss, embeddings, ridge, scalar metrics.

All inputs are provided by the caller (ADR 0015 §6); source-backed collection
lives at the deferred LSDB evaluation seam.
"""

import math

import numpy as np
import pytest
import torch
from legacy_fixture import record_stream

from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.eval import linear_probe, scalar_head, val_loss


def _batches(tiny_config, n=2):
    sequencer = ObjectSequencer(tiny_config)
    collator = PackedCollator(tiny_config, seq_len=896)
    return [
        collator([sequencer.build(record) for record in record_stream(4)])
        for _ in range(n)
    ]


def _probe_set(tiny_config, n=12):
    sequencer = ObjectSequencer(tiny_config)
    objects, targets = [], []
    for record in record_stream(64):
        if record.get("Z") is None or "spectrum" not in record:
            continue
        obj = sequencer.build(record, include_scalars=False)
        if "spectra" not in obj.masks:
            continue
        objects.append(obj)
        targets.append(float(record["Z"]))
        if len(objects) >= n:
            break
    return objects, np.asarray(targets, dtype=np.float64)


def test_evaluate_over_provided_batches_is_finite(tiny_model, tiny_config):
    batches = _batches(tiny_config)
    result = val_loss.evaluate(tiny_model, batches=batches)
    assert math.isfinite(result["loss"]) and result["n_batches"] == len(batches)
    assert "images" in result["modality_losses"]
    with pytest.raises(ValueError, match="no validation batches"):
        val_loss.evaluate(tiny_model, batches=[])


@pytest.mark.parametrize("evaluate", [
    lambda model: val_loss.evaluate(model, batches=()),
    lambda model: linear_probe.embed_objects(model, model.config, []),
    lambda model: scalar_head.scalar_head_metrics(model, [], []),
])
def test_evaluation_requires_eval_mode(tiny_model, evaluate):
    tiny_model.train()
    try:
        with pytest.raises(ValueError, match=r"model\.eval\(\)"):
            evaluate(tiny_model)
    finally:
        tiny_model.eval()


def test_ridge_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 16))
    w = rng.normal(size=16)
    y = X @ w + 0.05 * rng.normal(size=400)
    assert linear_probe.ridge_r2(X, y, seed=0)["r2"] > 0.95


def test_ridge_no_signal_r2_near_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 16))
    y = rng.normal(size=400)
    assert linear_probe.ridge_r2(X, y, seed=0)["r2"] < 0.2


def test_embeddings_align_with_objects(tiny_model, tiny_config):
    objects, _ = _probe_set(tiny_config, 6)
    X = linear_probe.embed_objects(
        tiny_model, tiny_config, objects, seq_len=896, objects_per_batch=4,
        pool_modality="spectra",
    )
    assert X.shape == (6, tiny_config.hidden_size)
    assert np.isfinite(X).all()
    X_single = linear_probe.embed_objects(
        tiny_model, tiny_config, objects, seq_len=896, objects_per_batch=1,
        pool_modality="spectra",
    )
    assert np.allclose(X, X_single, atol=1e-4)


def test_probe_cache_roundtrip(tiny_config, tmp_path):
    objects, targets = _probe_set(tiny_config, 6)
    cache = tmp_path / "probe_set.npz"
    linear_probe._write_probe_cache(cache, {"k": 1}, objects, targets)
    key, objects2, targets2 = linear_probe._read_probe_cache(cache)
    assert key == {"k": 1}
    assert len(objects2) == len(objects)
    assert np.array_equal(targets, targets2)
    assert torch.equal(objects[0].input_ids, objects2[0].input_ids)


def test_scalar_head_metrics_shapes_and_ranges(tiny_model, tiny_config):
    sequencer = ObjectSequencer(tiny_config)
    objects, targets = [], []
    for record in record_stream(32):
        if record.get("Z") is None or "spectrum" not in record:
            continue
        others = sorted(m for m in sequencer.build(record).masks if m != "Z")
        obj = sequencer.build(record, modality_order=others + ["Z"])
        objects.append(obj)
        targets.append(float(obj.values["Z"][0, 0]))
        if len(objects) >= 6:
            break
    targets = np.asarray(targets)
    metrics = scalar_head.scalar_head_metrics(
        tiny_model, objects, targets, seq_len=896
    )
    assert metrics["n_objects"] == len(objects)
    assert 0.0 <= metrics["outlier_frac"] <= 1.0
    assert 0.0 <= metrics["coverage_1sig"] <= 1.0
    assert math.isfinite(metrics["nmad"])
