"""Offline checks for the bounded paired benchmark; never launch GPU work."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_factor_out as bench  # pyright: ignore[reportMissingImports]


def test_source_mask_uses_physical_channels_and_handles_zero_mad():
    modality = SimpleNamespace(channel_tokenization="fused", band_order=(), patch_size=2)
    pixels = torch.zeros(8, 4, 3) + torch.tensor([1.0, 10.0, 100.0])
    pixels[2, 1, 1] += 5
    mask = bench.source_mask((pixels / 0.01).asinh().flatten(1), modality, 0.01)
    assert mask.tolist() == [False, False, True, False, False, False, False, False]
    modality.channel_tokenization, modality.band_order = "per_band", ("a", "b", "c")
    tokens = (pixels / 0.01).asinh().permute(2, 0, 1).reshape(24, 4)
    assert torch.equal(bench.source_mask(tokens, modality, 0.01), mask.repeat(3))


def test_components_are_token_weighted():
    stats = {}
    bench.accumulate(stats, "total", torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([1.0]))
    bench.accumulate(stats, "total", torch.tensor([8.0, 8.0, 8.0]), torch.zeros(3), torch.ones(3))
    bench.accumulate(stats, "empty", torch.empty(0), torch.empty(0), torch.empty(0))
    assert bench.summarize(stats) == {"total": {"tokens": 4, "ar": 6.5, "prior": 0.75, "logdet": 1.0, "nll": 6.25}}


def test_collection_deduplicates_and_closes_on_success_and_exhaustion():
    class Stream:
        closed = False

        def draws(self, **kwargs):
            assert kwargs == {"seed": 7, "rank": 0, "worker_id": 0}
            try:
                while True:
                    yield 0, ["a", "a", "b"]
            finally:
                self.closed = True

    sequencer = SimpleNamespace(build=lambda record, epoch: SimpleNamespace(object_id=record))
    stream = Stream()
    assert [o.object_id for o in bench.collect(stream, sequencer, 2, 7)] == ["a", "b"]
    assert stream.closed
    stream = Stream()
    with pytest.raises(RuntimeError, match="2/3 unique"):
        bench.collect(stream, sequencer, 3, 7, max_draws=2)
    assert stream.closed


@pytest.mark.parametrize("fraction", [1.0, 0.4, 0.2])
def test_eval_matches_model_and_only_patch_width_changes(tiny_config, sequencer, collator, full_record, fraction):
    config = bench.arm_config(tiny_config, fraction)
    for old, new in zip(tiny_config.modalities, config.modalities, strict=True):
        assert {k: v for k, v in old.items() if k != "d_ar"} == {k: v for k, v in new.items() if k != "d_ar"}
        assert new["d_ar"] == (old["input_size"] if old["family"] == "scalar" else max(1, int(old["input_size"] * fraction)))
        assert old["d_ar"] == old["input_size"]
    model = bench.AstroPT3Model(config).eval()
    batch = collator([sequencer.build(full_record)])
    with torch.no_grad():
        expected = model(**batch)
    actual = bench.evaluate(model, [batch], torch.device("cpu"))
    for name, loss in expected.modality_losses.items():
        assert actual[name]["nll"] == pytest.approx(loss.item(), rel=1e-6, abs=1e-4)
        if name in expected.loss_components:
            for component, value in expected.loss_components[name].items():
                assert actual[name][component] == pytest.approx(value.item(), rel=1e-6, abs=1e-4)
    assert actual["images/source/total"]["tokens"] + actual["images/sky/total"]["tokens"] == actual["images"]["tokens"]


def test_train_arm_cpu_smoke(tiny_config, sequencer, collator, full_record):
    baseline = bench.AstroPT3Model(bench.arm_config(tiny_config, 1.0)).state_dict()
    batch = collator([sequencer.build(full_record)])
    recipe = {"tokens": {"train_steps": 10000}, "optimizer": {
        "weight_decay": 0.1, "clip_grad": 1.0,
        "learning_rate_scheduler": {"learning_rate": 0.001},
        "optimizer_factory": {"adam_beta1": 0.9, "adam_beta2": 0.95, "adam_eps": 1e-8}}}
    result = bench.train_arm(bench.arm_config(tiny_config, 0.4), baseline, [batch], [batch], recipe,
                             SimpleNamespace(seed=0, steps=2), torch.device("cpu"))
    assert result["timed_steps"] == result["warmup_steps"] == 1
    assert result["peak_train_memory_bytes"] is None
    assert result["mean_step_seconds"] > 0
    assert len(result["training_curve"]) == 2
    assert result["shared_state_tensors"] < len(baseline)
    assert result["eval"]["total"]["tokens"] > 0
