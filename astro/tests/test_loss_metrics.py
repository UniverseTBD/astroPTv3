"""Step loss pooling on two data ranks, each replicated on two tensor ranks."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn


def _check_dp_metrics(rank, rendezvous):
    from nanotron.astropt3_metrics import reduce_loss_metrics

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        dp_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
        dp = dp_groups[rank % 2]
        for targets in ([(6, 3), (8, 1)], [(0, 0), (14, 4)], [(0, 0), (0, 0)], [(0, 3), (0, 1)]):
            loss_sum, count = targets[rank // 2]
            stats = torch.tensor([(loss_sum, count), (loss_sum / 2, count), (0, 0), (0, 0)], dtype=torch.float64)
            outputs = [{"loss_stats": stats}, {"loss_stats": torch.zeros_like(stats)}]
            metrics = reduce_loss_metrics(outputs, ["patch"], dp)
            expected_count = sum(n for _, n in targets)
            expected: dict[str, float] = {
                "loss_count/patch": expected_count, "loss_count/image_family": expected_count,
                "loss_count/spectrum_family": 0, "loss_count/scalar_family": 0,
            }
            if expected_count:
                mean = sum(s for s, _ in targets) / expected_count
                expected.update(patch_loss=mean, image_family_loss=mean / 2)
            assert metrics == expected
        _check_validation(rank, dp, Path(rendezvous.removeprefix("file://")).parent)
    finally:
        dist.destroy_process_group()


def _check_validation(rank, dp, run_dir):
    from nanotron.astropt3_validation import evaluate_validation_shard, load_validation_shard
    from astropt3.data import validation_panel

    def load(model_config, objects, seq_len, batch_size, dp_size):
        assert rank == 0  # only the coordinator reads from the source
        assert (objects, seq_len, batch_size, dp_size) == (2, 32, 1, 2)
        return ["catalog:1", "catalog:2"], [[], [{"targets": torch.tensor(t)} for t in ([-2., -2., -2.], [4.])]]

    class Model(torch.nn.Module):
        def forward(self, targets):
            assert not self.training and not torch.is_grad_enabled()
            total, count = targets.sum().item(), targets.numel()
            return {"loss_stats": torch.tensor([(total, count), (total / 2, count), (0, 0), (0, 0)], dtype=torch.float64)}

    config = SimpleNamespace(
        tokens=SimpleNamespace(validation_objects=2, sequence_length=32, micro_batch_size=1, validation_manifest=None),
        checkpoints=SimpleNamespace(checkpoints_path=run_dir),
    )
    context = SimpleNamespace(world_pg=dist.group.WORLD, dp_pg=dp, get_local_ranks=lambda r: {"dp": r // 2})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(validation_panel, "load_validation_panel", load)
        batches = load_validation_shard(config, None, context, None)
    assert len(batches) == 2 * (rank // 2)
    model = Model()
    for _ in range(2):
        metrics = evaluate_validation_shard(model, batches, ["patch"], "cpu", dp)
        assert model.training
        assert metrics.pop("val/seconds") >= 0
        assert metrics == {
            "val/patch_loss": -0.5, "val/image_family_loss": -0.25,
            "loss_count/val/patch": 4, "loss_count/val/image_family": 4,
            "loss_count/val/spectrum_family": 0, "loss_count/val/scalar_family": 0,
        }
    assert (run_dir / validation_panel.PANEL_FILENAME).is_file()


def test_loss_metrics_dp2_tp2(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "nanotron/src"))
    spawn(_check_dp_metrics, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4)
