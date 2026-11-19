"""Step loss pooling on two data ranks, each replicated on two tensor ranks."""

from datetime import timedelta
from pathlib import Path

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
                "patch_loss_count": expected_count, "image_family_loss_count": expected_count,
                "spectrum_family_loss_count": 0, "scalar_family_loss_count": 0,
            }
            if expected_count:
                mean = sum(s for s, _ in targets) / expected_count
                expected.update(patch_loss=mean, image_family_loss=mean / 2)
            assert metrics == expected
    finally:
        dist.destroy_process_group()


def test_loss_metrics_dp2_tp2(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "nanotron/src"))
    spawn(_check_dp_metrics, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4)
