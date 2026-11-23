from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from astropt3.data.nanotron_loader import share_tp_batches


def _check_tp_stream(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=30))
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        builds = []
        reads = []

        def build_loader():
            builds.append(rank)

            def read_batches():
                for i in range(2):
                    reads.append(i)
                    if rank == 2 and i == 1:
                        raise ValueError("catalog fetch failed")
                    yield {
                        "ids": torch.tensor([rank, i]),
                        "mask": torch.tensor([True, False]),
                        "empty": torch.empty((0, 3)),
                    }

            return read_batches()

        batches = share_tp_batches(build_loader, groups[rank // 2], dist.group.WORLD)
        batch = next(batches)
        assert list(batch) == ["ids", "mask", "empty"]
        torch.testing.assert_close(batch["ids"], torch.tensor([rank // 2 * 2, 0]))
        torch.testing.assert_close(batch["mask"], torch.tensor([True, False]))
        torch.testing.assert_close(batch["empty"], torch.empty((0, 3)))
        with pytest.raises(RuntimeError, match="catalog reader failed") as failure:
            next(batches)
        if rank == 2:
            assert isinstance(failure.value.__cause__, ValueError)
        assert builds == ([rank] if rank % 2 == 0 else [])
        assert reads == ([0, 1] if rank % 2 == 0 else [])
    finally:
        dist.destroy_process_group()


def test_tp_catalog_stream_shares_batches_and_reader_failures(tmp_path):
    spawn(_check_tp_stream, args=((tmp_path / "rendezvous").as_uri(),), nprocs=4)
