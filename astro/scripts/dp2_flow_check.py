"""DP=2 check for the JetFormer flow under the real nanotron trainer.

Each DP rank trains a few steps on a DIFFERENT local batch (no LSDB reads).
Passing requires finite, rank-distinct losses and bit-identical parameters
across ranks afterwards (flows included), i.e. DP gradient sync covers the
whole-span flows.

Run (needs 2 GPUs and the nanotron GPU env):
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 astro/scripts/dp2_flow_check.py

Prints ``DP2 FLOW CHECK PASS`` on rank 0 when all assertions hold.
"""

# pyright: reportMissingImports=false
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from nanotron import distributed as dist
from nanotron.trainer import DistributedTrainer

ASTRO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASTRO_DIR / "tests"))
from test_jetformer_flow_nanotron_gpu import SEQ_LEN, _flat_batch  # noqa: E402

STEPS = 3


def main():
    config = yaml.safe_load((ASTRO_DIR / "tests/fixtures/astropt3-test-tiny.yaml").read_text())
    recipe = yaml.safe_load((ASTRO_DIR / "configs/nanotron/astropt3-70m.yaml").read_text())
    config["model"]["model_config"].update(modalities=recipe["model"]["model_config"]["modalities"])
    config["data_stages"] = recipe["data_stages"]  # parsed only: streams are never opened
    config["parallelism"]["dp"] = 2
    config["tokens"].update(sequence_length=SEQ_LEN, micro_batch_size=2)
    config["optimizer"]["learning_rate_scheduler"].update(lr_warmup_steps=0, lr_warmup_style="linear")
    workdir = Path(tempfile.mkdtemp(prefix="dp2_flow_check_"))
    config["checkpoints"]["checkpoints_path"] = str(workdir / "checkpoints")
    path = workdir / f"dp2-{dist.get_rank() if dist.is_initialized() else 'x'}.yaml"
    path.write_text(yaml.safe_dump(config))

    trainer = DistributedTrainer(str(path), benchmark=True)
    dp_pg = trainer.parallel_context.dp_pg
    assert dp_pg.size() == 2, dp_pg.size()
    rank = dist.get_rank(dp_pg)
    model = trainer.unwrapped_model
    flat = _flat_batch(model.config)
    for name in model.config.modality_names():
        flat[f"{name}_values"] = flat[f"{name}_values"] * (1 + rank)  # distinct data per rank

    model.train()
    losses = []
    for _ in range(STEPS):
        trainer.optimizer.zero_grad()
        loss = trainer.model(**flat)["loss"]  # DDP-wrapped forward: gradients all-reduce over DP
        trainer.grad_accumulator.backward(loss)
        trainer.optimizer.step()
        trainer.lr_scheduler.step()
        losses.append(loss.detach().float())
    losses = torch.stack(losses)
    assert torch.isfinite(losses).all(), losses

    gathered = [torch.empty_like(losses) for _ in range(2)]
    dist.all_gather(gathered, losses, group=dp_pg)
    assert not torch.equal(gathered[0], gathered[1]), "ranks saw identical data; check is vacuous"

    flow_params = 0
    for name, param in sorted(model.named_parameters()):
        copies = [torch.empty_like(param) for _ in range(2)]
        dist.all_gather(copies, param.detach().contiguous(), group=dp_pg)
        assert torch.equal(copies[0], copies[1]), f"DP ranks diverged in {name}"
        flow_params += ".flows." in name
    assert flow_params, "no flow parameters checked"
    assert all(
        torch.count_nonzero(block.out_proj.weight) > 0
        for flow in model.model.token_position_embeddings.pp_block.flows.values()
        for block in flow.blocks
    ), "flow couplings never left their zero init"
    if dist.get_rank() == 0:
        print(f"DP2 FLOW CHECK PASS ({flow_params} flow tensors, losses {gathered[0].tolist()} vs {gathered[1].tolist()})")
    dist.barrier()


if __name__ == "__main__":
    main()
