"""TP=2 replicated-module check for the nanotron AstroPT3 model.

Verifies the one subtle piece of the fork's TP story (plan item 3): modality
encoders/decoders/pos-embedders are plain modules replicated across TP ranks,
tied by ``mark_unsharded_params_as_tied_across_tp`` with ``reduce_op=None``
under ALL_REDUCE — i.e. "synced by design". This holds only if, given
identical inputs and identical replicated weights, every TP rank computes
identical losses AND identical gradients for those modules without any
communication. That is what this script asserts.

Run (needs 2 GPUs and [train] env; uses offline test records):
    CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=2 \
        astro/scripts/tp2_grad_check.py

Prints ``TP2 GRAD CHECK PASS`` on rank 0 when all assertions hold.

Flake note: on a COLD kernel cache (fresh node/allocation) the very first run
can fail the bitwise grad check at ~1e-4 (bf16 ulp) scale — timing-based
kernel autotune can pick different kernels per rank before the cache warms.
Re-run once; steady-state runs are bitwise stable (verified 4/4 on GH200).
"""

# Pyright runs in the CPU dev venv; these imports exist in the GPU Nanotron venv.
# pyright: reportMissingImports=false
import sys
import zlib
from pathlib import Path
from typing import cast

import nanotron.models
import torch
from nanotron import distributed as dist
from nanotron.config import (
    AstroPT3Config,
    OneForwardOneBackwardPipelineEngine,
    ParallelismArgs,
    TensorParallelLinearMode,
)
from nanotron.models.astropt3 import AstroPT3ForTraining
from nanotron.parallel import ParallelContext
from nanotron.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
)
from nanotron.trainer import mark_tied_parameters

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from legacy_fixture import make_record

from astropt3.data.nanotron_loader import flatten_packed_batch, hf_config_from_modalities
from astropt3.data.packing import ObjectSequencer, PackedCollator

MBS = 2
SEQ_LEN = 896

# nanotron parameter prefixes of the TP-replicated modality modules
REPLICATED_PREFIXES = (
    "model.token_position_embeddings.pp_block.encoders.",
    "model.token_position_embeddings.pp_block.pos_embeds.",
    "model.token_position_embeddings.pp_block.flows.",
    "model.modality_head.pp_block.decoders.",
)


def deterministic_init(model):
    """Seed every parameter from a hash of its name.

    Replicated modules end up bit-identical across TP ranks (same name, same
    seed); TP-sharded body params are filled per-shard, which is fine for
    this check.
    """
    for name, param in sorted(model.named_parameters()):
        seed = zlib.crc32(name.encode())  # process-independent, unlike hash()
        generator = torch.Generator(device=param.device).manual_seed(seed)
        with torch.no_grad():
            param.normal_(mean=0.0, std=0.02, generator=generator)


def gather_across_tp(tensor: torch.Tensor, tp_pg) -> list:
    chunks = [torch.empty_like(tensor) for _ in range(tp_pg.size())]
    dist.all_gather(chunks, tensor.contiguous(), group=tp_pg)
    return chunks


def main():
    config = AstroPT3Config(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=4096,
        rope_theta=100000.0,
        no_rope_layer=4,
        rms_norm_eps=1e-6,
        vocab_size=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        tie_word_embeddings=False,
        attention_bias=False,
        _attn_implementation="flash_attention_2",
        _use_qkv_packed=True,
        _use_doc_masking=True,
        log_attn_probs=False,
    )
    parallel_config = ParallelismArgs(
        dp=1,
        pp=1,
        tp=2,
        pp_engine=OneForwardOneBackwardPipelineEngine(),
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )
    parallel_context = ParallelContext(
        data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=2
    )
    tp_pg = parallel_context.tp_pg
    tp_rank = dist.get_rank(tp_pg)

    # jetformer draws curriculum noise under the trainer's tp_synced random
    # state — build one here so the check exercises that path
    random_states = RandomStates(
        {
            "tp_synced": get_synced_random_state(
                random_state=get_current_random_state(), pg=tp_pg
            )
        }
    )
    model = nanotron.models.build_model(
        model_builder=lambda: AstroPT3ForTraining(
            config=config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        ),
        parallel_context=parallel_context,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    model = cast(AstroPT3ForTraining, model)
    mark_tied_parameters(model=model, parallel_context=parallel_context)
    deterministic_init(model)
    # frac=0 -> sigma = noise_max: the noise path must stay TP-identical
    model.set_jet_noise_frac(0.0)

    replicated = {
        name: param
        for name, param in model.named_parameters()
        if name.startswith(REPLICATED_PREFIXES)
    }
    assert replicated, (
        "no replicated modality parameters found — did the naming change?"
    )

    # replicated weights identical across TP after deterministic init
    for name, param in sorted(replicated.items()):
        for other in gather_across_tp(param.detach(), tp_pg):
            assert torch.equal(param.detach(), other), f"init drift across TP in {name}"

    # Identical offline inputs isolate TP parity from catalog availability.
    hf_config = hf_config_from_modalities(config.modalities)
    sequencer = ObjectSequencer(hf_config)
    rows = [[sequencer.build(make_record(i, image_only_fraction=0.0))] for i in range(MBS)]
    packed = PackedCollator(hf_config, seq_len=SEQ_LEN).collate_rows(rows)
    flat = flatten_packed_batch(packed, hf_config, SEQ_LEN)
    flat = {k: v.cuda() for k, v in flat.items()}

    out = model(**flat)
    loss = out["loss"]

    # loss identical across TP ranks (replicated hidden stream under ALL_REDUCE)
    loss_value = loss.detach().reshape(1)
    for other in gather_across_tp(loss_value, tp_pg):
        assert torch.equal(loss_value, other), "loss differs across TP ranks"

    loss.backward()

    # gradients of replicated modules identical across TP ranks WITHOUT any
    # gradient reduction — the "synced by design" contract of reduce_op=None
    checked = 0
    for name, param in sorted(replicated.items()):
        grad = param.grad
        assert grad is not None, f"no grad for replicated param {name}"
        assert torch.isfinite(grad).all(), f"non-finite grad in {name}"
        for other in gather_across_tp(grad, tp_pg):
            assert torch.equal(grad, other), f"grad drift across TP in {name}"
        checked += 1

    # body (sharded) params must also have finite grads
    for name, param in model.named_parameters():
        if name not in replicated:
            assert param.grad is not None and torch.isfinite(param.grad).all(), name

    if tp_rank == 0:
        print(
            f"TP2 GRAD CHECK PASS ({checked} replicated params, loss {loss.item():.4f})"
        )


if __name__ == "__main__":
    main()
