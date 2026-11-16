# JetFormer flow

The sole jetformer flow (`src/astropt3/flows.py`, `JetFormerFlow`), ported from
the cross-patch pilot (see [lineage](jetformer-flow-lineage.md)). Each patch
modality gets `jetformer_flow_steps` affine couplings. Each coupling is
conditioned by `jetformer_flow_depth` pre-norm Transformer blocks of width
`jetformer_flow_hidden`, with full attention over one object's one modality
span (never across packed objects; `flow_object_ids`). It replaced the
per-token patch-MLP flow (`TinyFlow1D`), which was removed.

## Width rule

There is no published scaling rule. JetFormer held its flow fixed (~450M)
while growing the trunk. Ours is a convention: 8 stages, depth 4, 8 heads, and

    jetformer_flow_hidden = min(hidden_size // 4, 512)

Flow parameters scale as width², like a trunk layer, so the four flows stay at
~20–40% of trunk parameters up to 1B. The 512 cap (JetFormer's flow size)
keeps them at ~411M from 1B upward. `tests/test_jetformer_flow.py` enforces the
rule on every `configs/nanotron/*.yaml`. `scripts/count_params.py` reports
flows in their own column and excludes them from the nominal-size check.

| Trunk | hidden_size | Flow width | Four flows |
|---|---:|---:|---:|
| 70M | 512 | 128 | 27.5M |
| 160M | 768 | 192 | ~59M |
| 410M | 1024 | 256 | 105.3M |
| 1B and up | ≥2048 | 512 | ~411M |

## Limits

- CP=1 (asserted): the flow attends over whole object spans, which context
  parallelism would split. DP and TP are supported; flow parameters are
  TP-replicated like the other modality modules.
- MFU FLOPs (`modality_flops_per_token`) price attention over
  `max_positions`, an upper bound on span length (images use 144 of 361), so
  flow FLOPs and MFU are overstated.
- Permutations are derived from seed 42 + stage index, not stored; changing
  that rule breaks existing checkpoints.
