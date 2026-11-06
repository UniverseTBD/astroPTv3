"""Shuffled batch exposures from one already-loaded LSDB draw (no I/O)."""

from __future__ import annotations

import random

import torch

from ..tokenization import BOS_ID, PAD_ID
from .packing import span_order

ECHO_METADATA = "_astropt3_echo"


def validate_echo_factor(factor):
    if type(factor) is not int or factor < 1:
        raise ValueError("echo-factor must be a positive integer")


def reshuffle_spans(batch, token_ids, *, identity, seed):
    """Move whole spans, never objects or within-modality patches/targets."""
    original = batch["input_ids"]
    if original.device.type != "cpu":
        raise ValueError("Echoing requires CPU batches before device transfer")
    ids = torch.empty_like(original, pin_memory=original.is_pinned())
    ids.copy_(original)
    by_begin = {
        begin: (name, placeholder, end)
        for name, (begin, placeholder, end) in token_ids.items()
    }
    for row_index, row in enumerate(original.tolist()):
        cursor = 0
        while cursor < len(row) and row[cursor] != PAD_ID:
            if row[cursor] != BOS_ID:
                raise ValueError("Echo batch must contain whole packed objects")
            start = cursor + 1
            cursor = start
            spans = {}
            while cursor < len(row) and row[cursor] not in (BOS_ID, PAD_ID):
                if row[cursor] not in by_begin:
                    raise ValueError("Invalid modality delimiter in echo batch")
                name, placeholder, end_id = by_begin[row[cursor]]
                end = row.index(end_id, cursor + 1)
                if name in spans or any(
                    token != placeholder for token in row[cursor + 1:end]
                ):
                    raise ValueError(
                        "Echo batch requires one contiguous span per modality per object"
                    )
                spans[name] = row[cursor:end + 1]
                cursor = end + 1
            if len(spans) > 1:
                # Packed batches no longer carry object IDs; use draw-local identity.
                order = span_order(sorted(spans), f"{identity}:{row_index}:{start}", seed)
                ids[row_index, start:cursor] = torch.tensor(
                    [token for name in order for token in spans[name]], dtype=ids.dtype
                )
    echoed = {**batch, "input_ids": ids}
    for name, (_, placeholder, _) in token_ids.items():
        mask = batch[f"{name}_mask"]
        echoed[f"{name}_mask"] = torch.empty_like(mask, pin_memory=mask.is_pinned())
        torch.eq(ids, placeholder, out=echoed[f"{name}_mask"])
    return echoed


def echo_draw(batches, config, *, echo_factor, seed):
    """Shuffle exactly N exposures per batch, retaining only this draw's payloads.

    Metadata is worker-to-main only: TelemetryLoader removes it before training.
    Freshness/index follow actual presentation order, not the shuffled tickets.
    """
    validate_echo_factor(echo_factor)
    registry = config.modality_registry()
    token_ids = {name: registry.get_config(name).token_ids for name in registry.names()}
    order = list(range(len(batches))) * echo_factor
    if echo_factor > 1:
        random.Random(seed).shuffle(order)
    uses = [0] * len(batches)
    for batch_index in order:
        index = uses[batch_index]
        uses[batch_index] += 1
        batch = batches[batch_index]
        delivered = (
            reshuffle_spans(batch, token_ids, identity=f"{batch_index}:{index}", seed=seed)
            if index else dict(batch)
        )
        delivered[ECHO_METADATA] = (index, echo_factor)
        yield delivered
