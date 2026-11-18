"""Mean model loss over already-provided batches.

Model-side only (ADR 0015 §6): batch construction lives with the record
source, which is deferred; callers pass an iterable of HF forward-kwargs.
"""

import torch


@torch.no_grad()
def evaluate(model, *, batches):
    """Mean loss (and per-modality/family means) over ``batches``.

    ``batches`` is an iterable of HF ``AstroPT3Model`` forward keyword dicts,
    consumed on the model's own device and dtype. Call ``model.eval()`` first.
    """
    if model.training:
        raise ValueError("Call model.eval() before evaluating loss")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    total, per_key, per_key_n = 0.0, {}, {}
    per_family, per_family_n = {}, {}
    n = 0
    for kwargs in batches:
        kwargs = {
            k: (
                {
                    kk: vv.to(
                        device=device, dtype=dtype if vv.is_floating_point() else None
                    )
                    for kk, vv in v.items()
                }
                if isinstance(v, dict)
                else v.to(device)
            )
            for k, v in kwargs.items()
        }
        out = model(**kwargs)
        total += out.loss.item()
        for key, value in out.modality_losses.items():
            per_key[key] = per_key.get(key, 0.0) + value.item()
            per_key_n[key] = per_key_n.get(key, 0) + 1
        for key, value in out.family_losses.items():
            per_family[key] = per_family.get(key, 0.0) + value.item()
            per_family_n[key] = per_family_n.get(key, 0) + 1
        n += 1
    if n == 0:
        raise ValueError("no validation batches provided")
    return {
        "loss": total / n,
        "modality_losses": {k: per_key[k] / per_key_n[k] for k in sorted(per_key)},
        "family_losses": {
            k: per_family[k] / per_family_n[k] for k in sorted(per_family)
        },
        "n_batches": n,
    }
