"""Autoregressive scalar prediction metrics over provided objects.

Model-side only (ADR 0015 §6): consumes already-built :class:`ObjectSeq`
sequences whose target span is pinned LAST, and reads the GMM head at the
``starts-1`` position. Metrics in the normalized ``log(1+z)`` space where
sigma reads directly as ``dz/(1+z)``:

- ``nmad``:          1.4826 * median(|residual|) — robust scatter
- ``outlier_frac``:  fraction with |residual| > 0.15
- ``coverage_1sig``: fraction of truths inside the head's 1-sigma highest-
                     weight component interval (~0.68 when calibrated)
- ``r2``:            R^2 of the point estimates in RAW target space
"""

import numpy as np
import torch

from ..data.packing import PackedCollator
from ..data.scalar_registry import scalar_inverse

OUTLIER_THRESHOLD = 0.15  # |d log(1+z)| — the standard photo-z outlier cut


@torch.no_grad()
def scalar_head_metrics(
    model, objects, targets, *, target="Z", seq_len=896, objects_per_batch=8
):
    """Report normalized-space metrics for the target span. Call ``model.eval()`` first."""
    if model.training:
        raise ValueError("Call model.eval() before scoring scalar predictions")
    if target in model.categorical_names:
        raise ValueError(f"{target!r} is categorical; score it by accuracy, not regression metrics")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    collator = PackedCollator(model.config, seq_len=seq_len)
    preds, sigmas = [], []
    for i in range(0, len(objects), objects_per_batch):
        batch = collator(objects[i : i + objects_per_batch])
        kwargs = {
            k: (
                {kk: vv.to(device=device, dtype=dtype if vv.is_floating_point() else None) for kk, vv in v.items()}
                if isinstance(v, dict)
                else v.to(device)
            )
            for k, v in batch.items()
        }
        out = model(**kwargs, compute_loss=False)
        from ..modeling_astropt3 import left_shift_mask

        mask = kwargs["modality_masks"][target]
        logits_pi, mu, log_sigma = model.decoders[target](
            out.last_hidden_state[left_shift_mask(mask)]
        )
        pi = torch.softmax(logits_pi, dim=-1)
        # mixture mean as the point estimate; the top-weight component's
        # sigma as the reported uncertainty (a full-mixture interval is a
        # refinement the calibration gate does not need yet)
        preds.append((pi.unsqueeze(-1) * mu).sum(dim=-2)[:, 0].float().cpu().numpy())
        top = pi.argmax(dim=-1)
        idx = top.view(-1, 1, 1).expand(-1, 1, mu.size(-1))
        sigmas.append(log_sigma.gather(-2, idx)[:, 0, 0].exp().float().cpu().numpy())
    preds = np.concatenate(preds).astype(np.float64)
    sigmas = np.concatenate(sigmas).astype(np.float64)
    if len(preds) != len(targets):
        raise RuntimeError(f"{len(preds)} predictions for {len(targets)} targets")

    targets = np.asarray(targets, dtype=np.float64)
    residuals = preds - targets
    # R^2 in raw target space, comparable to the linear probe's ridge R^2
    raw_pred = scalar_inverse(target, torch.from_numpy(preds)).numpy()
    raw_true = scalar_inverse(target, torch.from_numpy(targets)).numpy()
    ss_tot = ((raw_true - raw_true.mean()) ** 2).sum()
    r2 = (
        1.0 - ((raw_true - raw_pred) ** 2).sum() / ss_tot
        if ss_tot > 0
        else float("nan")
    )  # pi-lens-ignore: unchecked-throwing-call-python (numpy scalar ops)
    metrics = {
        "target": target,
        "n_objects": len(targets),
        "nmad": 1.4826 * np.median(np.abs(residuals)),
        "outlier_frac": (np.abs(residuals) > OUTLIER_THRESHOLD).mean(),
        "coverage_1sig": (np.abs(residuals) <= sigmas).mean(),
        "bias": np.median(residuals),
        "r2": r2,
    }
    # pi-lens-ignore: unchecked-throwing-call-python (finite metrics above), unchecked-throwing-call-python
    return {key: value if key == "target" else float(value) for key, value in metrics.items()}
