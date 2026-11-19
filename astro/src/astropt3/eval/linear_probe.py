"""Ridge linear probe on mean-pooled hidden states — model-side functions.

Consumes already-built :class:`ObjectSeq` probe objects and targets
(ADR 0015 §6); collecting probe objects from a record source is deferred to
a future LSDB-backed evaluation seam.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

from ..data.packing import ObjectSeq, PackedCollator
from ..tokenization import BOS_ID, PAD_ID


def _float_value(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected numeric value, got {value!r}") from error


def _int_value(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected integer value, got {value!r}") from error


def _write_probe_cache(
    path: Path, key: dict, objects: list[ObjectSeq], targets
) -> None:
    arrays = {"targets": targets}
    metadata = {"key": key, "objects": []}
    for i, obj in enumerate(objects):
        item = {"object_id": obj.object_id, "masks": [], "values": [], "positions": []}
        arrays[f"o{i}_input_ids"] = obj.input_ids.numpy()
        for section in ("masks", "values", "positions"):
            for j, (name, tensor) in enumerate(getattr(obj, section).items()):
                array_key = f"o{i}_{section}_{j}"
                arrays[array_key] = tensor.numpy()
                item[section].append((name, array_key))
        metadata["objects"].append(item)
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def _read_probe_cache(path: Path):
    with np.load(path, allow_pickle=False) as arrays:
        try:
            metadata = json.loads(str(arrays["metadata"]))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid probe cache metadata in {path}") from error
        objects = []
        for i, item in enumerate(metadata["objects"]):
            sections = {
                section: {
                    name: torch.from_numpy(arrays[array_key].copy())
                    for name, array_key in item[section]
                }
                for section in ("masks", "values", "positions")
            }
            objects.append(
                ObjectSeq(
                    input_ids=torch.from_numpy(arrays[f"o{i}_input_ids"].copy()),
                    masks=sections["masks"],
                    values=sections["values"],
                    positions=sections["positions"],
                    object_id=item["object_id"],
                )
            )
        return metadata["key"], objects, arrays["targets"].copy()


@torch.no_grad()
def embed_objects(
    model, config, objects, *, seq_len=4096, objects_per_batch=8, pool_modality="legacy_north_g"
):
    """Mean-pool the CENTRAL layer state over one modality's tokens, per object.

    Central = ``hidden_states[num_hidden_layers // 2]`` (astroPT convention;
    embeddings sit at index 0, so this is the output of the middle block).
    Objects are packed with the shared collator; each object's span in a row
    starts at its ``<|bos|>`` and the packed row-major object order equals
    the input order, so embeddings align with the targets by construction.
    Call ``model.eval()`` first.
    """
    if model.training:
        raise ValueError("Call model.eval() before extracting probe embeddings")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    collator = PackedCollator(config, seq_len=seq_len)
    central = config.num_hidden_layers // 2
    features = []
    for i in range(0, len(objects), objects_per_batch):
        batch = collator(objects[i : i + objects_per_batch])
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
            for k, v in batch.items()
        }
        out = model(**kwargs, compute_loss=False, output_hidden_states=True)
        hidden = out.hidden_states[central]  # [B, T, H]
        input_ids = batch["input_ids"]
        mask = batch["modality_masks"].get(pool_modality)
        if mask is None:
            raise ValueError(f"no {pool_modality!r} tokens in probe batch")
        for b in range(input_ids.shape[0]):
            starts = (input_ids[b] == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            pad = (input_ids[b] == PAD_ID).nonzero(as_tuple=True)[0]
            end_of_row = _int_value(pad[0]) if len(pad) else input_ids.shape[1]
            bounds = starts + [end_of_row]
            for s, e in zip(bounds[:-1], bounds[1:]):
                span_mask = mask[b, s:e]
                emb = hidden[b, s:e][span_mask].float().mean(dim=0)
                features.append(emb.cpu().numpy())
    features = np.asarray(features, dtype=np.float64)
    if len(features) != len(objects):
        raise RuntimeError(
            f"recovered {len(features)} embeddings for {len(objects)} objects"
        )
    return features


def ridge_r2(X, y, *, seed=0, lambdas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Closed-form ridge with inner-split lambda selection; returns test R^2."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_test = max(1, len(X) // 5)
    n_val = max(1, (len(X) - n_test) // 5)
    test, val, train = (
        order[:n_test],
        order[n_test : n_test + n_val],
        order[n_test + n_val :],
    )

    mu, sigma = X[train].mean(axis=0), X[train].std(axis=0)
    sigma[sigma == 0] = 1.0
    Xn = (X - mu) / sigma
    y_mean = y[train].mean()

    def fit(idx, lam):
        A = Xn[idx].T @ Xn[idx] + lam * np.eye(Xn.shape[1])
        return np.linalg.solve(A, Xn[idx].T @ (y[idx] - y_mean))

    def r2(w, idx):
        pred = Xn[idx] @ w + y_mean
        ss_res = ((y[idx] - pred) ** 2).sum()
        ss_tot = ((y[idx] - y[idx].mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan

    best_lam = max(lambdas, key=lambda lam: r2(fit(train, lam), val))
    w = fit(np.concatenate([train, val]), best_lam)
    return {
        "r2": _float_value(r2(w, test)),
        "lambda": _float_value(best_lam),
        "n_test": _int_value(n_test),
    }


def probe_checkpoint(
    checkpoint,
    *,
    target="Z",
    seq_len=4096,
    objects_per_batch=8,
    pool_modality="legacy_north_g",
    device=None,
    seed=0,
    probe_set,
):
    """Probe one checkpoint against a pre-collected ``(objects, targets)`` set."""
    from transformers import AutoModel

    import astropt3  # noqa: F401  -- registers the Auto classes

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(checkpoint).to(device=device, dtype=dtype).eval()

    objects, targets = probe_set
    X = embed_objects(
        model,
        model.config,
        objects,
        seq_len=seq_len,
        objects_per_batch=objects_per_batch,
        pool_modality=pool_modality,
    )
    result = ridge_r2(X, targets, seed=seed)
    result.update(
        {
            "checkpoint": str(checkpoint),
            "target": target,
            "n_objects": len(objects),
            "pool_modality": pool_modality,
        }
    )
    return result
