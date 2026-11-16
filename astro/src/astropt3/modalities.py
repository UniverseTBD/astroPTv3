"""Modality registry and the modules that move data in and out of embedding space.

Ported from astroPT (src/astropt/model.py). Each modality contributes:

- ``Encoder``:   data space  -> embedding space (one token per patch)
- ``PositionEmbedder``: per-modality positional information, added to the
  input embeddings (SmolLM3's RoPE/NoPE over the flat sequence is unchanged).
- a per-modality flow (``flows.JetFormerFlow``) + ``GMMHead`` (embedding space -> data
  space, the jetformer regression head; see below).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModalityConfig:
    """Configuration for a single modality.

    Attributes:
        name: modality name; registry order (alphabetical) fixes sequence order.
        input_size: flattened patch vector length (e.g. 8*8*3 = 192 for images).
        patch_size: patch side length (images) or window length (1-D data).
        pos_type: "index" for learned integer-position embeddings,
            "continuous" for a projected float position (e.g. wavelength).
        pos_input_size: dimensionality of the continuous position vector.
        max_positions: number of learned positions when pos_type == "index".
        family: fixed ADR 0013 objective family: image, spectrum, or scalar.
        source: source survey/product name used for provenance and dispatch.
        record_keys: record field(s) carrying this modality.
        token_ids: frozen (begin, placeholder, end) special-token block.
        channel_tokenization: image factorisation (ADR 0014 §8).
            ``fused`` (default) makes one token per patch across all bands
            (8*8*3 = 192 floats); ``per_band`` makes one token per patch PER
            band (8*8 = 64 floats, 3x the tokens) from identical bytes — a
            finer autoregressive factorisation with narrower heads, not more
            information. ``E_values`` is unchanged either way.
        image_crop: central crop side in pixels before patchify (0 keeps the
            whole image); 96 turns 152x152 survey cutouts into 144 patch-8 tokens.
        band_order: fixed band serialization order for ``per_band``, in
            record band names (``des-g``, ``des-r``, ``des-z``). Fixed, not
            shuffled: band order as an ADR 0008 ordering decision is a
            separate experiment, and this one changes one variable.
        ADR 0008 scalar modality (``family == "scalar"``) — a one-token span holding a
            physical quantity (Z, ebv, photometry). Scalars bypass the
            jetformer flow (their dims are odd and a flow buys nothing on a
            scalar) and are predicted by a ``GMMHead`` directly, with
            ``gmm_nll`` on the raw normalized value as the loss.
    """

    name: str
    input_size: int
    patch_size: int
    pos_type: str = "index"
    pos_input_size: int = 1
    max_positions: int = 1024
    family: str | None = None
    source: str | None = None
    record_keys: tuple[str, ...] = ()
    token_ids: tuple[int, int, int] | None = None
    channel_tokenization: str = "fused"
    band_order: tuple[str, ...] = ()
    image_crop: int = 96

    def __post_init__(self):
        if self.family not in {"image", "spectrum", "scalar"}:
            raise ValueError(
                f"modality {self.name!r} has invalid family {self.family!r}"
            )
        if self.channel_tokenization not in {"fused", "per_band"}:
            raise ValueError(
                f"modality {self.name!r} has invalid channel_tokenization "
                f"{self.channel_tokenization!r} (fused | per_band)"
            )
        self.band_order = tuple(self.band_order)
        if type(self.image_crop) is not int or self.image_crop < 0:
            raise ValueError(f"modality {self.name!r} image_crop must be a nonnegative integer")
        if self.channel_tokenization == "per_band":
            if self.family != "image":
                raise ValueError(
                    f"modality {self.name!r} is {self.family!r}; per-band "
                    "tokenisation only applies to images"
                )
            if not self.band_order:
                raise ValueError(
                    f"modality {self.name!r} is per_band but names no "
                    "band_order; the order must be fixed, not inferred"
                )
            expected = self.patch_size * self.patch_size
            if self.input_size != expected:
                raise ValueError(
                    f"modality {self.name!r} is per_band with patch_size "
                    f"{self.patch_size}, so input_size must be {expected}, "
                    f"got {self.input_size}"
                )
        if not self.source:
            raise ValueError(f"modality {self.name!r} is missing source provenance")
        self.record_keys = tuple(self.record_keys)
        if not self.record_keys:
            raise ValueError(f"modality {self.name!r} has no record_keys")
        raw_token_ids = self.token_ids
        if raw_token_ids is None or len(raw_token_ids) != 3:
            raise ValueError(f"modality {self.name!r} needs three token_ids")
        try:
            first, second, third = (int(token_id) for token_id in raw_token_ids)
        except (TypeError, ValueError) as error:
            raise ValueError(f"modality {self.name!r} has invalid token_ids") from error
        token_ids = (first, second, third)
        if token_ids != tuple(range(token_ids[0], token_ids[0] + 3)):
            raise ValueError(f"modality {self.name!r} token_ids must be consecutive")
        self.token_ids = token_ids

    @property
    def scalar(self) -> bool:
        return self.family == "scalar"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "pos_type": self.pos_type,
            "pos_input_size": self.pos_input_size,
            "max_positions": self.max_positions,
            "family": self.family,
            "source": self.source,
            "record_keys": list(self.record_keys),
            "token_ids": list(self.token_ids or ()),
            "channel_tokenization": self.channel_tokenization,
            "band_order": list(self.band_order),
            "image_crop": self.image_crop,
        }


class ModalityRegistry:
    """Central registry for model modalities.

    ``names()`` returns modalities sorted alphabetically; this order defines
    the per-object sequence order (as in astroPT).
    """

    def __init__(self, modalities):
        mods = [
            m if isinstance(m, ModalityConfig) else ModalityConfig(**m)
            for m in modalities
        ]
        self.modalities = {m.name: m for m in mods}

    def get_config(self, name: str) -> ModalityConfig:
        return self.modalities[name]

    def names(self) -> list[str]:
        return sorted(self.modalities.keys())

    def to_list(self) -> list[dict]:
        return [self.modalities[n].to_dict() for n in self.names()]


class Encoder(nn.Module):
    """Data space -> embedding space: a single linear projection.

    The flow that precedes jetformer's latent space lives on the model
    (``AstroPT3Model.flows``); this just projects the (possibly flowed)
    patch value into embedding space.
    """

    def __init__(self, hidden_size: int, in_size: int, bias: bool = False):
        super().__init__()
        self.c_fc = nn.Linear(in_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.c_fc(x)


class PositionEmbedder(nn.Module):
    """Per-modality positional embedding, added to the input embeddings.

    pos_type "index": learned nn.Embedding over integer patch indices.
    pos_type "continuous": affine projection of a float position vector
    (e.g. normalized wavelength), so positions off the training grid embed
    sensibly.
    """

    def __init__(self, hidden_size: int, modality: ModalityConfig, bias: bool = False):
        super().__init__()
        self.pos_type = modality.pos_type
        if modality.pos_type == "index":
            self.embed = nn.Embedding(modality.max_positions, hidden_size)
        elif modality.pos_type == "continuous":
            self.embed = nn.Linear(modality.pos_input_size, hidden_size, bias=bias)
        else:
            raise ValueError(f"unknown pos_type {modality.pos_type!r}")

    def forward(self, pos):
        if self.pos_type == "index":
            return self.embed(pos)
        return self.embed(pos.to(self.embed.weight.dtype))


# --- "jetformer" tokeniser: normalizing flow + GMM head ---------------------
#
# Ported from astroPT v2's jetformer (src/astropt/jetformer.py), which
# follows JetFormer (Tschannen et al. 2024) / GIVT (Tschannen et al. 2023).
# v2 flowed on raw image space (TinyFlow2D); here the flow
# (``flows.JetFormerFlow``) runs on patch tokens, with a ViT conditioner
# attending across each object's modality span, because the packed collator
# only ever hands the model flat (N, D) token values — this is also what makes
# the flow modality-generic (images and spectra alike). Per-modality loss is
# ``mean_tokens(NLL_GMM(z) - logdet)``: exact likelihood in (standardized)
# patch space, so it can be negative. The v2 noise curriculum lives on the
# model (``AstroPT3Model.set_jet_noise_frac``), not here.


class GMMHead(nn.Module):
    """Embedding space -> diagonal Gaussian-mixture parameters per token.

    One linear layer emitting K mixture logits + K*(mu, log_sigma) pairs,
    as in GIVT/JetFormer.
    """

    def __init__(self, hidden_size: int, out_size: int, k: int, bias: bool = False):
        super().__init__()
        self.k = k
        self.d = out_size
        self.proj = nn.Linear(hidden_size, k * (1 + 2 * out_size), bias=bias)

    def forward(self, h):
        out = self.proj(h).view(*h.shape[:-1], self.k, 1 + 2 * self.d)
        logits_pi = out[..., 0]
        mu = out[..., 1 : 1 + self.d]
        log_sigma = out[..., 1 + self.d :].clamp(-7.0, 2.0)
        return logits_pi, mu, log_sigma


def gmm_nll(y, logits_pi, mu, log_sigma):
    """Per-token negative log-likelihood of y under the predicted GMM.

    y: (..., D); logits_pi: (..., K); mu/log_sigma: (..., K, D) -> (...,).
    """
    diff = y.unsqueeze(-2) - mu
    logp = (
        -0.5 * (diff.pow(2) * torch.exp(-2 * log_sigma)).sum(dim=-1)
        - log_sigma.sum(dim=-1)
        - 0.5 * mu.size(-1) * math.log(2 * math.pi)
    )
    return -torch.logsumexp(F.log_softmax(logits_pi, dim=-1) + logp, dim=-1)
