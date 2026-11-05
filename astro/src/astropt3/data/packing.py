"""Object -> token sequence assembly and greedy packing into fixed-length batches.

Per object (only the modalities present):

    <|bos|> <|begin_images|> p0 ... p360 <|end_images|>
            <|begin_spectra|> s0 ... s30 <|end_spectra|>
            <|begin_Z|> z <|end_Z|> ...

Multi-span objects serialize their spans in a UNIFORM random order, seeded
on ``crc32(object_id) ^ epoch`` (ADR 0008, superseding the 0005 bimodal
parity rule — at two spans the shuffle IS that rule's 50/50 flip). Every
conditional among the present spans lands in the training distribution;
the seed changes each epoch, and being a pure function of (object_id,
epoch) the order is exact under checkpoint resume (no ambient RNG state).
Checkpoints trained before these rules (fixed images-first) are
incompatible with sequences the rule builds — retrain.

ADR 0008 scalar modalities (Z / ebv / photometry) are one-token spans over
the record's catalog scalars, normalized by ``data/scalar_registry.py``;
``Z`` is gated on DESI's ``ZWARN == 0`` reliability flag. A missing scalar
is an absent span — the ordinary modality-optional path.

The collator packs whole objects greedily into rows of ``seq_len`` tokens;
objects are never split. ``position_ids`` restart at 0 on each object, which
is both the RoPE position and the packed-document boundary signal
(transformers' ``create_causal_mask`` builds the block-diagonal doc mask from
these restarts when ``attention_mask`` is None). Tail padding uses
``<|pad|>`` with position_id 0, so each pad token forms its own one-token
document and cannot attend to (or be attended by) real tokens.

Flattened ``modality_values``/``modality_positions`` are concatenated in
row-major (batch, time) order — the same order a boolean mask lookup
``tensor[mask]`` produces — so the model can align them without indices.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass
from typing import Any

import torch

from ..configuration_astropt3 import AstroPT3Config
from ..tokenization import (
    BOS_ID,
    PAD_ID,
    normalize_wavelength,
    patchify_image,
    patchify_spectrum,
    spiralise,
)
from .band_registry import _DIV_FACTOR, physical_normalize
from .scalar_registry import scalar_normalize
from .spectral import _DIV_FACTOR as _SPECTRA_DIV_FACTOR
from .spectral import spectral_normalize

# side of the central image crop applied before patchify, in pixels
IMAGE_CROP = 96

# ADR 0014 §5: bumped whenever :func:`span_order` changes, since a different
# permutation rule changes the emitted sequences without changing record
# order. Folded into the loader's resume fingerprint.
SPAN_ORDER_VERSION = "adr0008_uniform_v1"


def span_order(names, object_id: str, epoch: int = 0) -> list[str]:
    """The ADR 0008 span order for one object: uniform, seeded, resume-exact.

    Shared by :meth:`ObjectSequencer.build` and batch echoing, which redraws
    orders without re-tokenizing physical targets or changing model RNG state.
    """
    order = list(names)
    if len(order) > 1:
        seed = zlib.crc32(str(object_id).encode()) ^ epoch
        random.Random(seed).shuffle(order)
    return order


@dataclass
class ObjectSeq:
    """One object's token sequence and its continuous payloads."""

    input_ids: torch.Tensor  # int64 [L]
    masks: dict  # name -> bool [L]
    values: dict  # name -> [n_m, input_size]
    positions: dict  # name -> long [n_m] or float [n_m, pos_input_size]
    object_id: str = ""
    # the span order this sequence was serialized in
    order: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.input_ids)


class ObjectSequencer:
    """Turn an MMU-schema record into an :class:`ObjectSeq`.

    The jetformer tokeniser models an exact likelihood in patch space, so the
    record -> token map must stay invertible: no per-patch standardization
    (which would discard each patch's mean/std) is ever applied — tokens are
    the asinh-stretched (images) / raw (spectra) patch values.
    """

    def __init__(self, config: AstroPT3Config):
        self.registry = config.modality_registry()
        # spiral patch order comes from the config only (ADR 0004): the
        # checkpoint self-describes the order it trained in, and the inverse
        # path keys off the same field — a caller-supplied override would
        # reopen the silent-scramble mismatch, so there deliberately isn't one
        self.spiral = getattr(config, "spiral", True)
        # arcsinh knee of the physical image normalization; carried on the
        # config so checkpoints are self-describing and the inverse
        # (scripts/generate.py) uses the divisor the model trained with
        self.image_norm_divisor = getattr(config, "image_norm_divisor", _DIV_FACTOR)
        # spectra counterpart (ADR 0007): arcsinh knee of the DESI f_ν
        # normalization, likewise carried on the config
        self.spectra_norm_divisor = getattr(
            config, "spectra_norm_divisor", _SPECTRA_DIV_FACTOR
        )

    def _image_tokens(self, name: str, record: dict):
        mod = self.registry.get_config(name)
        image = record[mod.record_keys[0]]
        flux = torch.as_tensor(image["flux"], dtype=torch.float32)
        # central crop: 152x152 survey cutouts -> 96x96 (144 patch-8 tokens);
        # JWST cubes are already 96x96 and pass through untouched
        h, w = flux.shape[-2:]
        if h > IMAGE_CROP or w > IMAGE_CROP:
            top = (h - IMAGE_CROP) // 2
            left = (w - IMAGE_CROP) // 2
            flux = flux[..., top : top + IMAGE_CROP, left : left + IMAGE_CROP]
        # may arrive as a list or an array after a parquet round-trip
        bands = [str(b) for b in image["band"]]
        flux = physical_normalize(
            flux, bands, divisor=self.image_norm_divisor
        )
        if mod.channel_tokenization == "per_band":
            return self._per_band_tokens(mod, flux, bands)
        patches = patchify_image(flux, mod.patch_size)
        if self.spiral:
            patches = spiralise(patches)
        positions = torch.arange(len(patches), dtype=torch.long)
        return patches, positions

    def _per_band_tokens(self, mod, flux: torch.Tensor, bands: list[str]):
        """ADR 0014 §8: one token per patch PER band, band-major.

        The same 27,648 target values as the fused path, factorised 3x finer:
        144 x 192 becomes 432 x 64 for a Legacy crop. Cross-band structure is
        then learned autoregressively instead of pre-fused, and the position
        index runs across the whole concatenation, so a token's band is
        implied by ``position // patches_per_band``.

        Band order is FIXED by config (never shuffled) — one variable at a
        time, per §8 — and a record missing a named band raises rather than
        silently shifting every subsequent band's positions.
        """
        missing = [band for band in mod.band_order if band not in bands]
        if missing:
            raise ValueError(
                f"modality {mod.name!r} needs bands {list(mod.band_order)} for "
                f"per-band tokenisation; record has {bands} (missing {missing})"
            )
        per_band = []
        for band in mod.band_order:
            channel = flux[bands.index(band)].unsqueeze(0)
            patches = patchify_image(channel, mod.patch_size)
            if self.spiral:
                patches = spiralise(patches)
            per_band.append(patches)
        patches = torch.cat(per_band, dim=0)
        positions = torch.arange(len(patches), dtype=torch.long)
        return patches, positions

    def _spectrum_tokens(self, name: str, record: dict):
        mod = self.registry.get_config(name)
        spec = record[mod.record_keys[0]]
        flux = torch.as_tensor(spec["flux"], dtype=torch.float32)
        lam = torch.as_tensor(spec["lambda"], dtype=torch.float32)
        mask = torch.as_tensor(spec["mask"], dtype=torch.bool)
        flux = torch.where(mask, torch.zeros_like(flux), flux)
        flux = spectral_normalize(
            flux, lam, divisor=self.spectra_norm_divisor, source=mod.source or ""
        )
        patches, lam_mean = patchify_spectrum(flux, lam, mod.patch_size)
        positions = normalize_wavelength(lam_mean).unsqueeze(-1)
        return patches, positions

    def _scalar_value(self, name: str, record: dict):
        """The record's raw value(s) for a scalar modality, or None if absent.

        Per-scalar missingness IS the modality-optional path: a None here
        just means no span. ``Z`` additionally gates on DESI's ``ZWARN``
        reliability flag (ADR 0008 reuses ADR 0005's cut; a missing flag on
        a Z-bearing record — synthetic pre-ZWARN fixtures — passes).
        """
        mod = self.registry.get_config(name)
        values = []
        for key in mod.record_keys:
            raw_value: Any = record.get(key)
            if raw_value is None:
                return None
            try:
                values.append(float(raw_value))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"modality {name!r} has a non-numeric value"
                ) from error
        if not all(math.isfinite(value) for value in values):
            return None
        if name == "Z" and bool(record.get("ZWARN") or False):
            return None
        return values

    def _scalar_tokens(self, name: str, record: dict):
        value = self._scalar_value(name, record)
        if value is None:
            return None
        # one token of input_size values; per-patch standardization never
        # applies (the mean/std of a single value are degenerate) — the
        # scalar_registry transform is the whole normalization
        values = scalar_normalize(name, torch.tensor([value], dtype=torch.float32))
        return values, torch.zeros(1, dtype=torch.long)

    def build(
        self,
        record: dict,
        *,
        epoch: int = 0,
        modality_order: list[str] | None = None,
        include_scalars: bool = True,
    ) -> ObjectSeq:
        """``modality_order`` pins an explicit span order (generation
        templates, e.g. spectra-first for spectra-to-images); it must name
        exactly the modalities the record carries. ``epoch`` seeds the span
        shuffle — training loaders pass their live epoch.
        ``include_scalars=False`` omits every scalar span (the linear probe
        must pool over sequences that cannot contain the target, ADR 0008)."""
        parts = {}
        for name in self.registry.names():
            mod = self.registry.get_config(name)
            if mod.family == "scalar":
                if include_scalars:
                    tokens = self._scalar_tokens(name, record)
                    if tokens is not None:
                        parts[name] = tokens
                continue
            if record.get(mod.record_keys[0]) is None:
                continue
            if mod.family == "image":
                parts[name] = self._image_tokens(name, record)
            elif mod.family == "spectrum":
                parts[name] = self._spectrum_tokens(name, record)
        if not parts:
            raise ValueError(
                f"record {record.get('object_id')!r} has no known modality"
            )

        if modality_order is not None:
            if sorted(modality_order) != sorted(parts):
                raise ValueError(
                    f"modality_order {modality_order!r} must name exactly the "
                    f"record's modalities {sorted(parts)}"
                )
            order = list(modality_order)
        else:
            # ADR 0008: uniform span shuffle, seeded per (object_id, epoch) —
            # deterministic and resume-exact; at N=2 this is exactly the
            # superseded ADR 0005 parity rule's 50/50 flip in distribution
            order = span_order(parts, record.get("object_id", ""), epoch)

        ids = [BOS_ID]
        spans = {}
        for name in order:
            values, _ = parts[name]
            token_ids = self.registry.get_config(name).token_ids
            if (
                token_ids is None
            ):  # guarded by ModalityConfig; keeps type checkers honest
                raise ValueError(f"modality {name!r} has no token ids")
            begin_id, placeholder_id, end_id = token_ids
            ids.append(begin_id)
            spans[name] = (len(ids), len(ids) + len(values))
            ids.extend([placeholder_id] * len(values))
            ids.append(end_id)

        input_ids = torch.tensor(ids, dtype=torch.long)
        masks, values, positions = {}, {}, {}
        for name, (vals, pos) in parts.items():
            start, stop = spans[name]
            m = torch.zeros(len(ids), dtype=torch.bool)
            m[start:stop] = True
            masks[name] = m
            values[name] = vals
            positions[name] = pos
        return ObjectSeq(
            input_ids=input_ids,
            masks=masks,
            values=values,
            positions=positions,
            object_id=str(record.get("object_id", "")),
            order=tuple(order),
        )


class PackedCollator:
    """Greedily pack whole :class:`ObjectSeq`s into fixed-length rows."""

    def __init__(self, config: AstroPT3Config, seq_len: int = 4096):
        self.seq_len = seq_len
        self.modality_names = config.modality_registry().names()

    def __call__(self, objects: list[ObjectSeq]) -> dict:
        """Greedily pack a flat object list, then collate the rows it makes."""
        rows: list[list[ObjectSeq]] = [[]]
        used = 0
        for obj in objects:
            if len(obj) > self.seq_len:
                raise ValueError(
                    f"object of length {len(obj)} exceeds seq_len {self.seq_len}"
                )
            if used + len(obj) > self.seq_len:
                rows.append([])
                used = 0
            rows[-1].append(obj)
            used += len(obj)
        if not rows[-1]:
            rows.pop()
        return self.collate_rows(rows)

    def collate_rows(self, rows: list[list[ObjectSeq]]) -> dict:
        """Collate rows the caller has already formed.

        The streaming loader fills the emptiest row first, unlike the
        sequential greedy packing in __call__, so it supplies rows directly.
        """
        for row in rows:
            used = sum(len(obj) for obj in row)
            if used > self.seq_len:
                raise ValueError(f"row of length {used} exceeds seq_len {self.seq_len}")

        B, T = len(rows), self.seq_len
        input_ids = torch.full((B, T), PAD_ID, dtype=torch.long)
        position_ids = torch.zeros((B, T), dtype=torch.long)
        masks = {m: torch.zeros((B, T), dtype=torch.bool) for m in self.modality_names}
        values = {m: [] for m in self.modality_names}
        positions = {m: [] for m in self.modality_names}

        for b, row in enumerate(rows):
            t = 0
            for obj in row:
                L = len(obj)
                input_ids[b, t : t + L] = obj.input_ids
                position_ids[b, t : t + L] = torch.arange(L)
                for m in obj.masks:
                    masks[m][b, t : t + L] = obj.masks[m]
                    values[m].append(obj.values[m])
                    positions[m].append(obj.positions[m])
                t += L

        batch = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "modality_masks": {},
            "modality_values": {},
            "modality_positions": {},
        }
        for m in self.modality_names:
            if values[m]:
                batch["modality_masks"][m] = masks[m]
                batch["modality_values"][m] = torch.cat(values[m], dim=0)
                batch["modality_positions"][m] = torch.cat(positions[m], dim=0)
        return batch
