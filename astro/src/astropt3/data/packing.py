"""Object -> token sequence assembly and greedy packing into fixed-length batches.

Per object (only the modalities present):

    <|bos|> <|begin_legacy_north_g|> p0 ... p143 <|end_legacy_north_g|>
            <|begin_spectra|> s0 ... s30 <|end_spectra|>
            <|begin_Z|> z <|end_Z|> ...

Multi-span objects serialize their spans in a UNIFORM random order, seeded
on ``crc32(object_id) ^ epoch`` (ADR 0008, superseding the 0005 bimodal
parity rule — at two spans the shuffle IS that rule's 50/50 flip). Every
conditional among the present spans lands in the training distribution;
the seed changes each epoch, and being a pure function of (object_id,
epoch) the order is exact under checkpoint resume (no ambient RNG state).

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
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import torch

from ..configuration_astropt3 import AstroPT3Config
from ..modalities import ModalityConfig
from ..tokenization import (
    BOS_ID,
    PAD_ID,
    normalize_wavelength,
    patchify_image,
    patchify_spectrum,
    spiralise,
)
from .band_registry import RGB_CHANNELS, physical_normalize, validate_physical_bands
from .scalar_registry import MISSING_SENTINELS, scalar_normalize
from .spectral import spectral_normalize

# side of the central image crop applied before patchify, in pixels
IMAGE_CROP = ModalityConfig.image_crop

# ADR 0014 §5: bumped whenever :func:`span_order` changes, since a different
# permutation rule changes the emitted sequences without changing record
# order. Folded into the loader's resume fingerprint.
SPAN_ORDER_VERSION = "adr0008_uniform_v1"


def has_modality(modality, record: dict) -> bool:
    """Whether a record carries a modality; ``Z`` also needs ``ZWARN`` 0 or absent."""
    return (all(key in record for key in modality.record_keys)
            and (modality.name != "Z" or not record.get("ZWARN")))


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


# Normalized-unit floor for the fill noise; ~the 1st percentile of SDSS spectral
# pixel noise (DESI and imaging are noisier).
FILL_SIGMA_MIN = 0.01


def noise_fill(values: torch.Tensor, fill: torch.Tensor, seed: int) -> torch.Tensor:
    """Replace ``fill`` entries of ``values`` (C, ...) with N(0, sigma_c) noise.

    Masked bins, tail padding and no-data pixels are exact constants: point
    masses whose continuous likelihood the model can drive to -inf, which
    eventually dominates the gradient. Noise at each channel's own pixel-noise
    level (robust std of last-axis neighbour differences / sqrt 2, floored at
    ``FILL_SIGMA_MIN``) bounds it. Filled entries carried no signal, so the map
    stays invertible on real data. Seeded, so tokens remain a pure function of
    the record (resume-exact, deterministic eval).
    """
    if not fill.any():
        return values
    diff = values[..., 1:] - values[..., :-1]
    valid = ~(fill[..., 1:] | fill[..., :-1])
    diff = torch.where(valid, diff, torch.nan).reshape(len(values), -1)
    centred = diff - diff.nanmedian(dim=-1, keepdim=True).values
    sigma = 1.4826 / 2**0.5 * centred.abs().nanmedian(dim=-1).values
    sigma = torch.nan_to_num(sigma, nan=FILL_SIGMA_MIN).clamp(min=FILL_SIGMA_MIN)
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(values.shape, generator=gen) * sigma.view(-1, *[1] * (values.ndim - 1))
    return torch.where(fill, noise.to(values.dtype), values)


@dataclass
class ObjectSeq:
    """One object's token sequence and its continuous payloads."""

    input_ids: torch.Tensor  # int64 [L]
    masks: dict  # name -> bool [L]
    values: dict  # name -> [n_m, input_size]
    positions: dict  # name -> long [n_m] or float [n_m, pos_input_size]
    object_id: str = ""

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
        self.spiral = config.spiral
        # arcsinh knee of the physical image normalization; carried on the
        # config so checkpoints are self-describing and the inverse
        # (scripts/generate.py) uses the divisor the model trained with
        self.image_norm_divisor = config.image_norm_divisor
        # spectra counterpart (ADR 0007): arcsinh knee of the DESI f_ν
        # normalization, likewise carried on the config
        self.spectra_norm_divisor = config.spectra_norm_divisor

    def _image_tokens(self, name: str, record: dict, seed: int = 0):
        mod = self.registry.get_config(name)
        image = record[mod.record_keys[0]]
        bands = [str(band) for band in image["band"]]
        if mod.image_format == "rgb":
            if tuple(bands) != RGB_CHANNELS:
                raise ValueError(f"RGB modality {name!r} requires channels {RGB_CHANNELS}; got {bands}")
            selected = slice(None)
        else:
            validate_physical_bands(bands)
            band = mod.band
            if band is None or bands.count(band) != 1:
                raise ValueError(f"image modality {name!r} needs exactly one {band!r} channel; got {bands}")
            channel = bands.index(band)
            selected = slice(channel, channel + 1)
        cube = torch.as_tensor(image["flux"])
        if cube.ndim != 3 or cube.shape[0] != len(bands):
            raise ValueError(f"image modality {name!r} has inconsistent flux shape and band labels")
        flux = cube[selected].to(torch.float32)
        fill = None
        if "fill_mask" in image:
            mask = torch.as_tensor(image["fill_mask"], dtype=torch.bool)
            if mask.shape != cube.shape:
                raise ValueError(f"image modality {name!r} has inconsistent flux and fill_mask shapes")
            fill = mask[selected]
        h, w = flux.shape[-2:]
        crop = mod.image_crop  # 0 keeps the whole image
        if crop and (h > crop or w > crop):
            top = max(0, (h - crop) // 2)
            left = max(0, (w - crop) // 2)
            region = (..., slice(top, top + crop), slice(left, left + crop))
            flux = flux[region]
            if fill is not None:
                fill = fill[region]
        flux = physical_normalize(flux, bands[selected], divisor=self.image_norm_divisor)
        # Use the published pixel validity when available; do not erase measured zeros.
        flux = noise_fill(flux, flux == 0 if fill is None else fill, seed)
        patches = patchify_image(flux, mod.patch_size)
        if self.spiral:
            patches = spiralise(patches)
        positions = torch.arange(len(patches), dtype=torch.long)
        return patches, positions

    def _spectrum_tokens(self, name: str, record: dict, seed: int = 0):
        mod = self.registry.get_config(name)
        spec = record[mod.record_keys[0]]
        flux = torch.as_tensor(spec["flux"], dtype=torch.float32)
        lam = torch.as_tensor(spec["lambda"], dtype=torch.float32)
        mask = torch.as_tensor(spec["mask"], dtype=torch.bool)
        # zero first so masked garbage never reaches the normalization
        flux = torch.where(mask, torch.zeros_like(flux), flux)
        flux = spectral_normalize(
            flux, lam, divisor=self.spectra_norm_divisor, source=mod.source or ""
        )
        patches, lam_mean = patchify_spectrum(flux, lam, mod.patch_size)
        # masked bins and the last patch's tail padding become noise, not exact zeros
        fill = torch.ones(patches.numel(), dtype=torch.bool)
        fill[: len(mask)] = mask
        patches = noise_fill(patches.reshape(1, -1), fill.reshape(1, -1), seed).view_as(patches)
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
        if any(value == MISSING_SENTINELS.get(name) for value in values):
            return None
        if not has_modality(mod, record):
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
        only_modalities: Collection[str] | None = None,
    ) -> ObjectSeq:
        """``modality_order`` pins an explicit span order (generation
        templates, e.g. spectra-first for spectra-to-images); it must name
        exactly the modalities selected from the record (all present by default,
        or restricted by ``only_modalities``). ``epoch`` seeds the span
        shuffle — training loaders pass their live epoch.
        ``include_scalars=False`` omits every scalar span (the linear probe
        must pool over sequences that cannot contain the target, ADR 0008)."""
        parts = {}
        for name in self.registry.names():
            if only_modalities is not None and name not in only_modalities:
                continue
            mod = self.registry.get_config(name)
            if mod.family == "scalar":
                if include_scalars:
                    tokens = self._scalar_tokens(name, record)
                    if tokens is not None:
                        parts[name] = tokens
                continue
            if record.get(mod.record_keys[0]) is None:
                continue
            # fill-noise seed: per (object, modality, epoch), like the span order
            seed = zlib.crc32(f"{record.get('object_id', '')}/{name}".encode()) ^ epoch
            if mod.family == "image":
                parts[name] = self._image_tokens(name, record, seed)
            elif mod.family == "spectrum":
                parts[name] = self._spectrum_tokens(name, record, seed)
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
