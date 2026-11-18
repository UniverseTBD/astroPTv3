"""Special-token vocabulary and patchification.

Existing special-token ids are FROZEN: released checkpoints depend on them.
Layout: 0 <|pad|>, 1 <|bos|>, and three consecutive IDs per modality:
<|begin_m|>, <|m|> (placeholder), <|end_m|>. Built-ins own fixed blocks;
automatically allocated modalities take free blocks below 64 first.

Patchification is pinned to the verified MMU schemas:
- images:  float32 (3, 152, 152) cubes, center-cropped to 96x96 by the
  sequencer -> 144 patch-8 tokens of 64 floats per band
- spectra: float32 (7781,) flux -> pad to 7936 -> 31 patches of 256 floats
"""

from __future__ import annotations

import math

import einops
import numpy as np
import torch
import torch.nn.functional as F

VOCAB_SIZE = 64
PAD_ID = 0
BOS_ID = 1

# Survey-wide image blocks must not acquire new meanings.
_RESERVED_IMAGE_BASES = (2, 20, 53)
# name -> (source, payload band or None for display RGB, first marker ID)
FIXED_IMAGE_CATEGORIES = {
    "galaxies": ("galaxies_with_hats", None, 23),
    "legacy_north_g": ("legacy_north", "des-g", 160),
    "legacy_north_r": ("legacy_north", "des-r", 163),
    "legacy_north_z": ("legacy_north", "des-z", 166),
    "legacy_south_g": ("legacy_south", "des-g", 169),
    "legacy_south_r": ("legacy_south", "des-r", 172),
    "legacy_south_i": ("legacy_south", "des-i", 175),
    "legacy_south_z": ("legacy_south", "des-z", 178),
    "hsc_g": ("hsc", "hsc-g", 181),
    "hsc_r": ("hsc", "hsc-r", 184),
    "hsc_i": ("hsc", "hsc-i", 187),
    "hsc_z": ("hsc", "hsc-z", 190),
    "hsc_y": ("hsc", "hsc-y", 193),
}
_MODALITY_ID_BLOCKS = {
    "spectra": 5,  # begin=5, placeholder=6, end=7
    # ADR 0008 scalar modalities (one-token spans, GMM heads)
    "Z": 8,  # begin=8, placeholder=9, end=10
    "ebv": 11,  # begin=11, placeholder=12, end=13
    "photometry": 14,  # begin=14, placeholder=15, end=16
    **{name: base for name, (_, _, base) in FIXED_IMAGE_CATEGORIES.items()},
}


def assign_modality_token_ids(modalities: list[dict]) -> list[dict]:
    """Return config dicts with stable, non-overlapping three-id blocks.

    Fixed blocks are reserved regardless of which modalities are enabled.
    Other modalities consume free blocks in declaration order.
    """
    if not modalities:
        raise ValueError("at least one modality is required")
    names = [str(modality.get("name")) for modality in modalities]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate modality names: {names}")

    used = {PAD_ID, BOS_ID}
    used.update(
        token_id
        for base in (*_RESERVED_IMAGE_BASES, *_MODALITY_ID_BLOCKS.values())
        for token_id in range(base, base + 3)
    )
    out = []
    next_reserved, next_appended = 17, VOCAB_SIZE
    for raw in modalities:
        modality = dict(raw)
        name = str(modality["name"])
        raw_ids = modality.get("token_ids")
        if raw_ids is None and name in _MODALITY_ID_BLOCKS:
            base = _MODALITY_ID_BLOCKS[name]
            token_ids = (base, base + 1, base + 2)
        elif raw_ids is None:
            while next_reserved <= 59 and any(
                token_id in used for token_id in range(next_reserved, next_reserved + 3)
            ):
                next_reserved += 3
            if next_reserved <= 59:
                base = next_reserved
                next_reserved += 3
            else:
                while any(
                    token_id in used
                    for token_id in range(next_appended, next_appended + 3)
                ):
                    next_appended += 3
                base = next_appended
                next_appended += 3
            token_ids = (base, base + 1, base + 2)
        else:
            try:
                token_ids = tuple(int(token_id) for token_id in raw_ids)
            except (TypeError, ValueError) as error:
                raise ValueError(f"modality {name!r} has invalid token_ids") from error
            if len(token_ids) != 3 or token_ids != tuple(
                range(token_ids[0], token_ids[0] + 3)
            ):
                raise ValueError(
                    f"modality {name!r} token_ids must be three consecutive ids"
                )
        if name in _MODALITY_ID_BLOCKS:
            if token_ids != modality_token_ids(name):
                raise ValueError(f"modality {name!r} must use token_ids {modality_token_ids(name)}")
        elif overlap := used.intersection(token_ids):
            raise ValueError(
                f"modality {name!r} token_ids collide at "
                f"{sorted(overlap)}"
            )
        used.update(token_ids)
        modality["token_ids"] = list(token_ids)
        out.append(modality)
    return out


def required_vocab_size(modalities: list[dict]) -> int:
    """Embedding rows required by an explicitly tokenized modality config."""
    return max(
        VOCAB_SIZE, 1 + max(max(modality["token_ids"]) for modality in modalities)
    )


def modality_token_ids(name: str) -> tuple[int, int, int]:
    """Return the frozen block for a built-in modality."""
    base = _MODALITY_ID_BLOCKS[name]
    return base, base + 1, base + 2


def special_token_map(modalities: list[dict] | None = None) -> dict[str, int]:
    """Human-readable token map for built-in or config-carried modalities."""
    tokens = {"<|pad|>": PAD_ID, "<|bos|>": BOS_ID}
    blocks = (
        {name: (base, base + 1, base + 2) for name, base in _MODALITY_ID_BLOCKS.items()}
        if modalities is None
        else {
            str(modality["name"]): tuple(modality["token_ids"])
            for modality in modalities
        }
    )
    for name, (begin, placeholder, end) in blocks.items():
        tokens[f"<|begin_{name}|>"] = begin
        tokens[f"<|{name}|>"] = placeholder
        tokens[f"<|end_{name}|>"] = end
    return tokens


def patchify_image(flux: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(c, h, w) image cube -> (n_patches, patch_size*patch_size*c) tokens."""
    c, h, w = flux.shape
    if h % patch_size or w % patch_size:
        raise ValueError(
            f"image size {(h, w)} not divisible by patch size {patch_size}"
        )
    return einops.rearrange(
        flux, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
    )


def unpatchify_image(
    patches: torch.Tensor, patch_size: int, channels: int, side: int
) -> torch.Tensor:
    """Inverse of :func:`patchify_image`."""
    n_side = side // patch_size
    return einops.rearrange(
        patches,
        "(h w) (p1 p2 c) -> c (h p1) (w p2)",
        h=n_side,
        w=n_side,
        p1=patch_size,
        p2=patch_size,
        c=channels,
    )


def patchify_spectrum(
    flux: torch.Tensor, lam: torch.Tensor, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D spectrum -> (n_patches, patch_size) tokens + per-patch mean wavelength.

    Flux is zero-padded up to a multiple of patch_size (as in astroPT); the
    per-patch position is the mean of the *unpadded* wavelengths in the patch
    (padded bins would drag the last patch's position toward zero).
    """
    (w,) = flux.shape
    pad_w = (patch_size - w % patch_size) % patch_size
    padded_flux = F.pad(flux, (0, pad_w))
    padded_lam = F.pad(lam, (0, pad_w))
    patches = einops.rearrange(padded_flux, "(n p) -> n p", p=patch_size)
    lam_patches = einops.rearrange(padded_lam, "(n p) -> n p", p=patch_size)
    counts = einops.rearrange(
        F.pad(torch.ones_like(lam), (0, pad_w)), "(n p) -> n p", p=patch_size
    ).sum(dim=1)
    lam_mean = lam_patches.sum(dim=1) / counts.clamp(min=1)
    return patches, lam_mean


def unpatchify_spectrum(patches: torch.Tensor, length: int) -> torch.Tensor:
    """Inverse of :func:`patchify_spectrum` (drops the zero padding)."""
    return einops.rearrange(patches, "n p -> (n p)")[:length]


def normalize_wavelength(lam: torch.Tensor) -> torch.Tensor:
    """Normalize wavelength in Angstroms to ~[0, 1] (astroPT convention)."""
    return (lam - 3000.0) / 7000.0


def spiral_index(n: int) -> np.ndarray:
    """Spiral index array of side length n (astroPT local_datasets._spiral)."""
    a = np.arange(n * n)
    b = a.reshape((n, n))
    indices = []
    for _ in range(n, 0, -2):
        indices.extend(b[0, :])
        indices.extend(b[1:, -1])
        indices.extend(b[-1, :-1][::-1])
        indices.extend(b[1:-1, 0][::-1])
        b = b[1:-1, 1:-1]
    a[indices] = a.copy()
    a = abs(a - n * n + 1)
    return a.reshape((n, n))


def spiralise(patches: torch.Tensor) -> torch.Tensor:
    """Reorder raster-order ViT patches into spiral order (astroPT Fig. 8)."""
    n = math.isqrt(len(patches))
    assert n * n == len(patches), "spiralise needs a square-rootable patch count"
    indices = spiral_index(n).reshape(-1)
    out = torch.empty_like(patches)
    out[indices] = patches
    return out


def antispiralise(patches: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`spiralise`."""
    n = math.isqrt(len(patches))
    assert n * n == len(patches), "antispiralise needs a square-rootable patch count"
    indices = spiral_index(n).reshape(-1)
    return patches[indices]
