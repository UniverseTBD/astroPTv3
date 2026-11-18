"""Fixed-template sample rendering for converted HF checkpoints.

The template record fixes the token skeleton and positions; keeping the
record(s) and the sampling seed fixed across a run's checkpoints means the
rendered panels differ only through the model weights. Model-side only
(ADR 0015 §6): template RECORDS are provided by the caller; the source-
backed ``load_template_record`` is deferred to a future LSDB evaluation seam.

Modes:
- ``unconditional``:      sample every span the template has (jetformer only).
- ``image-to-spectra``:   teacher-force the image tokens, sample the spectra
                          span (jetformer only).
- ``spectra-to-images``:  teacher-force the spectra, then sample the image bands.
- ``reconstruct``:        one-step teacher-forced predictions for every span.

Physical image bands are rendered individually using their configured
inverse normalization; display imagery stays RGB and spectra use the
source-specific f_ν map.
"""

import math
from pathlib import Path

import numpy as np
import torch

from ..data.band_registry import RGB_CHANNELS, physical_inverse
from ..data.packing import ObjectSequencer
from ..data.spectral import spectral_inverse
from ..generation import generate, reconstruct
from ..tokenization import antispiralise, unpatchify_image, unpatchify_spectrum

MODES = ("unconditional", "image-to-spectra", "spectra-to-images", "reconstruct")


def _present_modalities(sequencer, record):
    return [m for m in sequencer.registry.modalities.values()
            if m.family != "scalar" and record.get(m.record_keys[0]) is not None]


def build_template(sequencer, record: dict, mode: str, *, band: str | None = None):
    """The mode's template: pinned span order, no scalar spans.

    Explicit order (not the sequencer's shuffle) so the conditioning span
    always precedes the generated one under a causal mask. Scalar spans are
    omitted (ADR 0008): a Z span ahead of a generated span would leak the
    label into the panel, and scalar-free templates keep the panels
    comparable with pre-0008 runs.
    """
    present = _present_modalities(sequencer, record)
    if band is not None:
        if mode not in ("unconditional", "spectra-to-images"):
            raise ValueError(f"band selection is not supported in {mode!r} mode")
        if not any(m.name == band and m.band is not None for m in present):
            raise ValueError(f"template has no image band {band!r}")
        present = [m for m in present if m.name == band or (
            mode == "spectra-to-images" and m.family == "spectrum"
        )]
    first_family = "spectrum" if mode == "spectra-to-images" else "image"
    present.sort(key=lambda m: m.family != first_family)
    order = [m.name for m in present]
    return sequencer.build(record, modality_order=order, include_scalars=False, only_modalities=order)


def save_image_png(
    values: np.ndarray,
    path: Path,
    title: str,
    truth: np.ndarray | None = None,
    truth_label: str = "truth",
):
    """[n, C, H, W] -> a grayscale (C=1) or RGB (C=3) PNG grid.

    With ``truth`` [C, H, W] the ground-truth panel leads the grid.
    """
    import matplotlib.pyplot as plt

    panels = ([(truth_label, truth)] if truth is not None else []) + [
        (f"sample {i}", img) for i, img in enumerate(values)
    ]
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3 * len(panels), 3.2), squeeze=False
    )
    for ax, (label, img) in zip(axes[0], panels, strict=True):
        pixels = img[0] if img.shape[0] == 1 else img.transpose(1, 2, 0)
        lo, hi = np.percentile(pixels, [1, 99])
        ax.imshow(np.clip((pixels - lo) / (hi - lo + 1e-8), 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize="small")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_spectra_png(
    flux: np.ndarray,
    lam: np.ndarray,
    path: Path,
    title: str,
    truth: np.ndarray | None = None,
    truth_label: str = "truth",
):
    """[n, W] flux + [W] wavelength -> one subplot per spectrum, stacked.

    With ``truth`` [W] the ground-truth spectrum leads the stack (mirroring
    the image grid); shared axes keep the panels comparable.
    """
    import matplotlib.pyplot as plt

    panels = ([(truth_label, truth)] if truth is not None else []) + [
        (f"sample {i}", f) for i, f in enumerate(flux)
    ]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10, 2 * len(panels)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, (label, f) in zip(axes[:, 0], panels, strict=True):
        color = "black" if truth is not None and f is truth else None
        ax.plot(lam, f, lw=0.7, color=color)
        ax.set_title(label, fontsize="small")
        ax.set_ylabel("f$_\\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\\AA^{-1}$]")
    axes[-1, 0].set_xlabel("wavelength [$\\AA$]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def default_modes(config) -> list[str]:
    """The sampling modes a checkpoint supports."""
    modes = ["unconditional"]
    families = {m.family for m in config.modality_registry().modalities.values()}
    if {"image", "spectrum"} <= families:
        modes.extend(["image-to-spectra", "spectra-to-images"])
    return modes


def sample_template(
    model,
    template,
    mode: str,
    *,
    n: int = 4,
    temperature: float = 1.0,
    argmax: bool = False,
    generator: torch.Generator | None = None,
) -> dict:
    """Run one sampling mode against a template: ``{name: [n, T, D]}``."""
    if mode == "reconstruct":
        return {m: v.unsqueeze(0) for m, v in reconstruct(model, template).items()}
    if mode in ("image-to-spectra", "spectra-to-images"):
        registry = model.config.modality_registry()
        target = "spectrum" if mode == "image-to-spectra" else "image"
        context = "image" if target == "spectrum" else "spectrum"
        gen_modalities = {name for name in template.masks if registry.get_config(name).family == target}
        given = {name for name in template.masks if registry.get_config(name).family == context}
        if not given or not gen_modalities:
            raise ValueError(f"{mode} needs a template carrying both modalities")
        if max(template.masks[name].nonzero()[0].item() for name in given) > min(
            template.masks[name].nonzero()[0].item() for name in gen_modalities
        ):
            first = "spectra" if context == "spectrum" else "images"
            raise ValueError(f"{mode} needs a {first}-first template (build_template)")
    elif mode == "unconditional":
        gen_modalities = set(template.masks)
    else:
        raise ValueError(f"unknown mode {mode!r} (expected one of {MODES})")
    return generate(
        model,
        template,
        gen_modalities,
        n=n,
        temperature=temperature,
        argmax=argmax,
        generator=generator,
    )


def render_sampled_tokens(
    model,
    record: dict,
    template,
    sampled: dict,
    *,
    out_dir: Path,
    tag: str,
    show_truth: bool,
    truth_label: str = "truth",
) -> dict:
    """Invert sampled tokens and write one PNG per modality: ``{name: path}``."""
    registry = model.config.modality_registry()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = {}
    for name, tokens in sampled.items():
        mod = registry.get_config(name)
        if mod.family not in ("image", "spectrum"):
            continue  # scalar spans have no panel rendering
        tokens = tokens.cpu().float()  # bf16 -> f32 so matplotlib/numpy can ingest
        png = out_dir / f"{name}_{tag}.png"
        if mod.family == "image":
            bands: list[str] = [*RGB_CHANNELS] if mod.band is None else [mod.band]
            patches_per_side = math.isqrt(tokens.shape[1])
            if patches_per_side**2 != tokens.shape[1]:
                raise ValueError(f"{name} has a non-square patch count")
            side = patches_per_side * mod.patch_size
            # the checkpoint's own arcsinh knee, so the inverse matches
            # the normalization its training data went through
            divisor = model.config.image_norm_divisor
            # a spiral checkpoint's tokens (sampled AND template) are in
            # spiral order; unpatchify expects raster, so undo the exact
            # order the checkpoint trained in (ADR 0004)
            spiral = model.config.spiral

            def to_pixels(
                t,
                *,
                spiral=spiral,
                patch_size=mod.patch_size,
                channels=len(bands),
                side=side,
            ):
                return unpatchify_image(
                    antispiralise(t) if spiral else t, patch_size, channels, side
                )

            imgs = physical_inverse(
                torch.stack([to_pixels(t) for t in tokens]), bands, divisor=divisor
            )
            truth = None
            if show_truth:
                truth = physical_inverse(
                    to_pixels(template.values[name].float()), bands, divisor=divisor
                ).numpy()
            save_image_png(
                imgs.numpy(), png, f"{name} {tag}", truth=truth, truth_label=truth_label
            )
        elif mod.family == "spectrum":
            lam = np.asarray(record[mod.record_keys[0]]["lambda"])
            lam_t = torch.as_tensor(lam, dtype=torch.float32)
            # the checkpoint's own arcsinh knee, mirroring the image path
            divisor = model.config.spectra_norm_divisor
            flux = spectral_inverse(
                torch.stack([unpatchify_spectrum(t, len(lam)) for t in tokens]),
                lam_t,
                divisor=divisor,
                source=mod.source,
            )
            truth = (
                spectral_inverse(
                    unpatchify_spectrum(template.values[name].float(), len(lam)),
                    lam_t,
                    divisor=divisor,
                    source=mod.source,
                ).numpy()
                if show_truth
                else None
            )
            save_spectra_png(
                flux.numpy(),
                lam,
                png,
                f"{name} {tag}",
                truth=truth,
                truth_label=truth_label,
            )
        pngs[name] = png
    return pngs


def sample_checkpoint(
    checkpoint,
    records: list[dict],
    *,
    modes: list[str] | None = None,
    band: str | None = None,
    n: int = 4,
    temperature: float = 1.0,
    seed: int = 0,
    out_dir: Path,
    device=None,
    step: int | None = None,
) -> dict:
    """Sample + render every (record, mode) pair from a converted checkpoint.

    ``records`` are pre-loaded template records (load once per sweep, not per
    step). A fresh seeded generator per (record, mode) keeps the sampling
    noise identical at every checkpoint, so a run's panels differ only
    through the model. ``step``, when given, is folded into the PNG tag so
    filenames are self-identifying across steps. Returns
    ``{"{mode}/{name}/{object_id}": str(png)}``.
    """
    from transformers import AutoModel

    import astropt3  # noqa: F401  -- registers the Auto classes

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(checkpoint).to(device=device, dtype=dtype).eval()
    if modes is None:
        modes = ["unconditional"] if band is not None else default_modes(model.config)
    sequencer = ObjectSequencer(model.config)

    pngs = {}
    for record in records:
        families = {m.family for m in _present_modalities(sequencer, record)}
        if not families:
            continue
        for mode in modes:
            if mode in ("image-to-spectra", "spectra-to-images") and not {"image", "spectrum"} <= families:
                continue
            template = build_template(sequencer, record, mode, band=band)
            generator = torch.Generator(device=device).manual_seed(seed)
            sampled = sample_template(
                model,
                template,
                mode,
                n=n,
                temperature=temperature,
                generator=generator,
            )
            # unconditional samples aren't tied to the template's object, but
            # its record still makes a useful visual reference
            rendered = render_sampled_tokens(
                model,
                record,
                template,
                sampled,
                out_dir=out_dir,
                tag=(f"step{step}_" if step is not None else "")
                + f"{mode}_{template.object_id}_seed{seed}",
                show_truth=True,
                truth_label="truth (reference)" if mode == "unconditional" else "truth",
            )
            for name, png in rendered.items():
                pngs[f"{mode}/{name}/{template.object_id}"] = str(png)
    return pngs
