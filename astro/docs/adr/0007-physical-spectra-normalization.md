# ADR 0007: Physical nanomaggie normalization for the spectra modality

- **Status:** Implemented (accepted 2026-07-18)
- **Date:** 2026-07-17
- **References:**
  - `astro/src/astropt3/data/band_registry.py` — the image-side physical
    normalization this mirrors (rescale → clamp → `arcsinh(flux/0.01 nMgy)`)
  - `astro/src/astropt3/data/packing.py` — `ObjectSequencer._spectra_tokens`,
    the only place spectra flux becomes tokens
  - `astro/src/astropt3/generation.py` + `astro/src/astropt3/eval/samples.py`
    — the decode/render side that needs the inverse
  - [ADR 0001](0001-jetformer-inverse-variance-loss.md) — rejected
    ivar-*weighting* of the jetformer loss; this ADR is a data-side
    change of variables, which ADR 0001's rationale explicitly permits
  - ADR 0002 — Proposed-PARKED; its
    spectra note ("ivar is already in the record") stays valid and
    composes with this ADR if spectra ever move to the Huber head
  - `astro/docs/README.md` ADR section — the "future ADR" note this
    discharges
  - DESI datamodel (spectra coadds): flux in `10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹`
    on a fixed linear grid, https://desidatamodel.readthedocs.io/

## Question

Jetformer spectra enter the flow as raw DESI flux — the only modality
with no physical normalization. On the first spectra-rich run
(`astropt3-70m-jetformer-pilotv2`, ADR 0005 corpus, 33% spectrum-bearing
draws) the imbalance stopped being cosmetic: `spectra_loss` started at
1.28M nats/token, plateaued at ~180 by step 2k with essentially no
further improvement through 10k+ (images: 268 → −110 and still falling),
and `grad_norm` sat 300–800× above `clip_grad` the whole run. What is
the right fixed, invertible, physically meaningful normalization for
DESI spectra tokens?

## Context

- **Why the loss stalls.** The jetformer loss is an exact patch-space
  likelihood; tokens must arrive O(1) for the flow/GMM head and the
  absolute noise curriculum (`jetformer_noise_max: 0.1`) to operate in
  the regime J1–J4 validated. Raw DESI flux spans two orders of
  magnitude object-to-object (measured on `pilot_v2` shards: per-object
  median 0.37–46 in DESI units) with per-pixel sky noise σ = 0.27–2.0
  varying ~7× across the wavelength range. The images modality had the
  identical failure shape before the knee-unit fix (commit 9799730):
  wrong global scale ⇒ logdet-dominated loss, hard clipping every step.
- **Invertibility constraint.** Jetformer configs skip per-patch
  standardization because the exact-likelihood loss needs an invertible
  record→token map (`AGENTS.md`). Any normalization must be a *fixed*
  change of variables using quantities available at decode time.
- **The images precedent** (`band_registry.py`): rescale raw survey flux
  to a common physical unit (LegacySurvey nanomaggies), then
  `arcsinh(flux/knee)` with the knee at the sky-noise scale (0.01 nMgy)
  so noise stays linear and galaxy light log-compresses. Tokens *are*
  flux, in knee units — no per-corpus calibration, no learned statistics.
- **DESI spectra are f_λ; maggies are f_ν.** DESI coadds ship
  `flux` in 10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹ (spectral flux density per unit
  wavelength). The AB/maggie system is per unit frequency:
  1 nMgy = 10⁻⁹ · 3631 Jy = 3.631×10⁻²⁹ erg s⁻¹ cm⁻² Hz⁻¹. The exact
  conversion is `f_ν = f_λ · λ²/c`, per-pixel, using the record's own
  wavelength grid.
- **The DESI grid is fixed.** Verified on `pilot_v2` shards: every
  spectrum lives on the same linear grid, 3600–9824 Å in 7781 bins of
  exactly 0.8 Å. λ at every token index is therefore a *constant of the
  data format*, not a per-object quantity — the λ²-dependent transform
  is invertible even for unconditionally sampled spectra.
- **Measured scales in nMgy** (180 `pilot_v2` spectra, unmasked pixels):
  f_ν p50 = 3.0, p99 = 381, max ≈ 880 nMgy; noise σ p50 = 1.9,
  p99 = 15.7 nMgy. Sanity check: 3 nMgy ⇒ m_AB ≈ 21.3, a typical DESI
  galaxy fiber magnitude — the unit conversion is self-consistent.

## Decision

Normalize DESI spectra to **AB nanomaggies with an arcsinh knee at the
spectroscopic sky-noise scale**, exactly mirroring the image recipe:

```
f_ν(λ) [nMgy] = flux_DESI · λ_Å² · 10⁻¹⁷ / (c_Å/s · 3.631×10⁻²⁹)
             = flux_DESI · λ_Å² · 9.19×10⁻⁸

token = arcsinh( f_ν / K ),   K = 10 nMgy  (`spectra_norm_divisor`)
```

applied in `ObjectSequencer._spectra_tokens` for **both tokenisers**
(affine keeps its per-patch standardization on top, exactly as images
do); inverse `f_λ = sinh(t)·K / (λ²·9.19×10⁻⁸)` on the render path.
Masked pixels (`mask != 0`, folding in `ivar <= 0`) keep today's
behavior: token 0.

With K = 10 nMgy the measured token distribution is: noise p50 ≈ 0.19
(images: ~0.1), signal p50 ≈ 0.30, p99 ≈ 4.3, max ≈ 5.2 (images
ceiling: 11.3) — the same O(1) regime, so the noise curriculum and GMM
head operate as validated, with no retune.

## Rationale

1. **Tokens are physical flux, same system as images.** After this
   change both modalities speak AB-referenced flux in knee units of
   their own noise floor: images 0.01 nMgy (broadband sky), spectra
   10 nMgy (per-0.8 Å fiber sky — ~10³ noisier per pixel, hence the
   ~10³ larger knee; the *design rule* "knee = noise scale" is
   identical). A model relating an object's image to its spectrum sees
   consistent photometry across the `<|end_images|>` boundary.
2. **Exact likelihood stays valid; sampling stays consistent.** The
   transform is a fixed diagonal change of variables with known
   Jacobian, absorbed into the data like the image norm — training
   NLL and `generation.py`'s reverse path see the same map. Because
   the λ grid is a format constant, unconditional and image-to-spectra
   samples invert to physical f_λ with no side information.
3. **Why not ivar in the map** (the "SNR whitening" alternative,
   `arcsinh(flux·√ivar)`): it equalizes wavelength-dependent noise, but
   (a) tokens become dimensionless SNR — losing the shared physical
   flux system with images (the stated requirement); (b) decode-time
   inversion needs a per-object ivar vector the model doesn't generate,
   so unconditional samples are stuck in SNR space; (c) per ADR 0001,
   modeling heteroscedastic noise is the density head's native job —
   the GMM's learned per-patch variance is the ivar. ivar stays what it
   is today: a mask source (and a loss-weight option for a future
   Huber-spectra config, per ADR 0002's spectra note).
4. **One knob, same shape as images.** `spectra_norm_divisor: 10.0`
   lands next to `image_norm_divisor` in `AstroPT3Config`, threaded the
   same way (config → ObjectSequencer → inverse in generation), plus
   the fork config field + HF↔nanotron conversion mapping. No
   per-corpus statistics, nothing to recompute when the corpus grows.

## Consequences

- **Checkpoint break (spectra-bearing runs).** The record→token map
  changes, so all prior spectra-bearing checkpoints are incompatible —
  retrain. Migration note goes in `docs/architecture.md` alongside the
  PU-asinh and knee-unit entries. The in-flight
  `astropt3-70m-jetformer-pilotv2` run completes as the pre-norm
  baseline; its relaunch under this ADR is the A/B.
- **Touch points** (the vertical slice, all mirroring the image-norm
  work): new `spectral_normalize`/`spectral_inverse` +
  `DESI_LAMBDA_GRID` constants in `data/band_registry.py`'s sibling
  module `data/spectral.py`; `_spectra_tokens` applies it;
  `generation.py`/`eval/samples.py` render through the inverse;
  `AstroPT3Config.spectra_norm_divisor`; fork config field +
  `convert_{nanotron_to_hf,hf_to_nanotron}` mapping; synthetic fixtures
  emit DESI-unit flux so smoke training sees the same regime (as done
  for nMgy images); tests for round-trip inversion and token-scale
  bounds.
- **Non-DESI spectra** (future surveys) will need their own
  unit/grid entries — the module keys on the record's survey the way
  `band_registry` keys on band names, and unknown formats raise rather
  than silently pass through.
- **Expected observables on the A/B relaunch:** `spectra_loss` starting
  O(10²) not O(10⁶), trending negative like images; `grad_norm` inside
  or near `clip_grad` after warmup rather than 300–800× above it.
