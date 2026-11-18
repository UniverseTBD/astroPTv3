# Legacy Survey DR10 South: decoder contract (#58)

**South has its own filter categories.** `legacy_south_{g,r,i,z}` select planes from `legacy_south_image` (decoder `decode_legacy_south_image`) by band label. Each has its own token block and model modules, distinct from the corresponding North bands. See [the image token contract](architecture.md#image-categories-and-checkpoints).

## What is actually on disk?

Use [`hugging-science/mmu_legacysurvey_dr10_south_21@982f321784d8daf2a5f983eb3ec560f31b90d667`][pin]. The pinned [collection][collection] names a default **10 arcsec** [margin][margin]; the [primary table][primary] has **123,185,970** rows and `moc_sky_fraction=0.41656` (HATS coverage, not the fraction matching another survey). `_healpix_29` is a spatial index, **not an object ID**. `ra`, `dec` are float64 degrees. `object_id` is the string `BRICKNAME-OBJID`; keep the string verbatim and qualify with the anchor source (`legacy_south:2462m200-369`) when forming stream IDs. Do not infer overlap or identity from equal bare IDs.

Inspected the schema and all four rows of [order 10, pixel 11359295][shard], 2,802,723 bytes. Row 0: `2462m200-369`, (246.11103396°, −20.12371631°), `EBV=0.886696`, `FLUX_G=0.0236653`, `FLUX_I=2.11022`. Units below come from the [DR10 coadd files][files], [Tractor catalogue definitions][tractor] and [MMU cutout builder][build]; Arrow stores numeric types, not a per-pixel units string.

| Nested `image` field | Physical contract |
|---|---|
| `band` | List of strings, exactly `des-g, des-r, des-i, des-z`, in this order. |
| `flux`, `ivar` | Lists of float32 planes, each `(4,160,160)` cube. Calibrated coadd flux in nanomaggies/pixel (AB zero point 22.5); inverse variance of those flux values, not a standard deviation. |
| `scale`, `psf_fwhm` | Four float32 values, respectively arcsec/pixel (0.262 in every band) and arcsec. Row 0 PSF widths: 1.2570, 1.1989, 1.1649, 1.6706. Seeing is metadata, not a flux multiplier. |
| `mask` | Four boolean 160×160 planes. **True = clean**, not bad. The builder clears pixels with any of bits 0–7, 11, 14, 15; the [MMU exporter][export] repeats the clean sky mask for each band. This is not the top-level `object_mask` (segmentation) or the `rgb`/`blobmodel` preview. |

Row 0 has 13,820 true mask pixels per band out of 25,600, with positive ivar everywhere. Using only ivar would admit 11,780 flagged pixels per plane. Reproduce directly with `pytest -m network tests/test_legacy_south.py`; the canary reads the pinned shard, not a dataset-viewer preview.

## Decode and pack

Project `image.{band,flux,mask,ivar,scale}`. Reject unexpected bands/order, shape, nonboolean mask or wrong scale. A usable pixel is `mask & isfinite(flux) & isfinite(ivar) & (ivar > 0)`. Bad values are replaced with zero before normalization and carried in a **separate** `fill_mask` (True = fill); a valid zero or negative flux is a measurement. The sequencer crops flux and mask together (`[32:128,32:128]`), applies the checkpoint's physical band normalization (identity rescale, bright clamp, `asinh(flux / image_norm_divisor)`), and uses the existing deterministic channel-noise fill only on invalid pixels. Ivar gates coverage; it does not reweight the loss. North keeps its existing zero-as-no-data behavior. Each filter yields 144 patch-8 tokens of 64 floats and two delimiters, for **585** tokens in a South-only image record including BOS.

## Selection and scalar quality

The [MMU selection][build] requires extinction-corrected `22.5 − 2.5 log10(FLUX_Z / MW_TRANSMISSION_Z) < 21`, at least one observation in each optical band, `TYPE != PSF`, and none of [MASKBITS][bits] 0–7, 11, 14, 15 at the object's location. The image builder skips incomplete cutouts. These are **source selection cuts**, not per-field uncertainty checks and not guarantees about other pixels or WISE detections.

The shard exposes **flat target** float32 `EBV` (mag), `FLUX_{G,R,I,Z,W1,W2,W3,W4}` (observed nMgy, *not* divided by MW transmission), `SHAPE_R` (arcsec), `SHAPE_E1/E2` (dimensionless). Its nested `catalog.*` lists describe neighbouring objects, padded with zeros; they are not the central target's scalar labels. No top-level `NOBS_*`, `FLUX_IVAR_*`, `MW_TRANSMISSION_*`, `MASKBITS`, redshift, fiberflux or psfdepth fields appear. Pixel ivar is **not** uncertainty on Tractor photometry. In row 1 (`2462m200-311`), `FLUX_G=-0.429724` nMgy despite selection: do not impose a positive-only flux cut or treat negative flux as a universal sentinel.

**No South scalar targets are enabled by this decoder.** Non-finite/null labels are absent, not zero; do not claim per-band S/N or reliable WISE detections without source quality fields. Any future scalar enablement needs its own quality predicate and physical transform; keep a four-band photometry vector distinct from North's frozen three-band token. These four rows establish format and mask behavior, not corpus-wide scalar completeness or overlap.

[pin]: https://huggingface.co/datasets/hugging-science/mmu_legacysurvey_dr10_south_21/tree/982f321784d8daf2a5f983eb3ec560f31b90d667
[collection]: https://huggingface.co/datasets/hugging-science/mmu_legacysurvey_dr10_south_21/resolve/982f321784d8daf2a5f983eb3ec560f31b90d667/collection.properties
[primary]: https://huggingface.co/datasets/hugging-science/mmu_legacysurvey_dr10_south_21/resolve/982f321784d8daf2a5f983eb3ec560f31b90d667/mmu_legacysurvey_dr10_south_21/hats.properties
[margin]: https://huggingface.co/datasets/hugging-science/mmu_legacysurvey_dr10_south_21/resolve/982f321784d8daf2a5f983eb3ec560f31b90d667/mmu_legacysurvey_dr10_south_21_10arcs/hats.properties
[shard]: https://huggingface.co/datasets/hugging-science/mmu_legacysurvey_dr10_south_21/blob/982f321784d8daf2a5f983eb3ec560f31b90d667/mmu_legacysurvey_dr10_south_21/dataset/Norder%3D10/Dir%3D11350000/Npix%3D11359295.parquet
[files]: https://www.legacysurvey.org/dr10/files/
[tractor]: https://www.legacysurvey.org/dr10/catalogs/
[bits]: https://www.legacysurvey.org/dr10/bitmasks/
[build]: https://github.com/MultimodalUniverse/MultimodalUniverse/blob/4e53d93d29a67fdf56fba6a3cb596d451c1b3da9/scripts/legacysurvey/build_parent_sample.py
[export]: https://github.com/MultimodalUniverse/MultimodalUniverse/blob/4e53d93d29a67fdf56fba6a3cb596d451c1b3da9/scripts/legacysurvey/legacysurvey.py
