"""Deterministic offline records matching the verified MMU schemas.

Unit fixture for tests — NOT a training source (ADR 0015). The records mimic:

- ``UniverseTBD/mmu_ssl_legacysurvey_north``: ``image.flux`` float32
  (3, 152, 152) in LegacySurvey nanomaggies (galaxy cores ~0.1 nMgy, sky
  noise ~0.001 nMgy — the real pilot flux scale, so the physical band
  normalization's 0.01 nMgy arcsinh knee lands in the same regime as on
  real data), ``image.band`` = des-g/r/z, plus catalog scalars.
- ``UniverseTBD/mmu_desi_edr_sv3``: ``spectrum`` with 7781-bin ``flux``,
  ``lambda`` (3600-9824 A), ``ivar``, ``lsf_sigma``, ``mask``, plus ``Z``.

Images contain a redshift-correlated Gaussian blob and spectra a continuum +
emission line, so a smoke-trained model has real structure to learn. A
fraction of records are image-only (no ``spectrum``), matching the pilot
corpus where the DESI crossmatch covers only ~1/14 of the images; a
(default-zero) fraction are spectrum-only (no ``image``), matching the
non-crossmatched DESI rows of ``pilot_v2`` (ADR 0005).
"""

import nested_pandas as npd
import numpy as np
import pandas as pd

IMAGE_SIDE = 152
IMAGE_BANDS = ["des-g", "des-r", "des-z"]
SPECTRUM_LENGTH = 7781
LAMBDA_MIN = 3600.0
LAMBDA_MAX = 9824.0

# Explicit Legacy anchor + optional DESI payload for all five tiny modalities.
TINY_CATALOG_STREAM = {
    "catalogs": [
        {
            "name": "legacy",
            "repo": "UniverseTBD/mmu_ssl_legacysurvey_north",
            "decoder": "legacy_image",
            "record_key": "image",
            "scalar_columns": {key: key for key in ("ebv", "flux_g", "flux_r", "flux_z")},
        },
        {
            "name": "desi",
            "repo": "UniverseTBD/mmu_desi_edr_sv3",
            "decoder": "desi_spectrum",
            "record_key": "spectrum",
            "scalar_columns": {"Z": "Z", "ZWARN": "ZWARN"},
        },
    ],
    "how": "outer",
    "min_surveys": 1,
}


def make_record(
    index: int,
    image_only_fraction: float = 0.3,
    spectrum_only_fraction: float = 0.0,
) -> dict:
    """Build one deterministic synthetic record keyed by ``index``.

    A record is image-only with probability ``image_only_fraction``,
    spectrum-only with ``spectrum_only_fraction``, otherwise bimodal. The RNG
    draw order never changes, so ``spectrum_only_fraction=0`` reproduces the
    historical records exactly.
    """
    rng = np.random.default_rng(index)
    z = float(rng.uniform(0.01, 1.5))

    yy, xx = np.mgrid[0:IMAGE_SIDE, 0:IMAGE_SIDE].astype(np.float32)
    cx, cy = IMAGE_SIDE / 2 + rng.uniform(-10, 10), IMAGE_SIDE / 2 + rng.uniform(-10, 10)
    # large smooth structure so most patches are learnable rather than pure
    # sky noise (per-patch standardization turns flat sky into irreducible
    # N(0,1) targets); size still correlates with the redshift proxy
    sigma = 15.0 + 25.0 * z
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    amps = rng.uniform(0.01, 0.1, size=3).astype(np.float32)
    flux = amps[:, None, None] * blob[None, :, :]
    flux += rng.normal(0.0, 0.001, size=flux.shape).astype(np.float32)

    ra = float(rng.uniform(0, 360))
    dec = float(rng.uniform(-1.6, 81.5))
    # drawn in the historical order — healpix BEFORE seeing — because the rng
    # draw order is frozen and hoisting these out of the dict must not swap it
    healpix = int(rng.integers(0, 2**40))
    seeing = float(rng.uniform(1.0, 2.0))
    record = {
        "object_id": f"synth_{index:08d}",
        "ra": ra,
        "dec": dec,
        "_healpix_29": healpix,
        "image": {
            "flux": flux.astype(np.float32),
            # real MMU rows carry ONE fwhm PER BAND; the single historical draw
            # is kept as the base and scaled by fixed factors so the frozen rng
            # order still reproduces every earlier record exactly
            "psf_fwhm": [float(seeing * f) for f in (1.10, 1.00, 0.78)],
            "band": IMAGE_BANDS,
            "scale": 0.262,
        },
        "z_spec": z,
        # ADR 0008 image-catalog scalars, DERIVED rather than drawn (the rng
        # draw order is frozen so historical records reproduce exactly):
        # aperture fluxes summed from the blob itself (nMgy, correlated with
        # the image and z like real photometry), ebv as a smooth sky function
        "flux_g": float(flux[0].sum()),
        "flux_r": float(flux[1].sum()),
        "flux_z": float(flux[2].sum()),
        "ebv": 0.02 + 0.08 * (dec + 1.6) / 83.1,
        # ADR 0014 A8 free scalars, likewise derived. fiberflux is a CENTRAL
        # APERTURE sum, so it correlates with but is not proportional to the
        # total — the concentration signal A8 measured at 0.64-0.67 on real
        # rows is what makes it worth a span. psfdepth is a smooth sky field.
        "fiberflux_g": float(flux[0, 44:52, 44:52].sum()),
        "fiberflux_r": float(flux[1, 44:52, 44:52].sum()),
        "fiberflux_z": float(flux[2, 44:52, 44:52].sum()),
        "psfdepth_g": 617.0 * (1.0 + 0.2 * (dec + 1.6) / 83.1),
        "psfdepth_r": 178.0 * (1.0 + 0.2 * (dec + 1.6) / 83.1),
        "psfdepth_z": 120.0 * (1.0 + 0.2 * (dec + 1.6) / 83.1),
        # the synthetic stream feeds make_record straight to the sequencer and
        # never runs streaming._attach_image, so the per-band seeing has to be
        # flattened here too — same keys the real adapter writes
        **{
            f"psf_fwhm_{band}": float(seeing * factor)
            for band, factor in zip(IMAGE_BANDS, (1.10, 1.00, 0.78))
        },
    }

    u = rng.uniform()
    if u >= image_only_fraction:
        lam = np.linspace(LAMBDA_MIN, LAMBDA_MAX, SPECTRUM_LENGTH, dtype=np.float32)
        # steep continuum: intra-patch slope dominates the noise, so the
        # standardized patch shape is learnable (flat continua standardize
        # to pure noise)
        continuum = 5.0 + 20.0 * (lam - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN)
        line_centre = 6563.0 * (1 + z) / (1 + 0.5)  # keep the line on-grid
        line = 20.0 * np.exp(-((lam - line_centre) ** 2) / (2 * 25.0**2))
        sflux = continuum + line + rng.normal(0.0, 0.05, size=SPECTRUM_LENGTH)
        record["spectrum"] = {
            "flux": sflux.astype(np.float32),
            "lambda": lam,
            "ivar": np.full(SPECTRUM_LENGTH, 1.0 / 0.09, dtype=np.float32),
            "lsf_sigma": np.full(SPECTRUM_LENGTH, 1.0, dtype=np.float32),
            "mask": np.zeros(SPECTRUM_LENGTH, dtype=bool),
        }
        record["Z"] = z
        record["ZWARN"] = False  # DESI reliability flag; gates the Z span (ADR 0008)
        if u < image_only_fraction + spectrum_only_fraction:
            # non-crossmatched DESI row (ADR 0005): a spectrum with no
            # cutout image and no image-catalog scalars
            del record["image"], record["z_spec"]
            del record["flux_g"], record["flux_r"], record["flux_z"], record["ebv"]
            for band in ("g", "r", "z"):
                del record[f"fiberflux_{band}"], record[f"psfdepth_{band}"]
            for band in IMAGE_BANDS:
                del record[f"psf_fwhm_{band}"]

    return record


def record_stream(
    n: int,
    image_only_fraction: float = 0.3,
    start: int = 0,
    spectrum_only_fraction: float = 0.0,
):
    """Yield ``n`` deterministic records starting at ``start``."""
    for i in range(start, start + n):
        yield make_record(
            i,
            image_only_fraction=image_only_fraction,
            spectrum_only_fraction=spectrum_only_fraction,
        )


def legacy_row(index: int) -> dict:
    """One raw uncrossmatched LegacySurvey row, column-shaped like a
    HATS/``InfiniteStream`` pandas row (nested ``image`` dict, one scalar
    column per catalog field, derived keys absent).

    ADR 0015 interim check: this is what
    ``nanotron_loader.decode_legacy_row`` must decode back to the
    ``make_record`` image-side contract.
    """
    record = make_record(index, image_only_fraction=1.0)
    return {
        "object_id": record["object_id"],
        "ra": record["ra"],
        "dec": record["dec"],
        "_healpix_29": record["_healpix_29"],
        "image": {
            "band": list(record["image"]["band"]),
            "flux": record["image"]["flux"].tolist(),
            "psf_fwhm": record["image"]["psf_fwhm"],
        },
        "ebv": record["ebv"],
        "flux_g": record["flux_g"],
        "flux_r": record["flux_r"],
        "flux_z": record["flux_z"],
        "fiberflux_g": record["fiberflux_g"],
        "fiberflux_r": record["fiberflux_r"],
        "fiberflux_z": record["fiberflux_z"],
        "psfdepth_g": record["psfdepth_g"],
        "psfdepth_r": record["psfdepth_r"],
        "psfdepth_z": record["psfdepth_z"],
        "z_spec": record["z_spec"],
    }


def nested_frame(rows: list[dict], nested_fields: tuple[str, ...]) -> "npd.NestedFrame":
    """Pack raw catalog rows into a real ``nested_pandas.NestedFrame``.

    Nested struct columns exercise ``map_rows`` exactly as a HATS partition
    does. Missing nested fields become null elements, as in an unmatched join.
    """
    scalar_cols = sorted({key for row in rows for key in row} - set(nested_fields))
    base = pd.DataFrame([{key: row.get(key) for key in scalar_cols} for row in rows])
    frame = npd.NestedFrame(base)
    for field in nested_fields:
        frame = frame.join_nested([row.get(field) for row in rows], name=field)
    return frame
