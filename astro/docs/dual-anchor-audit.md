# Dual-anchor live sky audit — #58

**South adds multimodal objects, not just more images.** In a 5′ cone outside every North HATS partition, 124 of 227 South objects had a partner within 1″. One South object had all four partners. Conversely, keeping *only* South objects outside the North footprint would throw away real coverage: in a second cone, 83 of 121 South objects were unpaired with North even though all 121 lay in North-occupied cells. These are two deliberately selected fields, not a corpus-wide match-rate estimate.

## Inputs and scope

Read live on 2026-11-17 UTC. Baseline: [astroPTv3 `54b7093`, active 70m recipe](https://github.com/UniverseTBD/astroPTv3/blob/54b7093884705ce265030d9de382aa3e10d840e2/astro/configs/nanotron/astropt3-70m.yaml); question: [#58](https://github.com/UniverseTBD/astroPTv3/issues/58). Public, immutable Hugging Face revisions below. South is the full DR10 product, not the mini dataset. “North” means the pinned SSL catalogue, **not** a declination cut. Each link points to the exact revision.

| Role | Dataset | Revision | Main partitions | Default margin |
| --- | --- | --- | ---: | ---: |
| North | [UniverseTBD/mmu_ssl_legacysurvey_north](https://huggingface.co/datasets/UniverseTBD/mmu_ssl_legacysurvey_north/tree/f634744d3c44dd4fde0dee3172d4887c5e3c31c0) | `f634744d3c44dd4fde0dee3172d4887c5e3c31c0` | 5,488 | 10″ |
| South | [hugging-science/mmu_legacysurvey_dr10_south_21](https://huggingface.co/datasets/hugging-science/mmu_legacysurvey_dr10_south_21/tree/982f321784d8daf2a5f983eb3ec560f31b90d667) | `982f321784d8daf2a5f983eb3ec560f31b90d667` | 74,440 | 10″ |
| SDSS | [UniverseTBD/mmu_sdss_sdss](https://huggingface.co/datasets/UniverseTBD/mmu_sdss_sdss/tree/da175ca9bb931f4301ab950431922ba9f99089ba) | `da175ca9bb931f4301ab950431922ba9f99089ba` | 789 | 10″ |
| DESI | [UniverseTBD/mmu_desi_edr_sv3](https://huggingface.co/datasets/UniverseTBD/mmu_desi_edr_sv3/tree/9fd88ba48233cb9857701ce802a7eade2d4c4a88) | `9fd88ba48233cb9857701ce802a7eade2d4c4a88` | 306 | 10″ |
| HSC | [hugging-science/mmu_hsc_pdr3_wide_21](https://huggingface.co/datasets/hugging-science/mmu_hsc_pdr3_wide_21/tree/7de6b92fd99097207a00b9c42f2d2862d62662bf) | `7de6b92fd99097207a00b9c42f2d2862d62662bf` | 1,422 | 10″ |
| GWH | [Smith42/galaxies-with-hats, train](https://huggingface.co/datasets/Smith42/galaxies-with-hats/tree/c0188b776c4ce6312a805a04cbc25c891a075933/train) | `c0188b776c4ce6312a805a04cbc25c891a075933` | 1,555 | **none** |

## Whole-catalogue coverage (metadata, not row crossmatches)

Read each `collection.properties`, the primary table's `partition_info.csv`, and each advertised default margin's `hats.properties` (`hats_margin_threshold`). Read primary `skymap.fits` for both anchors and the three outer partners. From FITS extension 1 use `T` counts at implicit pixel indices, or scatter `VALUE` at `PIXEL` into a zero array for HSC. All five maps are celestial NESTED HEALPix, NSIDE 1024 (order 10). A nonzero cell contains a source; it need not be filled with sources. Area per cell = `(4π × (180/π)²)/(12 × 1024²)` = 0.00327849 deg². For comparison, expand every adaptive primary leaf `(o,p)` to the half-open interval `[p × 4^(10−o), (p+1) × 4^(10−o))` to get the coarser partition footprint. Do not use a leaf intersection as an object or even an occupied-cell intersection.

| Region | Partition cells (deg²) | Occupied cells (deg²) | North rows | South rows | Eligible occupied cells | Eligible North / South rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| North only | 1,683,587 (5,519.62) | 1,569,458 (5,145.45) | 14,079,967 | 0 | 12,226 | 121,513 / 0 |
| Both | 41,153 (134.92) | 10,748 (35.24) | 94,236 | 280,986 | 3,310 | 32,004 / 92,389 |
| South only | 5,200,433 (17,049.57) | 4,728,145 (15,501.18) | 0 | 122,904,984 | 64,228 | 0 / 1,817,240 |

Each column uses its own masks: for example, “South only” in the occupied columns means `S>0 & N==0`, not “outside every North partition”. Sum the anchor counts in that mask, not the number of cells. The row columns recover the published totals: 14,174,203 North and 123,185,970 South. The overlap row counts are **not duplicate counts**; sources on opposite sides of a cell boundary can also be within 1″.

“Eligible” further requires at least two of `SDSS>0`, `DESI>0`, `HSC>0` in the cell. This is an order-10 footprint measure for one anchor at a time, following the baseline `min_catalogs: 3` policy (anchor plus outer partners; neither GWH nor the alternative anchor adds a vote). It is **not** a requirement that two partners match an individual row. No dual-stream sampling weights or realized draw fractions are inferred from these counts.

## Projected row audit: how to reproduce

1. Use two circular apertures of radius **300 arcsec** about these exact **NESTED order-8 pixel centres**: `436132` (150.1171875°, +2.238685960842514°; South-only, near COSMOS) and `300463` (355.078125°, +0.4476278310304714°; overlap). The first targeted a multi-partner South field. The second was the middle sorted order-8 pixel among 1,099 whose area intersects North, South, SDSS, HSC and GWH leaf footprints. A distinct North-only cone was not read; North-only sky/row totals above come from the full maps.
2. For each catalogue, get order-10 cells intersecting a **302″** cone (`cdshealpix.cone_search(..., depth=10, flat=True)`) and select overlapping main leaves by the intervals above. No margin rows are counted. Parquet paths relative to the primary table are `dataset/Norder={o}/Dir={10000*(p//10000)}/Npix={p}.parquet`. The primary directory is the dataset basename except GWH, which is `train/train` relative to its repo root. The following is the complete leaf manifest (`order/pixel`, dash = no intersecting leaf):

   | Pixel | North | South | SDSS | DESI | HSC | GWH |
   | --- | --- | --- | --- | --- | --- | --- |
   | 436132 | — | 7/109033 | 4/1703 | 6/27258 | 6/27258 | 4/1703 |
   | 300463 | 5/4694 | 7/75115 | 4/1173 | — | 6/18778 | 4/1173 |

3. Use `HfFileSystem.open(block_size=131072, cache_type="readahead")`, `pyarrow.parquet.ParquetFile(pre_buffer=False)`. Read row groups whose inclusive `_healpix_29` min/max interval, shifted right 38 bits, intersects the selected order-10 cells (read the group if statistics are absent). Project only `ra, dec, object_id` (`dr8_id` for GWH) plus the nine GWH columns below. Cast coordinates to float64 degrees; retain rows within 302″ by great-circle distance. Do not join by raw ID or fetch images/spectra.
4. For each anchor within **300″**, independently select the nearest right row in each catalogue with spherical separation **≤1″** (`astropy.coordinates.angular_separation` on radians). Search all candidates, not just the same pixel; no ties or anchors with multiple candidates in one partner catalogue occurred here. To count unused *core partner* rows, include anchor centres out to **301″** and their possible partners out to **302″**. For North/South duplicates count all position pairs within 1″, then check multiplicity and literal IDs separately. All overlap pairs in the halo had both ends in the core. No image equality test is implied.
5. Below, count all core rows **before** the footprint cut to separate sky coverage from match yield. Applying the order-10 eligibility mask keeps 227/227 South in the South-only core, and 15/38 North and 46/121 South in the overlap core. These are direct positional results, not a transcript of a stochastic LSDB draw. Versions: LSDB `0.11.3.dev3+g8aa1e63f9`, PyArrow `26.0.1`, huggingface_hub `1.22.0`, cdshealpix `0.8.1`, Astropy `8.0.1`, NumPy `2.4.6`, pandas `3.0.6`. The repository's `telemetry.install_byte_probe` counted **11,860,913 bytes in 95 returned ranges** across the 10 listed Parquet files; separate metadata reads (including FITS) and HTTP overhead are excluded. These are projection bytes, not training-loader throughput.

## Results at 1 arcsec

| Core / anchor | Anchors | Any partner | No partner | SDSS | DESI | HSC | GWH |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| South-only / South | 227 | 124 | 103 | 2 | 49 | 94 | 15 |
| Overlap / North | 38 | 9 | 29 | 4 | 0 | 3 | 6 |
| Overlap / South | 121 | 25 | 96 | 4 | 0 | 19 | 6 |

Partner columns count anchors with a match and are not additive. Each is also the number of distinct selected right rows for that path in this sample. In the South-only cone a single South anchor, `1501p022-6629` at (150.0758396755°, +2.2033528567°), selects all of:

| Partner | Local identifier (SDSS numeric portion shown) | Separation (arcsec) |
| --- | --- | ---: |
| SDSS | `564183642964781056` | 0.04251 |
| DESI | `39627841603635442` | 0.02266 |
| HSC | `43158734788125132` | 0.04071 |
| GWH | `343928_1789` | 0.02266 |

All five rows are in order-10 cell `6978115`. There is **no North primary leaf** anywhere in the expanded cone; the absence of a North anchor here is not a failed nearest-neighbour lookup. Outer partner singletons in some other stream are not equivalent to these associations.

### Unmatched partners and ownership

Each entry below is the number of core right rows **not selected** on that path / total core right rows. For SDSS, DESI and HSC these are candidate outer-only rows within a path's selected footprint; GWH is left-only, so an unused GWH row never becomes a standalone record. An absent catalogue is shown as a dash.

| Core / path | SDSS | DESI | HSC | GWH |
| --- | ---: | ---: | ---: | ---: |
| South-only / South | 0/2 | 52/101 | 15/109 | 0/15 |
| Overlap / North | 3/7 | — | 17/20 | 0/6 |
| Overlap / South | 3/7 | — | 1/20 | 0/6 |

* **Anchor identity:** 38 one-to-one North/South positional pairs in the overlap, separations 0.00918–0.83807″; zero exact coordinate pairs and zero shared literal anchor IDs. Example: North `12439705` and South `3551p002-3458`, 0.04170″ apart. No repeated local IDs within any one catalogue in either core. Under this positional criterion the overlap core has 38 shared, **0 North-only and 83 South-only** objects. The South-only core adds 227 more. All 83 unpaired overlap-core South objects are in North-*occupied* order-10 cells; rejecting an entire shared cell would lose them, including 16 South/HSC associations.
* **Partner identity:** in the overlap core both paths select the same 4 SDSS, 3 HSC and 6 GWH rows. Three SDSS rows and one HSC row are unused on **both** paths; 16 HSC rows are unused on North but attached on South. Independent outer joins can therefore duplicate both singleton/singleton and singleton/attached rows; a row unmatched against one anchor is not necessarily unowned. These are conditional counts in the geometric aperture, not a count of emitted LSDB singletons.
* **Ownership trade-off:** the configured policy gives North the shared eligible footprint. Of the 46 eligible South objects in this overlap cone, 31 have no North counterpart within 1″ (three match HSC). These are foregone South objects, not duplicates removed. Source-qualified IDs and a shared spatial split protect provenance and evaluation, but do not reconstruct those associations. Recovering them would require object-level arbitration for anchors and for partners reachable through both paths. Position pairs here are *candidate* same objects, not globally validated IDs; the audit does not resolve crowded-field ambiguity or certify identical image bytes.

### Scalar coverage on matched anchors

Counts are of **usable anchor spans**, not merely non-null GWH rows. Apply the baseline `ObjectSequencer._scalar_value` / `scalar_registry.MISSING_SENTINELS` rule: require finite values and exclude `-99` for photo-z mass, SFR and sSFR. One South-only mass was finite but `-99`. No extra science cuts were imposed. The denominator for coverage in the entire anchor stream is the anchor count in the table above, **not** the matched count here. The overlap columns describe the same six GWH rows twice, not twelve unique measurements.

| GWH source field (record keys have a `gwh_` prefix) | South-only / South (15 GWH) | Overlap / North (6 GWH) | Overlap / South (6 GWH) |
| --- | ---: | ---: | ---: |
| redshift | 15 | 6 | 6 |
| mass_med_photoz | 14 | 6 | 6 |
| total_sfr_median | 1 | 5 | 5 |
| total_ssfr_median | 1 | 5 | 5 |
| smooth-or-featured_smooth_fraction | 15 | 6 | 6 |
| smooth-or-featured_featured-or-disk_fraction | 15 | 6 | 6 |
| disk-edge-on_yes_fraction | 1 | 3 | 3 |
| has-spiral-arms_yes_fraction | 0 | 1 | 1 |
| merging_merger_fraction | 15 | 6 | 6 |
| Any / all nine usable | 15 / 0 | 6 / 1 | 6 / 1 |

## Boundary and interpretation limits

At the revisions audited above, five catalogues had 10″ margins and GWH had none. The baseline left join could therefore miss a GWH match across a partition boundary. The coverage counts above describe that baseline, not the [current training footprint](training.md). The buffered projection searches every intersecting primary partition and therefore gives a local geometric reference, not proof that the training join recovers every reference pair. Neither a global GWH boundary-loss rate nor catalogue-wide identity multiplicities can be inferred from these two cones.

This is a sky and association audit. It does not validate image decoding, assess photometric agreement, choose a sampling ratio, or measure object lengths, packing, worker RSS or pilot learning. Those require decoded stream draws or the separate training pilot, not projected ID and coordinate columns.
