# AstroPTv3 labbook

This directory is the project labbook: durable notes recording what was
planned, run, and learned on each part of AstroPTv3. The authoritative
phase history (formerly the plan) with all fixed decisions is
[`../EXPERIMENTS.md`](../EXPERIMENTS.md) Part I; the
agent-facing rules are in [`../../AGENTS.md`](../../AGENTS.md). Code-level
architecture and the operational run guide live here as **reference** docs;
everything else is a **workstream** entry.

Convention: filenames are `lowercase_snake_case.md`. New workstream entries
are appended to the relevant section below with a one-line status so the
index stays auditable.

---

## Reference (stable background)

| Doc | What it covers |
|-----|----------------|
| [`architecture.md`](architecture.md) | Model design: SmolLM3 body + per-modality regression heads, tokenization, packing, two-implementation contract (HF release + nanotron training). Read first when touching the model. |
| [`training.md`](training.md) | Operational guide: environments, data, launching, checkpoint/resume, eval, and the traps already hit. The counterpart to `architecture.md`. |

---

## Workstreams (plan → run → diagnosis, in order)

### JetFormer tokeniser (additive `tokeniser: jetformer` path)

Per-modality `JetFormerFlow` ([`jetformer-flow.md`](jetformer-flow.md);
[lineage](jetformer-flow-lineage.md)) + `GMMHead`, exact patch-space likelihood
`mean(NLL_GMM(z) − logdet)`; standardization skipped so the record→token
map stays invertible. Tracked as `astro-phase5`; gates in
`tests/test_nanotron_gpu.py`.

The plan/run-guide/diagnosis entries were deleted 2026-09-28 once their
conclusions lived in the ADRs and the lab book: the noise/convergence
diagnosis is [`../EXPERIMENTS.md`](../EXPERIMENTS.md) §12; the first 70M
run and the low-LR follow-up are recorded in the lab book's phase notes.
Recover the deleted files from git history if a derivation is needed.

### Physical image normalization (port of galactiktok `feat/norm`)

Replace the data-driven Platonic-Universe asinh stretch with galactiktok's
physical, band-registry-keyed normalization for the image modality.
Spectra unchanged. Additive; gated on `uv run pytest` + the
`train_smoke --assert-decrease` smoke gate.

Implemented — see `data/band_registry.py` and `architecture.md`; the
plan entry was deleted 2026-09-28 (git history has it). Source of truth
was `../galactiktok` branch `feat/norm`.

---

## MMU corpus expansion

| Entry | Kind | Status |
| --- | --- | --- |
| [`evidence/adr0013-anchor-scout-2026-08-04/README.md`](evidence/adr0013-anchor-scout-2026-08-04/README.md) | Evidence | Pinned North/South scout; both capped inconclusive, owner selected North first and retained South. |
| [`evidence/adr0013-source-spokes-2026-08-05/README.md`](evidence/adr0013-source-spokes-2026-08-05/README.md) | Evidence | Bounded schema-v2 index, live stream, transform, and transferred-byte smoke for all four North spokes. |

The five-spoke North graph and its match-index/source-graph run configs
(`*-north-5spoke-*.yaml`, `bench-north-5spoke-b0.yaml`) are retired: ADR 0015
is a hard cutover to `lsdb.streams.InfiniteStream` over a single uncrossmatched
catalog, which has no source graph, no match index, and no five-spoke
composition to run against. See
[`adr/0015-lsdb-infinite-stream-training.md`](adr/0015-lsdb-infinite-stream-training.md).
The expansion plan, its test plan, the crossmatch research and the per-run
handoff notes were deleted 2026-08-06 once ADRs 0011–0014 carried their
conclusions — recover them from git history if a derivation is needed.

---

## ADRs (decision records, `adr/`)

| ADR | Status |
| ----- | -------- |
| [`adr/0001-jetformer-inverse-variance-loss.md`](adr/0001-jetformer-inverse-variance-loss.md) | Rejected — ivar-weighted loss does not transfer to the jetformer likelihood head. |
| [`adr/0002-ivar-weighted-huber-loss.md`](adr/0002-ivar-weighted-huber-loss.md) | Proposed (Parked), moot — ivar-weighted Huber for the affine tokeniser, which is since removed. |
| [`adr/0003-checkpoint-samples-in-eval-sidecar.md`](adr/0003-checkpoint-samples-in-eval-sidecar.md) | Accepted — sample panels render in the eval sweep, never in the trainer. |
| [`adr/0004-spiral-token-order-for-imagery.md`](adr/0004-spiral-token-order-for-imagery.md) | Accepted — centre-out spiral patch order for images (default on). |
| [`adr/0005-include-spectra-from-non-crossmatched-desi.md`](adr/0005-include-spectra-from-non-crossmatched-desi.md) | Accepted — ZWARN==0 spectrum-only rows train too; generalised by ADR 0008's span order. |
| [`adr/0006-stream-mmu-upstream.md`](adr/0006-stream-mmu-upstream.md) | Accepted (closed 2026-08-04) — stream MMU live from the hub; the local reshard is deleted. |
| [`adr/0007-physical-spectra-normalization.md`](adr/0007-physical-spectra-normalization.md) | Accepted — DESI spectra → AB nanomaggies, `arcsinh(f_ν/10 nMgy)` (`data/spectral.py`), the symmetric counterpart of `band_registry.py`. |
| [`adr/0008-scalar-modalities.md`](adr/0008-scalar-modalities.md) | Accepted — one-token scalar modalities (`Z`, `ebv`, `photometry`) with GMM heads and the uniform random span order. |
| [`adr/0011-skim-crossmatch-scans.md`](adr/0011-skim-crossmatch-scans.md) | Accepted (amended 2026-08-04 to crossmatch-only), retired by ADR 0015 — the match index it defined is deleted; kept as the historical record. |
| [`adr/0012-gate-mmu-streaming-throughput.md`](adr/0012-gate-mmu-streaming-throughput.md) | Closed by ADR 0014 — gated throughput work on measured byte economics. |
| [`adr/0013-legacy-centred-mmu-expansion.md`](adr/0013-legacy-centred-mmu-expansion.md) | Superseded by ADR 0015 — the source graph it built is retired; never formally accepted before the cutover. |
| [`adr/0014-byte-efficiency-and-mfu-programme.md`](adr/0014-byte-efficiency-and-mfu-programme.md) | Accepted — the gated byte-efficiency + MFU programme: replay (built, ~3x) gated for production, per-band A/B, projection shipped, cache conversation opened, concurrency fixes rejected. |
| [`adr/0015-lsdb-infinite-stream-training.md`](adr/0015-lsdb-infinite-stream-training.md) | Accepted for an experimental branch — hard cutover to `lsdb.streams.InfiniteStream` over a single uncrossmatched catalog; retires the match index, mmu-stream, and the ADR 0013 source graph. |
