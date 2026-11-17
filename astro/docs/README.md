# AstroPTv3 docs

Reference docs for the current code. Experiment results are in the lab book,
[`../EXPERIMENTS.md`](../EXPERIMENTS.md); agent rules are in
[`../AGENTS.md`](../AGENTS.md). Docs describe the current code only; git
history is the change record.

---

## Reference

| Doc | What it covers |
|-----|----------------|
| [`architecture.md`](architecture.md) | Model design: SmolLM3 body + per-modality regression heads, tokenization, packing, two-implementation contract (HF release + nanotron training). Read first when touching the model. |
| [`training.md`](training.md) | Operational guide: environments, data, launching, checkpoint/resume, eval, and the traps already hit. The counterpart to `architecture.md`. |
| [`jetformer-flow.md`](jetformer-flow.md) | The per-modality `JetFormerFlow` + `GMMHead` (exact patch-space likelihood `mean(NLL_GMM(z) − logdet)`); [lineage](jetformer-flow-lineage.md). |

Physical image normalization is in `data/band_registry.py`; see `architecture.md`.

---

## ADRs (decision records, `adr/`)

| ADR | Status |
| ----- | -------- |
| [`adr/0001-jetformer-inverse-variance-loss.md`](adr/0001-jetformer-inverse-variance-loss.md) | Rejected — ivar-weighted loss does not transfer to the jetformer likelihood head. |
| [`adr/0003-checkpoint-samples-in-eval-sidecar.md`](adr/0003-checkpoint-samples-in-eval-sidecar.md) | Accepted — sample panels render in the eval sweep, never in the trainer. |
| [`adr/0004-spiral-token-order-for-imagery.md`](adr/0004-spiral-token-order-for-imagery.md) | Accepted — centre-out spiral patch order for images (default on). |
| [`adr/0005-include-spectra-from-non-crossmatched-desi.md`](adr/0005-include-spectra-from-non-crossmatched-desi.md) | Accepted — ZWARN==0 spectrum-only rows train too; generalised by ADR 0008's span order. |
| [`adr/0006-stream-mmu-upstream.md`](adr/0006-stream-mmu-upstream.md) | Accepted (closed 2026-08-04) — stream MMU live from the hub; the local reshard is deleted. |
| [`adr/0007-physical-spectra-normalization.md`](adr/0007-physical-spectra-normalization.md) | Accepted — DESI spectra → AB nanomaggies, `arcsinh(f_ν/10 nMgy)` (`data/spectral.py`), the symmetric counterpart of `band_registry.py`. |
| [`adr/0008-scalar-modalities.md`](adr/0008-scalar-modalities.md) | Accepted — one-token scalar modalities (`Z`, `ebv`, `photometry`) with GMM heads and the uniform random span order. |
| [`adr/0012-gate-mmu-streaming-throughput.md`](adr/0012-gate-mmu-streaming-throughput.md) | Closed by ADR 0014 — gated throughput work on measured byte economics. |
| [`adr/0014-byte-efficiency-and-mfu-programme.md`](adr/0014-byte-efficiency-and-mfu-programme.md) | Accepted — the gated byte-efficiency + MFU programme: replay (built, ~3x) gated for production, per-band A/B, projection shipped, cache conversation opened, concurrency fixes rejected. |
| [`adr/0015-lsdb-infinite-stream-training.md`](adr/0015-lsdb-infinite-stream-training.md) | Accepted for an experimental branch — hard cutover to `lsdb.streams.InfiniteStream` over a single uncrossmatched catalog. |
