# JetFormer flow lineage: definitions versus selected training paths

**Status:** the cross-patch flow recommended below is now the sole flow (`astropt3.flows.JetFormerFlow`, [jetformer-flow.md](jetformer-flow.md)); the patchwise `TinyFlow1D` was removed. The rest of this note is the historical record.

## Conclusion

**There is not one interchangeable ancestor JetFormer.** The verified `sogol_branch` contains three distinct flow implementations:

1. A **patchwise `TinyFlow1D` helper unselected by the inspected trainers**, with four MLP couplings and hidden width 128.
2. The **integrated AstroPT training path**, selecting a four-coupling **image-space convolutional `TinyFlow2D`**, not that helper.
3. A later **canonical ancestor checkpoint-training script**, selecting **16 full-image ViT couplings**, each with four width-512 transformer blocks.

Thus the statement “ported from Sogol” supports source attribution, but **does not establish preservation of the trained architecture or replication of published JetFormer**. Local history supplies a particularly important finding: AstroPTv3 explicitly documents changing `TinyFlow2D` image-space processing to patch-token processing because packed batches expose flat `(N,D)` values. That is an acknowledged architectural adaptation, not just a rename.

This note combines pinned ancestor sources with local v3 source/history inspection. The canonical configuration, flow implementation, training caller, and checkpoint-attribution commit were independently re-read when verifying the research. No architecture change is authorized by this research.

## Repository and revision provenance

GitHub identifies the canonical public repository as [`Smith42/astroPT`](https://api.github.com/repos/Smith42/astroPT), default branch `main`. The [branch API](https://api.github.com/repos/Smith42/astroPT/branches?per_page=100) confirms the exact spelling **`sogol_branch`**.

| Revision | GitHub author/committer date, UTC | Significance |
|---|---|---|
| [`41110b266f15853bf217dac25df1d9074bfef6d8`](https://github.com/Smith42/astroPT/commit/41110b266f15853bf217dac25df1d9074bfef6d8) | 2025-11-11 18:31:54 | SogolSanjaripour / GitHub; adds standalone convolutional `scripts/jetformer/train_jetformer.py`, PR #53. |
| [`aa042aa06ebad71170be2ec2998777a0c399f54b`](https://github.com/Smith42/astroPT/commit/aa042aa06ebad71170be2ec2998777a0c399f54b) | 2025-11-21 08:43:43 | Sogol / Sogol; integrates Jet into AstroPT, adds `src/astropt/jetformer.py` and integration notes. |
| [`f89ef6007835f1c25484e4b161a3ab90434d52eb`](https://github.com/Smith42/astroPT/commit/f89ef6007835f1c25484e4b161a3ab90434d52eb) | 2026-06-05 06:44:35 | Sogol / Sogol; inspected `sogol_branch` tip. Adds canonical `train_jetformer_sogol.py`; commit message says it produced `sogol_checkpoint_step_0079999`. |
| [`fa361ae5b6631c89525429755ed3fc2b2d304eca`](https://github.com/Smith42/astroPT/commit/fa361ae5b6631c89525429755ed3fc2b2d304eca) | 2026-09-28 21:26:41 | Mike Smith / GitHub; inspected `main` tip, documentation PR #80. |

All ancestor source links below pin `f89ef600…`; branch names themselves are mutable. GitHub tree blob identities confirm `src/astropt/jetformer.py` and `src/astropt/model.py` are byte-identical between `aa042aa…` and `f89ef600…` (blobs `4de962fe2c02846fe9490bcc6e5a13c5f55b6f0a` and `adf1fac40191b638072021d8234370c1f8ef91b9`). Thus the integrated convolutional path is not a late replacement of the patchwise helper between those revisions.

**Evidence limit:** the canonical checkpoint attribution is the author's commit statement, not independently validated checkpoint tensors, launch logs, or training results. No exact v3 source pin is established by merely reading this remote repository.

## 1. The patchwise helper exists, but is not the selected ancestor experiment

[`TinyFlow1DConfig`, `CouplingMLP`, and `TinyFlow1D`][flow1d] define defaults `steps=4`, `hidden_dim=128`. For even token dimension D, each conditioner is `Linear(D/2,128) → GELU → Linear(128,D)`, predicting scale/shift for the other feature half; scale is bounded by `1.5*tanh`. Alternating couplings flip which half is transformed.

The flow explicitly reshapes `[B,T,D]` to `[B*T,D]`. Therefore **there is no token-to-token or patch-to-patch mixing** inside it. There is **no explicit zero-output/identity initialization**; linear layers retain their ordinary initialization when this standalone helper is constructed.

Crucially, the [integration notes][integration] call `TinyFlow1D` **“kept for potential future use.”** A textual caller search of the inspected source/training/config files finds its definitions and that note, but no training construction. Its presence therefore supports a possible source of implementation primitives, **not evidence that Sogol trained the patchwise architecture**. It also has no independently selected optimizer, GMM size, or noise curriculum: those belong to callers.

## 2. Integrated AstroPT: convolutional image-space flow

### Actual caller chain

[`scripts/train_jetformer.py`][integrated-config] sets `tokeniser="jetformer"`, then [passes its settings into `GPTConfig` and constructs `GPT`][integrated-build]. [`GPT._init_native_backbone`][model-select] selects `JetformerImageEncoder` for images. That encoder [constructs `TinyFlow2D`, flows raw images, and only then patchifies][encoder].

[`jetformer_run.job`][job] launches this integrated script with `--init_from=resume --batch_size=32 --compile=False`; it **does not launch the later canonical ViT script**. Because it resumes external weights/configuration, the job file is not proof of the complete historical experiment state. The script also supports CLI/config overrides; the values below are checked-in scratch-path selections, not recovered run metadata.

### Spatial behavior and configuration

- [`TinyFlow2D`][flow2d] selects **four** affine couplings by default; the trainer explicitly selects four too.
- Every conditioner contains **three 3×3 convolutions**, channels `C → 128 → 128 → 2C`, with ReLU between them. Each conditioner has a **7×7 local receptive field**; stacked couplings propagate spatial information further. This mixes neighboring pixels **across eventual patch boundaries**, unlike independent patch MLPs, but is **not global/full-image attention**.
- Despite the helper name `checker_mask`, the actual mask alternates **channels**, uniformly across spatial positions: `m[:, ::2, :, :] = 1`. It is not a spatial checkerboard.
- No convolutional output zeroing or identity initialization is implemented. `GPT`'s [recursive initializer][model-init] changes `Linear`/`Embedding`, not `Conv2d`.
- Selected defaults: RGB 256×256, patch 16, hence **768 latent dimensions per token**, **K=4 GMM components**, native GPT **12 layers / width 768 / 12 heads**. The [GMM head][model-select] predicts the full patch dimension; there is **no learned/prior factor-out split**.
- [Noise][encoder]: linear standard-deviation schedule **0.1 → 0**, with the [global iteration fraction][schedule] propagated into encoders. It perturbs the embedded token copy only; clean tokens were cached first. Uniform pixel dequantization is separate.
- [Optimizer/config][integrated-config]: shared flow/trunk **AdamW**, peak **6e-4**, betas **(0.9,0.95)**, weight decay **0.1** on tensors of rank ≥2, zero otherwise; [one shared optimizer][optimizer], not a special flow optimizer. [Scheduler][integrated-schedule]: 2,000-step warmup, cosine to **6e-5** by **29,700** steps, then floor. Global clipping is **1.0**. Five epochs normally determine `max_iters`; this does not automatically change the separately configured LR-decay endpoint.

**Objective discrepancy worth preserving in the record:** the encoder stores `last_logdet = logdet.detach()` before the loss subtracts it. Thus this integrated path does **not** backpropagate the log-determinant term to the flow. Also, its [loss][integrated-loss] predicts tokens `1:` but subtracts the whole-image logdet. These are source facts, not evidence about the cause of any current v3 training behavior.

## 3. Original standalone convolutional alternative

[`scripts/jetformer/train_jetformer.py`][standalone] directly constructs `JetFormerLite → TinyFlow`, with the same image-space three-convolution conditioner/channel-mask design. Its selected `CFG()` values differ from the integrated trainer:

- Four flow couplings, patch 16 / full **768-dimensional** latent targets, **K=4**.
- AR transformer **16 layers / width 1536 / 24 heads**.
- **200 epochs**, shared AdamW at constant **3e-4**, weight decay **0.01**, no explicit beta override, global clipping **1.0**.
- Linear latent-noise **0.1 → 0**. Unlike the integrated encoder, this script overwrites `tokens` before making both embeddings **and targets**, so both are noisy. Its logdet is not detached.

Its [`train()`][standalone-train] instantiates `CFG()` and `JetFormerLite(cfg)` directly. These are selected script settings, not just otherwise-unused class defaults.

## 4. Canonical ancestor checkpoint script: substantially larger ViT flow

[`scripts/jetformer/train_jetformer_sogol.py`][canonical] is standalone: its `train()` instantiates `CFG() → JetFormer(cfg)`, rather than importing `astropt.model.GPT` or `TinyFlow1D`.

| Property | Selected configuration and implementation |
|---|---|
| Flow depth | **16 couplings** from `CFG.flow_steps`, passed explicitly. `ViTFlow.__init__` defaults to **32**, but that default is **not selected**. |
| Conditioner | Four `TransformerEncoderLayer`s, width **512**, **8 heads**, FFN width **2048**, zero dropout, pre-norm GELU. No attention mask: **full attention across all 1,024 image patches**. |
| Patch shape | RGB 256×256, patch **8**, latent width **192**, 1,024 tokens. |
| Initialization | Conditioner `out_proj` weights/biases explicitly **zeroed**. Affine scale/shift start at zero. Each coupling also applies a fixed random feature permutation, so the complete flow is initially a **volume-preserving permutation**, not literally the identity map. |
| Flow size | **212,511,744 parameters**, calculated from the selected layer dimensions including biases and positional embeddings; source-derived count, not checkpoint inspection or model execution. |
| GMM / factor-out | **256 components**, predicts **all 192 dimensions**. No 128-dimensional AR / remaining-Gaussian-prior split. |
| AR backbone | Local `AstroPTBackbone`: **12 layers / width 768 / 12 heads**, RMSNorm, rotary attention, gated GELU MLP. Not v3's SmolLM3/Qwen trunk. |
| Optimizer | One AdamW over **all model parameters**, **1e-4** LR, weight decay **1e-4**, betas **(0.9,0.95)**. No flow/trunk LR separation. |
| LR / duration | **80,000 iterations**; **10,000-step warmup**, then cosine decay to zero at the horizon. Global clipping **0.5**. |
| Noise | Cosine RGB sigma **64/255 → 0**, before flow with clamping; cosine latent sigma **0.3 → 0** after flow. Both decay over the complete horizon, training-only. Uniform dequantization is additional. |

Sources: [selected `CFG`][canonical-cfg], [couplings/initialization][canonical-flow], [model/noise/loss][canonical-forward], [optimizer/schedule/train caller][canonical-train].

The canonical forward overwrites `z` with latent noise before both embedding and target selection, so **targets are also noisy**. It retains differentiable logdet, but again scores `z[:,1:]` against whole-image logdet. Its zero-output initialization is therefore a genuine distinction from both small-flow implementations; “all ancestor flows start at identity” would be inaccurate.

Compared with [published JetFormer](jetformer-optimization-literature.md), this canonical script is closer in **full-image ViT coupling structure**, but still differs in selected depth, mixture size, factor-out, LR, and clipping. It is not itself an exact published-model reproduction.

## 5. Main branch and implications for the v3 attribution

The inspected [main tree](https://api.github.com/repos/Smith42/astroPT/git/trees/fa361ae5b6631c89525429755ed3fc2b2d304eca?recursive=1) has **no Jet implementation/training files** corresponding to these branch paths. Its native [model configuration](https://github.com/Smith42/astroPT/blob/fa361ae5b6631c89525429755ed3fc2b2d304eca/src/astropt/model.py#L352-L375) lists AIM/affine tokenization; AION has its separate training path. Main's [v3 overview](https://github.com/Smith42/astroPT/blob/fa361ae5b6631c89525429755ed3fc2b2d304eca/docs/source/astropt3.rst#L19-L31) explicitly attributes v3 tokenization to Sogol, but gives **no source commit or exact architecture**. That prose cannot resolve which branch implementation was copied.

Local history: v3 commit `f8f69d920797bd5c9251d792690010b8c9df03c7` dated 2026-07-13 introduced its `CouplingMLP`/`TinyFlow1D`; current `astro/src/astropt3/modalities.py:211–220` explains that v2 used `TinyFlow2D`, whereas v3 flows patch tokens because its collator supplies flat token values. Together with the verified caller chain, this supports **“adapted Sogol's flow/GMM approach for packed multimodal tokens,” not “preserved Sogol's trained flow architecture.”** Similar class names alone were not used to establish ancestry.

### Advice, not an implementation decision

First distinguish the goal: reproducing published JetFormer, reproducing the later ancestor checkpoint architecture, and retaining v3's inexpensive packed multimodal adaptation are different experiments. For an architecture ablation, **restoring within-object spatial context** is a substantive independent variable; merely increasing GMM components or patchwise MLP width does not restore it. Full-image flows would require an object-aware processing seam, not an unexamined hyperparameter change inside the existing flat-token flow.

For the current project, the recommended next architecture experiment is a
**within-object, within-modality cross-patch flow**, using the canonical ViT
conditioner and its zero-output initialization as the reference. Keep the
current patchwise model as the control; hold the head and optimizer recipe
fixed for a flow-only ablation rather than simultaneously changing mixture
count, LR ratio, noise, and latent factor-out. This would be a cross-patch
flow experiment, not an exact reproduction of the ancestor's complete recipe.

Two constraints matter before implementing it:

- Packed token arrays combine multiple objects. Flow attention must be
  isolated to the intended object's modality span, with spatial positions
  preserved; unrestricted attention over the flat array is incorrect.
- A verbatim ~212M-parameter image flow already exceeds the nominal 70M
  model budget, before duplicating it for other modalities. The project's
  [fixed scaling ladder and learning-dynamics charter](../EXPERIMENTS.md#0-charter--mmu-streaming--astropt)
  require explicit parameter/compute accounting. Benchmark an image-only
  control/reference pair before assuming a four-modality version is practical.

Do not transfer the ancestor's uint8 dequantization, RGB `[0,1]` clamping,
or `64/255` noise scale onto calibrated astronomical flux. Preserve v3's
physical normalization, first-patch prediction, and differentiable logdet.
Assess held-out likelihood, samples/noise statistics, relevant downstream
probes, and compute cost before promoting a variant to the default.

No current-run diagnosis follows from this lineage review. Historical v3 low-LR/noise/flow observations were not investigated or used as causal evidence. No checkpoints were downloaded, models executed, environments changed, or training launched. Source search covered the branch/main trees and relevant source, training, config, job, and documentation text (122 files; unrelated Euclid/COLM scripts excluded). Only this research note was written.

[flow1d]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/src/astropt/jetformer.py#L9-L123
[flow2d]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/src/astropt/jetformer.py#L126-L227
[integration]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/src/astropt/JETFORMER_INTEGRATION.md#L831-L842
[encoder]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/src/astropt/model.py#L304-L362
[model-select]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/src/astropt/model.py#L509-L547
[model-init]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/src/astropt/model.py#L642-L648
[schedule]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/src/astropt/model.py#L481-L493
[optimizer]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/src/astropt/model.py#L1132-L1164
[integrated-loss]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/src/astropt/model.py#L714-L740
[integrated-config]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/scripts/train_jetformer.py#L127-L221
[integrated-build]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/scripts/train_jetformer.py#L449-L495
[integrated-schedule]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/scripts/train_jetformer.py#L681-L723
[job]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/jetformer_run.job
[standalone]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/scripts/jetformer/train_jetformer.py#L27-L79
[standalone-train]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/scripts/jetformer/train_jetformer.py#L597-L659
[canonical]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/scripts/jetformer/train_jetformer_sogol.py
[canonical-cfg]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/scripts/jetformer/train_jetformer_sogol.py#L77-L150
[canonical-flow]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/scripts/jetformer/train_jetformer_sogol.py#L327-L395
[canonical-forward]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/scripts/jetformer/train_jetformer_sogol.py#L567-L619
[canonical-train]: https://github.com/Smith42/astroPT/blob/f89ef6007835f1c25484e4b161a3ab90434d52eb/scripts/jetformer/train_jetformer_sogol.py#L679-L783
