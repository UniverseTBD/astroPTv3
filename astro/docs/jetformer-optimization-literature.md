# JetFormer optimization: literature and released implementation

## Sources and conclusion

Michael Tschannen, André Susano Pinto, Alexander Kolesnikov, **JetFormer: An Autoregressive Generative Model of Raw Images and Text**, [arXiv:2411.19722][paper] / ICLR 2025: [v1 **2024-11-29**](https://arxiv.org/html/2411.19722v1), [v2 **2025-05-19**](https://arxiv.org/html/2411.19722v2). Checked both versions’ training recipe and Appendix A/Table 6. Official code is public in Google Research `big_vision`; evidence below is pinned to authors’ [“JetFormer code release” commit][release], `9c006bfbfb443d6da8aa3bbc6a7a380f54fc190b`.

**Bottom line:** released ImageNet and image–text recipes use one optimizer, equal flow/trunk learning rates, and one shared schedule—not a faster-flow multiplier. This is verified from the complete parameter/update path, not inferred from a missing paper setting.

## Verified evidence

- **[Paper §4, “Training recipe”](https://arxiv.org/html/2411.19722v2#S4):** “Normalizing flow (or learnable invertible patch embedding) is trained along with the transformer-backbone end-to-end.” Also: “We use the Adam optimizer with learning rate 10⁻³, decoupled weight decay of 10⁻⁴, β₂ parameter 0.95 and clip the gradient norms to 1.0.” Mathematical notation is transcribed; wording unchanged. The paper alone does **not report** component-specific LR/schedule or clipping scope.
- **One joint update, not alternating optimizers:** [trainer][trainer] inserts `params["params_adaptor"] = params_adaptor` (L266), constructs one optimizer over that tree (L280), then executes `jax.value_and_grad(loss_fn, has_aux=True)` and `updates, opt = tx.update(grads, opt, params)` (L528–531).
- **LR and schedule:** [ImageNet config][imagenet] L161–190 and [image–text config][multimodal] L214–242 specify `config.lr = 0.001` and `('.*', dict(decay_type='cosine', warmup_percent=0.1))`. [Optimizer construction][optimizer] L75–150 applies `lr_mult_txs = [optax.scale(config.lr)]` to the combined tree; neither recipe supplies additional multipliers. Both use Adam with β₂=0.95 and decoupled decay 1e-4. Warmup is linear for **10% of total steps**, then cosine **toward zero**, not a 0.1-peak floor: [scheduler][scheduler] contains `lr = lr * 0.5 * (1. + jnp.cos(jnp.pi * progress))`. Image–text’s 500,000-step default means 50,000 warmup steps.
- **Clipping:** [optimizer][optimizer] L99–105 uses `clip_tx = optax.clip_by_global_norm(clip_norm)` before Adam, masked only to exclude frozen parameters. Thus **global norm 1.0 across trainable flow + trunk**, not separately per group. Frozen `FREEZE_ME` entries are structural [coupling masks][flow], not the entire flow.
- **Pretraining/stop-gradient:** main recipe starts jointly from scratch. However §4 explicitly says “we stop the gradient at the flow output during pretraining” for image-prefix captioning. Image–text sets `stop_grad_nvp_prefix = True`; [trainer][trainer] L414–419 detaches only image-first examples. ImageNet has no such branch enabled. §4.1/Table 2 separately tests Gaussian-prior flow pretraining followed by a frozen flow: FID 11.16 versus 7.84 jointly trained; this is an ablation, not the default.
- **Fine-tuning:** [Appendix A/Table 6](https://arxiv.org/html/2411.19722v2#A1.T6) reports LR **1e-4** for COCO FID and **3e-5** for VQAv2/captioning; separate flow LR and fine-tuning schedule are **not reported**. Released configs are not exact historical-run records: ImageNet uses batch 2048 versus paper’s general 4k; image–text ships dummy COCO data.

### Weight-decay convention: do not copy the raw number blindly

The released [optimizer chain][optimizer] applies the base LR **before** adding
decoupled weight decay, then applies the common schedule to both. Optax's
[`add_decayed_weights`](https://github.com/google-deepmind/optax/blob/v0.2.4/optax/transforms/_adding.py)
adds `wd * parameter` to the incoming update. Thus the released recipe's update
on a decayed parameter is `-schedule(t) * (lr * Adam_update + wd * parameter)`.
PyTorch AdamW instead uses `-lr(t) * (Adam_update + weight_decay * parameter)`.
For the published equal-rate setting `lr=1e-3`, `wd=1e-4` therefore corresponds
to PyTorch `weight_decay=wd/lr=0.1`, **not** `1e-4`, with matching decay masks and
schedules. This is an algebraic conversion, not a reproduction result. The
actual masks and schedules also differ between the projects; the raw
coefficients alone do not establish a 1000-fold decay mismatch.

## Translation to AstroPTv3

[Table 5](https://arxiv.org/html/2411.19722v2#A1.T5)’s flow has **32 coupling blocks, width 512, block depth 4/4/6**, 447M/447M/650M parameters. AstroPTv3 uses **8 couplings, depth 4**, with width `min(hidden_size // 4, 512)` and full attention within each object's modality span. See [the flow architecture and parameter table](jetformer-flow.md). Its likelihood is trained end to end without detachment and aggregated by modality family. Flow capacity and GMM size remain independent experimental variables.

More precisely, [paper §4](https://arxiv.org/html/2411.19722v2#S4) specifies
32 coupling blocks with 4 or 6 **ViT blocks** per conditioner. The released
[`jet.DNN`](https://github.com/google-research/big_vision/blob/9c006bfbfb443d6da8aa3bbc6a7a380f54fc190b/big_vision/models/proj/jet/jet.py#L34-L65)
passes the patch sequence to
[`vit.Encoder`](https://github.com/google-research/big_vision/blob/9c006bfbfb443d6da8aa3bbc6a7a380f54fc190b/big_vision/models/vit.py#L81-L98),
whose self-attention is unmasked: the conditioner sees **all image patches**.
Channel-wise coupling splits do not make this a patch-independent flow.

The paper also uses **1024 mixture components** over **128 autoregressive
latent dimensions**. Its 256×256 RGB images are patchified into 256 tokens
with 768 channels; the other 640 flow-output channels per token are modeled
by a Gaussian prior rather than the autoregressive trunk (§3.2 and Appendix D).
AstroPTv3 uses **4 mixture components** and defaults to predicting every
coordinate. Per-modality `d_ar` opts into the same AR/Gaussian-prior split;
see [Gaussian factor-out](jetformer-flow.md#gaussian-factor-out). Neither this
split nor the mixture count establishes an optimal flow/trunk LR ratio.

The current 1B recipe uses one AdamW peak **3e-4** for all model parameters, with **2000-step warmup + cosine floor 0.1** and global clip1.0. These are not JetFormer-equivalent hyperparameters or architecture; the literature does not establish an optimal flow/trunk LR ratio.

[paper]: https://arxiv.org/abs/2411.19722
[release]: https://github.com/google-research/big_vision/commit/9c006bfbfb443d6da8aa3bbc6a7a380f54fc190b
[trainer]: https://github.com/google-research/big_vision/blob/9c006bfbfb443d6da8aa3bbc6a7a380f54fc190b/big_vision/trainers/proj/jetformer/train.py
[imagenet]: https://github.com/google-research/big_vision/blob/9c006bfbfb443d6da8aa3bbc6a7a380f54fc190b/big_vision/configs/proj/jetformer/jetformer_imagenet2012.py#L161-L190
[multimodal]: https://github.com/google-research/big_vision/blob/9c006bfbfb443d6da8aa3bbc6a7a380f54fc190b/big_vision/configs/proj/jetformer/jetformer_image_text.py#L214-L242
[optimizer]: https://github.com/google-research/big_vision/blob/9c006bfbfb443d6da8aa3bbc6a7a380f54fc190b/big_vision/optax.py#L75-L150
[scheduler]: https://github.com/google-research/big_vision/blob/9c006bfbfb443d6da8aa3bbc6a7a380f54fc190b/big_vision/utils.py#L1070-L1144
[flow]: https://github.com/google-research/big_vision/blob/9c006bfbfb443d6da8aa3bbc6a7a380f54fc190b/big_vision/models/proj/jet/jet.py#L189-L199
