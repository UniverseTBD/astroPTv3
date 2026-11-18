"""AstroPT3: SmolLM3 decoder body with continuous-modality inputs and
per-modality regression heads.

Sequence contract (built by data/packing.py):
- ``input_ids`` [B, T] hold special-token ids everywhere; modality slots carry
  that modality's placeholder id, so ``embed_tokens`` provides a learned
  modality-type embedding at those positions.
- The continuous content is ADDED on top: slot embedding =
  ``embed_tokens(<|m|>) + encoder_m(value) + pos_embed_m(position)``.
- ``position_ids`` restart at 0 at each object; with ``attention_mask=None``
  transformers' ``create_causal_mask`` detects the packed format and applies
  block-diagonal (doc) masking (torch >= 2.6).
- Loss: predictions read one position LEFT of each modality token
  (``<|begin_m|>`` predicts patch 0 — astroPT's ``starts-1`` alignment).
  Each family's loss is one mean over all of its tokens (so a modality
  counts in proportion to its tokens in the batch); image and spectrum
  token NLLs are first divided by the modality's ``input_size`` (nats per
  dimension, so patch size does not set the weight). Present families
  combine at image:spectrum:scalar 1:1:0.1. ``modality_losses`` stay the
  raw per-token means. No loss on special or pad
  tokens unless ``special_token_ce_weight > 0``.
- The jetformer tokeniser (JetFormer/GIVT-style) is the sole regression
  head: patch values pass through a per-modality normalizing flow
  (``flows[m]``) to a latent z that is both embedded and predicted by a
  ``GMMHead``, and the per-modality loss is
  ``mean(NLL_GMM(z) - logdet)`` — an exact likelihood in patch space (may
  be negative). Same left-shift alignment.
- ADR 0008 scalar modalities (``ModalityConfig.scalar``) are GMM-headed
  directly (``scalar_gmm_k`` mixture over the raw normalized value; loss
  is plain ``gmm_nll``, no flow, no logdet — a scalar needs no
  invertibility machinery and its odd dim cannot feed a flow).
"""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.smollm3.modeling_smollm3 import (
    SmolLM3Model,
    SmolLM3PreTrainedModel,
)
from transformers.utils.generic import ModelOutput

from .configuration_astropt3 import AstroPT3Config
from .flows import FlowCoupling, JetFormerFlow, flow_object_ids
from .modalities import (
    Encoder,
    CategoricalHead,
    GMMHead,
    PositionEmbedder,
    gmm_nll,
)
from .tokenization import PAD_ID


@dataclass
class AstroPT3Output(ModelOutput):
    loss: torch.Tensor | None = None
    modality_losses: dict | None = None
    family_losses: dict | None = None
    predictions: dict | None = None
    last_hidden_state: torch.Tensor | None = None
    # per-layer states (embeddings first, HF convention), populated only
    # when forward(output_hidden_states=True); the linear probe pools the
    # central layer from here (astroPT convention)
    hidden_states: tuple | None = None


def left_shift_mask(mask: torch.Tensor) -> torch.Tensor:
    """Positions whose NEXT token is a masked-True position.

    Given a [B, T] boolean modality mask, returns a [B, T] mask that is True
    at position t iff mask[t+1] is True (last column always False). Hidden
    states at these positions predict the modality values at t+1.
    """
    shifted = torch.zeros_like(mask)
    shifted[:, :-1] = mask[:, 1:]
    return shifted


class AstroPT3Model(SmolLM3PreTrainedModel):
    config_class = AstroPT3Config

    def __init__(self, config: AstroPT3Config):
        super().__init__(config)
        self.model = SmolLM3Model(config)
        registry = config.modality_registry()
        self.modality_names = registry.names()
        self.modality_families = {
            name: cast(str, registry.get_config(name).family)
            for name in self.modality_names
        }
        self.modality_input_sizes = {
            name: registry.get_config(name).input_size for name in self.modality_names
        }
        # ADR 0008: scalar modalities are GMM-headed directly and never
        # flow — one-token spans over raw normalized values
        self.scalar_names = {
            name
            for name in self.modality_names
            if registry.get_config(name).family == "scalar"
        }
        patch_names = [n for n in self.modality_names if n not in self.scalar_names]
        self.encoders = nn.ModuleDict(
            {
                name: Encoder(config.hidden_size, registry.get_config(name).input_size)
                for name in self.modality_names
            }
        )
        # Per-modality flow (data -> latent z, with logdet) and GMM head; the
        # head predicts the latent z, and the loss path below stays a single
        # branch since every patch modality goes through it.
        self.flows = nn.ModuleDict()
        for name in patch_names:
            modality = registry.get_config(name)
            self.flows[name] = JetFormerFlow(
                modality.input_size,
                steps=config.jetformer_flow_steps,
                hidden_dim=config.jetformer_flow_hidden,
                depth=config.jetformer_flow_depth,
                num_heads=config.jetformer_flow_heads,
                pos_type=modality.pos_type,
                max_positions=modality.max_positions,
                pos_input_size=modality.pos_input_size,
            )
        self.decoders = nn.ModuleDict(
            {
                name: GMMHead(
                    config.hidden_size,
                    registry.get_config(name).input_size,
                    config.jetformer_gmm_k,
                )
                for name in patch_names
            }
        )
        self.categorical_names = {
            name for name in self.scalar_names if registry.get_config(name).categories
        }
        for name in sorted(self.scalar_names):
            self.decoders[name] = (
                CategoricalHead(config.hidden_size, registry.get_config(name).categories)
                if name in self.categorical_names
                else GMMHead(
                    config.hidden_size,
                    registry.get_config(name).input_size,
                    config.scalar_gmm_k,
                )
            )
        self.pos_embeds = nn.ModuleDict(
            {
                name: PositionEmbedder(config.hidden_size, registry.get_config(name))
                for name in self.modality_names
            }
        )
        if config.special_token_ce_weight > 0:
            self.special_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False
            )
        # Noise-curriculum position (jetformer, astroPT v2 semantics):
        # sigma = noise_max + (noise_min - noise_max) * frac, so frac 0 -> 1
        # anneals noise_max -> noise_min over training. Default 1.0 means
        # sigma = noise_min (= 0 by default): eval and existing checkpoints
        # are unaffected unless the trainer drives the curriculum.
        self.jet_noise_frac = 1.0
        self.post_init()
        for flow in self.flows.values():
            flow.reset_coupling_outputs()

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, FlowCoupling):
            module.reset_permutations()

    def set_jet_noise_frac(self, frac: float):
        self.jet_noise_frac = min(max(frac, 0.0), 1.0)

    def _jet_noise_sigma(self) -> float:
        cfg = self.config
        return (
            cfg.jetformer_noise_max
            + (cfg.jetformer_noise_min - cfg.jetformer_noise_max) * self.jet_noise_frac
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        if not isinstance(value, nn.Embedding):
            raise TypeError("AstroPT3 input embeddings must be nn.Embedding")
        self.model.embed_tokens = value

    def assemble_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        modality_values: dict,
        modality_masks: dict,
        modality_positions: dict,
    ) -> torch.FloatTensor:
        embeds = self.model.embed_tokens(input_ids)
        delta = torch.zeros_like(embeds)
        for name in self.modality_names:
            if name not in modality_masks or not modality_masks[name].any():
                continue
            mask = modality_masks[name]
            values = modality_values[name].to(embeds.dtype)
            content = self.encoders[name](values) + self.pos_embeds[name](
                modality_positions[name]
            )
            delta = delta.index_put((mask,), content.to(embeds.dtype))
        return embeds + delta

    def forward(
        self,
        input_ids: torch.LongTensor,
        modality_values: dict | None = None,
        modality_masks: dict | None = None,
        modality_positions: dict | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        compute_loss: bool = True,
        output_hidden_states: bool = False,
        values_are_latents: bool = False,
        **kwargs,
    ) -> AstroPT3Output:
        if values_are_latents and (compute_loss or self.training):
            raise ValueError("pre-flowed latent values are supported only for loss-free inference")
        modality_values = modality_values or {}
        modality_masks = modality_masks or {}
        modality_positions = modality_positions or {}

        # values -> flow -> latent z once, up front; z is both the embedded
        # input and the GMM target, and logdet joins the loss. The noise
        # curriculum perturbs only the embedded copy (training mode) — the
        # GMM target z and the logdet stay clean.
        flow_logdets = {}
        embed_values = modality_values
        if modality_values and not values_are_latents:
            flowed = {}
            for name in self.modality_names:
                # scalars bypass the flow: their raw normalized value is both
                # the embedded input and the GMM target (ADR 0008)
                if name not in modality_values or name in self.scalar_names:
                    continue
                flow = self.flows[name]
                values = modality_values[name].to(next(flow.parameters()).dtype)
                flowed[name], flow_logdets[name] = flow(
                    values, modality_positions[name],
                    flow_object_ids(position_ids, modality_masks[name]),
                )
            modality_values = {**modality_values, **flowed}
            embed_values = modality_values
            sigma = self._jet_noise_sigma()
            if self.training and sigma > 0:
                embed_values = {
                    **modality_values,
                    **{
                        name: z + sigma * torch.randn_like(z)
                        for name, z in flowed.items()
                    },
                }

        inputs_embeds = self.assemble_inputs_embeds(
            input_ids, embed_values, modality_masks, modality_positions
        )
        body_out = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
        hidden = body_out.last_hidden_state

        predictions = {}
        modality_losses = {}
        family_losses = {}
        loss = None
        if compute_loss:
            family_sums, family_counts = {}, {}
            for name in self.modality_names:
                if name not in modality_masks or not modality_masks[name].any():
                    continue
                mask = modality_masks[name]
                if mask[:, 0].any():
                    raise ValueError(
                        f"modality '{name}' token at sequence position 0 has no "
                        "preceding token to predict it from (missing <|bos|>?)"
                    )
                if name in self.categorical_names:
                    # one-hot target -> class index; cross-entropy, no density
                    logits = self.decoders[name](hidden[left_shift_mask(mask)]).float()
                    token_loss = F.cross_entropy(
                        logits, modality_values[name].argmax(dim=-1), reduction="none"
                    )
                    predictions[name] = torch.softmax(logits, dim=-1)
                elif name in self.scalar_names:
                    # ADR 0008: GMM NLL on the raw normalized scalar — no
                    # flow, no logdet
                    logits_pi, mu, log_sigma = self.decoders[name](
                        hidden[left_shift_mask(mask)]
                    )
                    target = modality_values[name].to(mu.dtype)
                    token_loss = gmm_nll(target, logits_pi, mu, log_sigma)
                    pi = torch.softmax(logits_pi, dim=-1)
                    predictions[name] = (pi.unsqueeze(-1) * mu).sum(dim=-2)
                else:
                    logits_pi, mu, log_sigma = self.decoders[name](
                        hidden[left_shift_mask(mask)]
                    )
                    target = modality_values[name].to(mu.dtype)  # latent z
                    nll = gmm_nll(target, logits_pi, mu, log_sigma)
                    token_loss = nll - flow_logdets[name].to(nll.dtype)
                    # z-space mixture mean as the point prediction; inverse
                    # flow (self.flows[name](z, reverse=True)) maps it back.
                    pi = torch.softmax(logits_pi, dim=-1)
                    predictions[name] = (pi.unsqueeze(-1) * mu).sum(dim=-2)
                modality_losses[name] = token_loss.mean()
                family = self.modality_families[name]
                # patch tokens in nats per dimension; scalar tokens as they are
                pooled = token_loss.float().sum()
                if family != "scalar":
                    pooled = pooled / self.modality_input_sizes[name]
                family_sums[family] = family_sums.get(family, 0.0) + pooled
                family_counts[family] = family_counts.get(family, 0) + token_loss.numel()

            family_weights = {"image": 1.0, "spectrum": 1.0, "scalar": 0.1}
            weighted = hidden.new_zeros(())
            weight_sum = 0.0
            for family in family_weights:
                if family not in family_sums:
                    continue
                family_loss = family_sums[family] / family_counts[family]
                family_losses[family] = family_loss
                weighted = weighted + family_weights[family] * family_loss
                weight_sum += family_weights[family]
            loss = weighted / weight_sum if weight_sum else weighted

            if self.config.special_token_ce_weight > 0:
                logits = self.special_head(hidden[:, :-1])
                targets = input_ids[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size),
                    targets.reshape(-1),
                    ignore_index=PAD_ID,
                )
                modality_losses["special_ce"] = ce
                loss = loss + self.config.special_token_ce_weight * ce

        return AstroPT3Output(
            loss=loss,
            modality_losses=modality_losses,
            family_losses=family_losses,
            predictions=predictions,
            last_hidden_state=hidden,
            hidden_states=getattr(body_out, "hidden_states", None),
        )
