"""AstroPT3 configuration: a SmolLM3 body with continuous-modality I/O."""

from __future__ import annotations

from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

from .data.band_registry import _DIV_FACTOR
from .data.spectral import _DIV_FACTOR as _SPECTRA_DIV_FACTOR
from .modalities import ModalityRegistry
from .tokenization import assign_modality_token_ids, required_vocab_size

# Pilot modalities, pinned to the verified MMU schemas (see plan):
# - images:  (3, 152, 152) flux cubes, center crop 96x96, patch 8 -> 144
#   tokens of 8*8*3 = 192 (max_positions 361 is a harmless ceiling)
# - spectra: 7781-bin DESI spectra, patch 256 -> 31 tokens; position = per-patch
#   mean wavelength, normalized, projected by an affine PositionEmbedder
# - Z / ebv / photometry: ADR 0008 scalar modalities — one-token spans over
#   the catalog scalars the records already carry, GMM-headed like every
#   other modality. The family loss collectively caps all present
#   scalar modalities at weight 0.1.
DEFAULT_MODALITIES = [
    {
        "name": "images",
        "input_size": 192,
        "patch_size": 8,
        "pos_type": "index",
        "max_positions": 361,
        "family": "image",
        "source": "legacy",
        "record_keys": ["image"],
        "token_ids": [2, 3, 4],
    },
    {
        "name": "spectra",
        "input_size": 256,
        "patch_size": 256,
        "pos_type": "continuous",
        "pos_input_size": 1,
        "family": "spectrum",
        "source": "desi",
        "record_keys": ["spectrum"],
        "token_ids": [5, 6, 7],
    },
    {
        "name": "Z",
        "input_size": 1,
        "patch_size": 1,
        "pos_type": "index",
        "max_positions": 1,
        "family": "scalar",
        "source": "desi",
        "record_keys": ["Z"],
        "token_ids": [8, 9, 10],
    },
    {
        "name": "ebv",
        "input_size": 1,
        "patch_size": 1,
        "pos_type": "index",
        "max_positions": 1,
        "family": "scalar",
        "source": "legacy",
        "record_keys": ["ebv"],
        "token_ids": [11, 12, 13],
    },
    {
        # one joint 3-dim span (g, r, z), not three modalities: colour is
        # the physical quantity, and a joint GMM models it directly
        "name": "photometry",
        "input_size": 3,
        "patch_size": 1,
        "pos_type": "index",
        "max_positions": 1,
        "family": "scalar",
        "source": "legacy",
        "record_keys": ["flux_g", "flux_r", "flux_z"],
        "token_ids": [14, 15, 16],
    },
]


def _complete_modalities(modalities: list[dict]) -> list[dict]:
    """Backfill only the five released modalities; new ones self-describe."""
    legacy = {modality["name"]: modality for modality in DEFAULT_MODALITIES}
    completed = []
    for raw in modalities:
        modality = dict(raw)
        defaults = legacy.get(modality.get("name"), {})
        for key in ("family", "source", "record_keys", "token_ids"):
            if key not in modality and key in defaults:
                modality[key] = defaults[key]
        if (
            "family" not in modality
            or "source" not in modality
            or "record_keys" not in modality
        ):
            raise ValueError(
                f"new modality {modality.get('name')!r} must declare family, source, and record_keys"
            )
        completed.append(modality)
    return assign_modality_token_ids(completed)


class AstroPT3Config(SmolLM3Config):
    model_type = "astropt3"

    def __init__(
        self,
        modalities: list[dict] | None = None,
        jetformer_flow_steps: int = 8,
        jetformer_flow_hidden: int = 128,
        jetformer_flow_depth: int = 4,
        jetformer_flow_heads: int = 8,
        jetformer_gmm_k: int = 4,
        jetformer_noise_max: float = 0.1,
        jetformer_noise_min: float = 0.0,
        scalar_gmm_k: int = 5,
        special_token_ce_weight: float = 0.0,
        image_norm_divisor: float = _DIV_FACTOR,
        spectra_norm_divisor: float = _SPECTRA_DIV_FACTOR,
        spiral: bool = True,
        vocab_size: int | None = None,
        hidden_size: int = 512,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        max_position_embeddings: int = 4096,
        rope_theta: float = 100_000.0,
        no_rope_layer_interval: int = 4,
        tie_word_embeddings: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 1,
        **kwargs,
    ):
        raw_modalities = modalities if modalities is not None else DEFAULT_MODALITIES
        self.modalities = _complete_modalities(raw_modalities)
        registry = self.modality_registry()
        for modality in self.modalities:
            modality["d_ar"] = registry.get_config(modality["name"]).d_ar
        required_vocab = required_vocab_size(self.modalities)
        if vocab_size is None:
            vocab_size = required_vocab
        elif vocab_size < required_vocab:
            raise ValueError(
                f"vocab_size={vocab_size} cannot hold modality token id "
                f"{required_vocab - 1}; set vocab_size >= {required_vocab}"
            )
        for value in (jetformer_flow_steps, jetformer_flow_hidden, jetformer_flow_depth, jetformer_flow_heads):
            if type(value) is not int or value <= 0:
                raise ValueError("jetformer flow dimensions must be positive integers")
        if jetformer_flow_hidden % jetformer_flow_heads:
            raise ValueError("jetformer_flow_hidden must be divisible by jetformer_flow_heads")
        if any(m["input_size"] % 2 for m in self.modalities if m["family"] != "scalar"):
            raise ValueError("jetformer flows require an even input_size for patch modalities")
        self.jetformer_flow_steps = jetformer_flow_steps
        self.jetformer_flow_hidden = jetformer_flow_hidden
        self.jetformer_flow_depth = jetformer_flow_depth
        self.jetformer_flow_heads = jetformer_flow_heads
        self.jetformer_gmm_k = jetformer_gmm_k
        self.jetformer_noise_max = jetformer_noise_max
        self.jetformer_noise_min = jetformer_noise_min
        # mixture count of the ADR 0008 scalar GMM heads (photometric-redshift
        # posteriors are multimodal; K=5 is the ADR's unswept starting point)
        self.scalar_gmm_k = scalar_gmm_k
        self.special_token_ce_weight = special_token_ce_weight
        # arcsinh divisor of the physical image normalization; consumed by
        # ObjectSequencer (forward) and scripts/generate.py (inverse), so a
        # checkpoint always normalizes and inverts with the divisor it
        # trained with
        self.image_norm_divisor = image_norm_divisor
        # arcsinh knee (nMgy) of the physical spectra normalization (ADR
        # 0007), the spectra counterpart of image_norm_divisor: consumed by
        # ObjectSequencer (forward) and eval/samples.py (inverse).
        self.spectra_norm_divisor = spectra_norm_divisor
        # center-outward spiral patch order for image tokens (ADR 0004).
        # The field is the single source of truth for the order a checkpoint
        # trained in: ObjectSequencer spiralises iff it is True, and the
        # inverse path (eval/samples.py) antispiralises iff the LOADED
        # checkpoint's config says True.
        self.spiral = spiral
        kwargs["use_cache"] = False  # reload passes it back through kwargs
        super().__init__(  # type: ignore[call-arg]
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,  # type: ignore[call-arg]
            no_rope_layer_interval=no_rope_layer_interval,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def modality_registry(self) -> ModalityRegistry:
        return ModalityRegistry(self.modalities)
