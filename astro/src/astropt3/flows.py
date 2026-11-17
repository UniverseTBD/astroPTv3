"""JetFormer affine flows over one object's complete modality span.

Each coupling is conditioned by a pre-norm Transformer (GELU, no dropout) with
full attention over the span (a ViT conditioner, as in JetFormer). Patches
remain in their existing serialization order.
Fixed feature permutations are derived from seed 42 + stage index, independently
of the training RNG. Like RoPE tables they are nonpersistent buffers: identical
configurations reconstruct identical permutations in HF and Nanotron.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint


def flow_object_ids(position_ids: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
    """Map row-major modality values to their packed objects, not batch rows."""
    if position_ids is None or position_ids.shape != modality_mask.shape:
        raise ValueError("JetFormer flows require packed position_ids matching the modality mask")
    starts = position_ids.reshape(-1) == 0
    return starts.cumsum(0)[modality_mask.reshape(-1)]


class FlowTransformerBlock(nn.Module):
    """TransformerEncoderLayer equations with ordinary, convertible Linear weights."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear2 = nn.Linear(4 * hidden_dim, hidden_dim)

    def forward(self, x, allowed_keys):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.num_heads, width // self.num_heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed_keys)
        x = x + self.attn_out(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.linear2(F.gelu(self.linear1(self.norm2(x))))


class FlowCoupling(nn.Module):
    def __init__(
        self, dim, hidden_dim, depth, num_heads, pos_type, max_positions,
        pos_input_size, stage,
    ):
        super().__init__()
        self.pos_type = pos_type
        self.dim = dim
        self.stage = stage
        generator = torch.Generator(device="cpu").manual_seed(42 + stage)
        permutation = torch.randperm(dim, generator=generator, device="cpu")
        self.register_buffer("permutation", permutation, persistent=False)
        self.register_buffer("inverse_permutation", permutation.argsort(), persistent=False)
        self.in_proj = nn.Linear(dim // 2, hidden_dim)
        self.position = (
            nn.Embedding(max_positions, hidden_dim)
            if pos_type == "index"
            else nn.Linear(pos_input_size, hidden_dim, bias=False)
        )
        self.layers = nn.ModuleList(
            FlowTransformerBlock(hidden_dim, num_heads) for _ in range(depth)
        )
        self.out_proj = nn.Linear(hidden_dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def reset_permutations(self):
        """Restore derived buffers after HF materializes empty tensors from meta."""
        generator = torch.Generator(device="cpu").manual_seed(42 + self.stage)
        permutation = torch.randperm(self.dim, generator=generator, device="cpu")
        self.permutation.copy_(permutation)
        self.inverse_permutation.copy_(permutation.argsort())

    def forward(self, x, positions, allowed_keys, reverse=False):
        if not reverse:
            x = x[..., self.permutation]
        ident, moved = x.chunk(2, dim=-1)
        h = self.in_proj(ident.to(self.in_proj.weight.dtype))
        position_input = (
            positions if self.pos_type == "index" else positions.to(self.position.weight.dtype)
        )
        h = h + self.position(position_input)
        for layer in self.layers:
            h = layer(h, allowed_keys)
        scale, shift = self.out_proj(h).to(x.dtype).chunk(2, dim=-1)
        scale = scale.tanh()
        if reverse:
            moved = (moved - shift) * torch.exp(-scale)
            z = torch.cat((ident, moved), dim=-1)[..., self.inverse_permutation]
            return z, -scale.sum(dim=-1)
        moved = moved * torch.exp(scale) + shift
        return torch.cat((ident, moved), dim=-1), scale.sum(dim=-1)


class JetFormerFlow(nn.Module):
    """Flat values in/out; full attention only within each object/modality.

    ``positions`` and ``object_ids`` are fixed conditioning information, not
    transformed coordinates. Padding introduced for batching is excluded from
    attention and outputs; pre-existing zero-filled spectral bins are unchanged.
    Affine arithmetic stays at least float32, including under bf16 training.
    """

    def __init__(
        self, dim: int, steps: int = 16, hidden_dim: int = 512,
        depth: int = 4, num_heads: int = 8, pos_type: str = "index",
        max_positions: int = 144, pos_input_size: int = 1,
    ):
        super().__init__()
        for name, value in {
            "dim": dim, "steps": steps, "hidden_dim": hidden_dim,
            "depth": depth, "num_heads": num_heads, "max_positions": max_positions,
            "pos_input_size": pos_input_size,
        }.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dim % 2 or hidden_dim % num_heads:
            raise ValueError("JetFormer flows require even dim and hidden_dim divisible by num_heads")
        if pos_type not in {"index", "continuous"}:
            raise ValueError(f"unsupported flow position type {pos_type!r}")
        self.dim = dim
        self.pos_type = pos_type
        self.pos_input_size = pos_input_size
        self.blocks = nn.ModuleList(
            FlowCoupling(
                dim, hidden_dim, depth, num_heads, pos_type, max_positions,
                pos_input_size, stage,
            )
            for stage in range(steps)
        )

    def reset_coupling_outputs(self):
        """Reapply zero-output initialization after the parent model initializes."""
        for block in self.blocks:
            nn.init.zeros_(block.out_proj.weight)
            nn.init.zeros_(block.out_proj.bias)

    def forward(self, values, positions, object_ids, reverse=False):
        if values.ndim != 2 or values.shape[-1] != self.dim:
            raise ValueError(f"expected flat [N, {self.dim}] flow values")
        if object_ids.ndim != 1 or len(object_ids) != len(values):
            raise ValueError("one object id is required per flow token")
        expected_position_shape = (
            (len(values),) if self.pos_type == "index"
            else (len(values), self.pos_input_size)
        )
        if positions.shape != expected_position_shape:
            raise ValueError(f"expected positions shaped {expected_position_shape}")
        if object_ids.numel() > 1 and (object_ids[1:] < object_ids[:-1]).any():
            raise ValueError("flow object ids must follow packed row-major order")
        if not len(values):
            # Empty modalities must still participate in the distributed graph.
            zero = sum(parameter.reshape(-1)[0] * 0 for parameter in self.parameters())
            return values + zero, values.new_zeros((0,)) + zero

        _, counts = torch.unique_consecutive(object_ids, return_counts=True)
        lengths = counts.tolist()
        work = values.float() if values.dtype in {torch.float16, torch.bfloat16} else values
        if min(lengths) == max(lengths):
            z = work.reshape(len(lengths), lengths[0], self.dim)
            pos = positions.reshape(len(lengths), lengths[0], *positions.shape[1:])
            valid = None
            allowed_keys = None
        else:
            z = pad_sequence(work.split(lengths), batch_first=True)
            pos = pad_sequence(positions.split(lengths), batch_first=True)
            valid = torch.arange(z.shape[1], device=z.device)[None, :] < counts[:, None]
            allowed_keys = valid[:, None, None, :]
        logdet = z.new_zeros(z.shape[:-1])
        blocks = reversed(self.blocks) if reverse else self.blocks
        for block in blocks:
            if self.training and torch.is_grad_enabled():
                z, ld = checkpoint(
                    block, z, pos, allowed_keys, reverse,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                z, ld = block(z, pos, allowed_keys, reverse)
            logdet = logdet + ld
        if valid is None:
            return z.reshape(-1, self.dim), logdet.reshape(-1)
        return z[valid], logdet[valid]
