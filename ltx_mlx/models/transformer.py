"""LTX-Video 2B Transformer (DiT) — pure MLX.

Architecture: 28 layers, 2048 hidden, 32 heads x 64 dim, 128 latent channels.
3D spatiotemporal RoPE, AdaLayerNorm (PixArt-style), cross-attention to T5.

Weight format (from combined safetensors checkpoint):
    model.diffusion_model.patchify_proj.{weight,bias}
    model.diffusion_model.adaln_single.emb.timestep_embedder.linear_{1,2}.{weight,bias}
    model.diffusion_model.adaln_single.linear.{weight,bias}
    model.diffusion_model.caption_projection.linear_{1,2}.{weight,bias}
    model.diffusion_model.transformer_blocks.{i}.attn1.to_{q,k,v}.{weight,bias}
    model.diffusion_model.transformer_blocks.{i}.attn1.{q_norm,k_norm}.weight
    model.diffusion_model.transformer_blocks.{i}.attn1.to_out.0.{weight,bias}
    model.diffusion_model.transformer_blocks.{i}.attn2.to_{q,k,v}.{weight,bias}
    model.diffusion_model.transformer_blocks.{i}.attn2.{q_norm,k_norm}.weight
    model.diffusion_model.transformer_blocks.{i}.attn2.to_out.0.{weight,bias}
    model.diffusion_model.transformer_blocks.{i}.ff.net.0.proj.{weight,bias}
    model.diffusion_model.transformer_blocks.{i}.ff.net.2.{weight,bias}
    model.diffusion_model.transformer_blocks.{i}.norm1.weight
    model.diffusion_model.transformer_blocks.{i}.norm2.weight
    model.diffusion_model.transformer_blocks.{i}.scale_shift_table
    model.diffusion_model.proj_out.{weight,bias}
    model.diffusion_model.norm_out.weight
    model.diffusion_model.scale_shift_table
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ============================================================
# 3D Spatiotemporal RoPE
# ============================================================

def compute_rope(
    seq_len: int,
    dim: int,
    num_frames: int,
    height: int,
    width: int,
    rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
    base_num_frames: int = 20,
    base_height: int = 2048,
    base_width: int = 2048,
    patch_size: int = 1,
    patch_size_t: int = 1,
    theta: float = 10000.0,
) -> Tuple[mx.array, mx.array]:
    """Compute 3D RoPE cos/sin for LTX-Video.

    Args:
        rope_interpolation_scale: (temporal, height, width) scaling factors.
            Default for LTX-Video: (vae_temporal_compression/frame_rate, vae_spatial_compression, vae_spatial_compression)
            = (8/24, 32, 32)

    Returns (cos, sin) each of shape (seq_len, dim).
    """
    # 3D coordinate grid: (T, H, W) -> (T*H*W, 3)
    grid_f = mx.arange(num_frames, dtype=mx.float32)
    grid_h = mx.arange(height, dtype=mx.float32)
    grid_w = mx.arange(width, dtype=mx.float32)

    gf = mx.broadcast_to(grid_f[:, None, None], (num_frames, height, width))
    gh = mx.broadcast_to(grid_h[None, :, None], (num_frames, height, width))
    gw = mx.broadcast_to(grid_w[None, None, :], (num_frames, height, width))

    # Apply rope_interpolation_scale (matches diffusers LTXVideoRotaryPosEmbed)
    if rope_interpolation_scale is not None:
        gf = gf * rope_interpolation_scale[0] * patch_size_t / base_num_frames
        gh = gh * rope_interpolation_scale[1] * patch_size / base_height
        gw = gw * rope_interpolation_scale[2] * patch_size / base_width

    coords = mx.stack([gf.reshape(-1), gh.reshape(-1), gw.reshape(-1)], axis=-1)

    # Frequencies
    freqs = theta ** mx.linspace(
        math.log(1.0) / math.log(theta),
        1.0,
        dim // 6,
    ).astype(mx.float32)
    freqs = freqs * math.pi / 2.0

    # (seq, 3) x (dim//6,) -> (seq, 3, dim//6)
    angles = (coords[:, :, None] * 2 - 1) * freqs[None, None, :]
    # -> (seq, dim//6, 3) -> (seq, dim//2)
    angles = angles.transpose(0, 2, 1).reshape(seq_len, -1)

    cos_freqs = mx.repeat(mx.cos(angles), 2, axis=-1)
    sin_freqs = mx.repeat(mx.sin(angles), 2, axis=-1)

    # Pad if dim % 6 != 0
    if dim % 6 != 0:
        pad_size = dim % 6
        cos_freqs = mx.concatenate([mx.ones((seq_len, pad_size)), cos_freqs], axis=-1)
        sin_freqs = mx.concatenate([mx.zeros((seq_len, pad_size)), sin_freqs], axis=-1)

    return cos_freqs, sin_freqs


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply rotary embedding. x: (B, L, D), cos/sin: (L, D)."""
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    x_rotated = mx.stack([-x_imag, x_real], axis=-1).reshape(x.shape)
    return x * cos[None, :, :] + x_rotated * sin[None, :, :]


# ============================================================
# AdaLayerNorm (PixArt-style timestep embedding)
# ============================================================

class AdaLayerNormSingle(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, 6 * dim, bias=True)
        self.timestep_embedder_linear_1 = nn.Linear(256, dim, bias=True)
        self.timestep_embedder_linear_2 = nn.Linear(dim, dim, bias=True)

    def _sinusoidal_embed(self, t: mx.array) -> mx.array:
        half = 128
        freqs = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

    def __call__(self, t: mx.array) -> Tuple[mx.array, mx.array]:
        emb = self._sinusoidal_embed(t)
        emb = self.silu(self.timestep_embedder_linear_1(emb))
        embedded_t = self.timestep_embedder_linear_2(emb)
        temb = self.linear(self.silu(embedded_t))
        return temb, embedded_t


# ============================================================
# Text Projection (PixArt-Alpha)
# ============================================================

class TextProjection(nn.Module):
    def __init__(self, in_features: int = 4096, hidden_size: int = 2048):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size, bias=True)
        self.act_1 = nn.GELU(approx="precise")
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(self.act_1(self.linear_1(x)))


# ============================================================
# Attention
# ============================================================

class LTXAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 32, head_dim: int = 64,
                 cross_dim: Optional[int] = None, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim
        kv_dim = cross_dim if cross_dim is not None else dim

        self.to_q = nn.Linear(dim, inner, bias=bias)
        self.to_k = nn.Linear(kv_dim, inner, bias=bias)
        self.to_v = nn.Linear(kv_dim, inner, bias=bias)
        self.to_out = nn.Linear(inner, dim, bias=bias)

        self.norm_q = nn.RMSNorm(inner, eps=1e-5)
        self.norm_k = nn.RMSNorm(inner, eps=1e-5)

    def __call__(self, x: mx.array, context: Optional[mx.array] = None,
                 rope: Optional[Tuple[mx.array, mx.array]] = None,
                 mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        kv = context if context is not None else x

        q = self.norm_q(self.to_q(x))
        k = self.norm_k(self.to_k(kv))
        v = self.to_v(kv)

        # RoPE before head split (self-attention only)
        if rope is not None:
            cos, sin = rope
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        # Standard BHLD multi-head attention format
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        S = kv.shape[1]
        k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.to_out(out)


# ============================================================
# FeedForward (GELU-tanh)
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner = int(dim * mult)
        self.proj_in = nn.Linear(dim, inner, bias=True)
        self.act = nn.GELU(approx="tanh")
        self.proj_out = nn.Linear(inner, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj_out(self.act(self.proj_in(x)))


# ============================================================
# Transformer Block
# ============================================================

class LTXBlock(nn.Module):
    def __init__(self, dim: int = 2048, num_heads: int = 32, head_dim: int = 64,
                 cross_dim: int = 2048, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        self.attn1 = LTXAttention(dim, num_heads, head_dim)
        self.norm2 = nn.RMSNorm(dim, eps=eps)
        self.attn2 = LTXAttention(dim, num_heads, head_dim, cross_dim=cross_dim)
        self.ff = FeedForward(dim)
        self.scale_shift_table = mx.zeros((6, dim))

    def __call__(self, x: mx.array, ctx: mx.array, temb: mx.array,
                 rope: Optional[Tuple[mx.array, mx.array]] = None,
                 encoder_mask: Optional[mx.array] = None) -> mx.array:
        B = x.shape[0]

        # AdaLN modulation
        ada = self.scale_shift_table[None, None, :, :] + temb.reshape(B, temb.shape[1], 6, -1)
        shift_msa, scale_msa, gate_msa = ada[:, :, 0], ada[:, :, 1], ada[:, :, 2]
        shift_mlp, scale_mlp, gate_mlp = ada[:, :, 3], ada[:, :, 4], ada[:, :, 5]

        # Self-attention with RoPE
        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        h = self.attn1(h, rope=rope)
        x = x + h * gate_msa

        # Cross-attention (no RoPE, with encoder mask)
        h = self.attn2(x, context=ctx, mask=encoder_mask)
        x = x + h

        # FFN
        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        h = self.ff(h)
        x = x + h * gate_mlp

        return x


# ============================================================
# Full LTX-Video Transformer
# ============================================================

class LTXVideoTransformer(nn.Module):
    """LTX-Video 2B DiT.

    28 layers, 2048 hidden, 32x64 heads, 128 latent channels.
    """

    def __init__(self, in_channels: int = 128, out_channels: int = 128,
                 inner_dim: int = 2048, num_heads: int = 32, head_dim: int = 64,
                 num_layers: int = 28, cross_dim: int = 2048,
                 caption_channels: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.proj_in = nn.Linear(in_channels, inner_dim)
        self.time_embed = AdaLayerNormSingle(inner_dim)
        self.caption_projection = TextProjection(caption_channels, inner_dim)

        self.blocks = [
            LTXBlock(inner_dim, num_heads, head_dim, cross_dim, eps)
            for _ in range(num_layers)
        ]

        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)
        self.scale_shift_table = mx.zeros((2, inner_dim))

    def __call__(self, x: mx.array, t: mx.array,
                 encoder_hidden_states: mx.array,
                 num_frames: int, height: int, width: int,
                 rope: Optional[Tuple[mx.array, mx.array]] = None,
                 encoder_attention_mask: Optional[mx.array] = None,
                 conditioning_mask: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: (B, L, C) latent tokens
            t: (B,) timestep values (sigma * 1000)
            encoder_hidden_states: (B, S, caption_channels) T5 text embeddings
            num_frames, height, width: latent grid dimensions
            rope: precomputed (cos, sin) RoPE
            encoder_attention_mask: (B, S) text attention mask
            conditioning_mask: (B, L) optional per-token mask. 1.0 = conditioned (timestep→0),
                0.0 = unconditioned (normal timestep). Used for image-to-video.
        """
        B, L, C = x.shape

        if rope is None:
            rope = compute_rope(L, self.inner_dim, num_frames, height, width)

        x = self.proj_in(x)

        if conditioning_mask is not None:
            # Per-token timestep: conditioned tokens get t=0, others get normal t
            temb_noisy, embedded_t_noisy = self.time_embed(t)
            temb_clean, embedded_t_clean = self.time_embed(mx.zeros_like(t))
            # Blend per-token: (B, L, D)
            m = conditioning_mask[:, :, None].astype(x.dtype)  # (B, L, 1)
            temb = (temb_clean[:, None, :] * m + temb_noisy[:, None, :] * (1 - m)).astype(x.dtype)
            embedded_t = (embedded_t_clean[:, None, :] * m + embedded_t_noisy[:, None, :] * (1 - m)).astype(x.dtype)
        else:
            temb, embedded_t = self.time_embed(t)
            temb = temb.astype(x.dtype).reshape(B, 1, temb.shape[-1])
            embedded_t = embedded_t.astype(x.dtype).reshape(B, 1, embedded_t.shape[-1])

        ctx = self.caption_projection(encoder_hidden_states)
        ctx = ctx.reshape(B, -1, self.inner_dim)

        # Build cross-attention mask: (B, 1, 1, S) for SDPA
        encoder_mask = None
        if encoder_attention_mask is not None:
            encoder_mask = mx.where(
                encoder_attention_mask[:, None, None, :] == 0,
                mx.array(float("-inf")),
                mx.array(0.0),
            )

        for block in self.blocks:
            x = block(x, ctx, temb, rope=rope, encoder_mask=encoder_mask)

        # Final norm + scale/shift
        shift = self.scale_shift_table[0:1, :] + embedded_t
        scale = self.scale_shift_table[1:2, :] + embedded_t
        x = self.norm_out(x) * (1 + scale) + shift

        return self.proj_out(x)


def load_transformer(ckpt_path: str, dtype: mx.Dtype = mx.bfloat16) -> LTXVideoTransformer:
    """Load LTX-Video transformer from combined safetensors checkpoint.

    Auto-detects model size (2B or 13B) from weight shapes.
    Uses mx.load for zero-copy memory-mapped loading.
    """
    raw = mx.load(ckpt_path)

    # Auto-detect architecture from proj_in weight shape
    prefix = "model.diffusion_model."
    proj_in_w = raw.get(prefix + "patchify_proj.weight")
    inner_dim = proj_in_w.shape[0] if proj_in_w is not None else 2048

    # Count transformer blocks
    block_ids = set()
    for k in raw.keys():
        if prefix + "transformer_blocks." in k:
            idx = int(k.split("transformer_blocks.")[1].split(".")[0])
            block_ids.add(idx)
    num_layers = max(block_ids) + 1 if block_ids else 28

    # Detect head config from norm_q weight (inner_dim) and actual head count
    # norm_q normalizes the full inner_dim. The actual head count is stored in
    # the attention module. Detect from scale_shift_table or use known configs.
    # 2B: 32 heads x 64 head_dim = 2048
    # 13B: 32 heads x 128 head_dim = 4096
    if inner_dim == 2048:
        num_heads, head_dim = 32, 64
    elif inner_dim == 4096:
        num_heads, head_dim = 32, 128
    else:
        # Fallback: assume 32 heads
        num_heads = 32
        head_dim = inner_dim // num_heads

    # Detect caption channels from caption_projection
    cap_w = raw.get(prefix + "caption_projection.linear_1.weight")
    caption_channels = cap_w.shape[1] if cap_w is not None else 4096

    print(f"  Detected: {inner_dim}d, {num_layers} blocks, {num_heads}x{head_dim} heads")

    model = LTXVideoTransformer(
        inner_dim=inner_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        cross_dim=inner_dim,
        caption_channels=caption_channels,
    )

    prefix = "model.diffusion_model."
    mapped = {}
    for k, v in raw.items():
        if not k.startswith(prefix):
            continue
        new_k = k[len(prefix):]

        # Key remapping
        new_k = new_k.replace("patchify_proj.", "proj_in.")
        new_k = new_k.replace("adaln_single.emb.timestep_embedder.linear_1.",
                              "time_embed.timestep_embedder_linear_1.")
        new_k = new_k.replace("adaln_single.emb.timestep_embedder.linear_2.",
                              "time_embed.timestep_embedder_linear_2.")
        new_k = new_k.replace("adaln_single.linear.", "time_embed.linear.")
        new_k = new_k.replace("transformer_blocks.", "blocks.")
        new_k = new_k.replace(".q_norm.", ".norm_q.")
        new_k = new_k.replace(".k_norm.", ".norm_k.")

        # Skip dropout layers
        if ".to_out.1." in new_k:
            continue
        new_k = new_k.replace(".to_out.0.", ".to_out.")

        # FFN remapping
        new_k = new_k.replace(".ff.net.0.proj.", ".ff.proj_in.")
        new_k = new_k.replace(".ff.net.2.", ".ff.proj_out.")

        mapped[new_k] = v.astype(dtype)

    import mlx.utils
    model.update(mlx.utils.tree_unflatten(list(mapped.items())))
    mx.eval(model.parameters())

    import mlx.utils
    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    print(f"  Transformer loaded: {n_params/1e9:.2f}B params, {dtype}")
    return model
