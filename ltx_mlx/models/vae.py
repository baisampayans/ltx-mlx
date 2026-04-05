"""LTX-Video Official VAE Decoder — pure MLX.

Timestep-conditioned CausalVideoAutoencoder decoder (from combined checkpoint).
Architecture: 128→1024→512→256→128→48 with 3x spatiotemporal 2x upsampling.
Each ResBlock has scale_shift_table for timestep modulation.

MLX uses channels-last (B, D, H, W, C); PyTorch uses (B, C, D, H, W).
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ============================================================
# Building blocks
# ============================================================

class CausalConv3d(nn.Module):
    """3D convolution with edge-replicated temporal padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride=1):
        super().__init__()
        self.kernel_t = kernel if isinstance(kernel, int) else kernel[0]
        s = stride if isinstance(stride, tuple) else (stride, stride, stride)
        h_pad = kernel // 2 if isinstance(kernel, int) else kernel[1] // 2
        w_pad = kernel // 2 if isinstance(kernel, int) else kernel[2] // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel,
                              stride=s, padding=(0, h_pad, w_pad), bias=True)

    def __call__(self, x):
        t_pad = (self.kernel_t - 1) // 2
        if t_pad > 0:
            pad_left = mx.repeat(x[:, :1], repeats=t_pad, axis=1)
            pad_right = mx.repeat(x[:, -1:], repeats=t_pad, axis=1)
            x = mx.concatenate([pad_left, x, pad_right], axis=1)
        return self.conv(x)


class RMSNormNoBias(nn.Module):
    """RMSNorm without learnable parameters."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dims = dims

    def __call__(self, x):
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / rms


class _TimestepEmbedderInner(nn.Module):
    """Inner timestep embedder: linear_1 -> SiLU -> linear_2."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, out_dim, bias=True)
        self.linear_2 = nn.Linear(out_dim, out_dim, bias=True)

    def __call__(self, emb: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(emb)))


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding: scalar -> vector.

    Matches checkpoint key hierarchy: timestep_embedder.linear_1, timestep_embedder.linear_2
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.timestep_embedder = _TimestepEmbedderInner(out_dim)

    @staticmethod
    def _sinusoidal(t: mx.array) -> mx.array:
        half = 128
        freqs = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

    def __call__(self, t: mx.array) -> mx.array:
        emb = self._sinusoidal(t)
        return self.timestep_embedder(emb)


# ============================================================
# Timestep-conditioned ResBlock
# ============================================================

class ResBlock3d(nn.Module):
    """ResNet block with timestep conditioning via scale_shift_table.

    scale_shift_table: (4, channels) — shift1, scale1, shift2, scale2
    """

    def __init__(self, in_ch: int, out_ch: int = None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.norm1 = RMSNormNoBias(in_ch)
        self.conv1 = CausalConv3d(in_ch, out_ch, kernel=3)
        self.norm2 = RMSNormNoBias(out_ch)
        self.conv2 = CausalConv3d(out_ch, out_ch, kernel=3)

        self.conv_shortcut = None
        if in_ch != out_ch:
            self.conv_shortcut = CausalConv3d(in_ch, out_ch, kernel=1)

        # Timestep modulation: (4, channels)
        self.scale_shift_table = mx.zeros((4, out_ch))

    def __call__(self, x, temb: Optional[mx.array] = None):
        B = x.shape[0]

        if temb is not None:
            # temb: (B, 4*C) -> (B, 4, C, 1, 1, 1) for NDHWC: (B, 4, 1, 1, 1, C)
            C = self.scale_shift_table.shape[1]
            ada = self.scale_shift_table[None, :, :] + temb.reshape(B, 4, C)
            # ada: (B, 4, C) -> broadcast over D, H, W
            shift1 = ada[:, 0:1, None, None, :]  # (B, 1, 1, 1, C)
            scale1 = ada[:, 1:2, None, None, :]
            shift2 = ada[:, 2:3, None, None, :]
            scale2 = ada[:, 3:4, None, None, :]

            h = self.norm1(x) * (1 + scale1) + shift1
        else:
            h = self.norm1(x)

        h = self.conv1(nn.silu(h))

        if temb is not None:
            h = self.norm2(h) * (1 + scale2) + shift2
        else:
            h = self.norm2(h)

        h = self.conv2(nn.silu(h))

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return h + x


# ============================================================
# Upsampler (channel-to-space)
# ============================================================

class Upsampler3d(nn.Module):
    """Channel-to-space upsampling with residual skip connection.

    Conv produces out_ch * stride_product channels, reshuffles to spatial dims.
    Residual: pixel-shuffles input to match output spatial dims, repeats channels.
    """

    def __init__(self, in_ch: int, out_ch: int, stride=(2, 2, 2)):
        super().__init__()
        self.stride = stride
        self.out_ch = out_ch
        self.in_ch = in_ch
        s0, s1, s2 = stride
        self.upscale_factor = s0 * s1 * s2 // (s0 * s1 * s2 // (s0 * s1 * s2 // max(1, in_ch // out_ch)))
        self.conv = CausalConv3d(in_ch, out_ch * s0 * s1 * s2, kernel=3)

    def _pixel_unshuffle(self, x):
        """Reshuffle channels to spatial dims (NDHWC format)."""
        B, D, H, W, C = x.shape
        s0, s1, s2 = self.stride
        c_reduced = C // (s0 * s1 * s2)
        x = x.reshape(B, D, H, W, c_reduced, s0, s1, s2)
        x = mx.transpose(x, axes=(0, 1, 5, 2, 6, 3, 7, 4))
        x = x.reshape(B, D * s0, H * s1, W * s2, c_reduced)
        if s0 > 1:
            x = x[:, s0 - 1:]
        return x

    def __call__(self, x, temb=None):
        # Residual: pixel-shuffle input + tile channels to match output
        s0, s1, s2 = self.stride
        stride_prod = s0 * s1 * s2
        residual = self._pixel_unshuffle(x)
        # residual has in_ch // stride_prod channels, need out_ch
        res_ch = self.in_ch // stride_prod
        repeats = self.out_ch // res_ch
        if repeats > 1:
            # torch.repeat TILES: [c0..c127, c0..c127, ...] (not interleave)
            # mx.tile achieves this: tile along last axis
            residual = mx.tile(residual, (1, 1, 1, 1, repeats))

        # Conv path
        x = self.conv(x)
        x = self._pixel_unshuffle(x)

        return x + residual


# ============================================================
# Block groups
# ============================================================

class ResBlockGroup(nn.Module):
    """Group of ResBlocks with shared timestep embedder."""

    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.res_blocks = [ResBlock3d(channels) for _ in range(num_blocks)]
        self.time_embedder = TimestepEmbedder(channels * 4)

    def __call__(self, x, temb=None):
        if temb is not None:
            block_temb = self.time_embedder(temb)
        else:
            block_temb = None

        for block in self.res_blocks:
            x = block(x, block_temb)
        return x


class MidBlock3d(nn.Module):
    """Mid block: ResBlocks at constant channels (no timestep conditioning in mid)."""

    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.res_blocks = [ResBlock3d(channels) for _ in range(num_blocks)]

    def __call__(self, x, temb=None):
        for block in self.res_blocks:
            x = block(x)  # Mid block has no timestep conditioning
        return x


# ============================================================
# Full Official Decoder
# ============================================================

class LTXVideoDecoder(nn.Module):
    """Official LTX-Video VAE Decoder (timestep-conditioned).

    Architecture (from combined checkpoint):
        conv_in: 128 → 1024
        mid_block: 2 ResBlocks at 1024 (no timestep)
        up_blocks:
            0: 5 ResBlocks at 1024 + time_embedder (4096 out)
            1: Upsample 1024→512 via conv 1024→8192 + (2,2,2) reshape
            2: 5 ResBlocks at 512 + time_embedder (2048 out)
            3: Upsample 512→256 via conv 512→4096 + (2,2,2) reshape
            4: 5 ResBlocks at 256 + time_embedder (1024 out)
            5: Upsample 256→128 via conv 256→2048 + (2,2,2) reshape
            6: 5 ResBlocks at 128 + time_embedder (512 out)
        norm_out: RMSNorm(128)
        last_time_embedder + last_scale_shift_table: final timestep conditioning
        conv_out: 128 → 48
        De-patch: 48 → 3 RGB via (patch_size=4, patch_size_t=1)
    """

    def __init__(self):
        super().__init__()
        self.conv_in = CausalConv3d(128, 1024, kernel=3)

        # Mid block (from checkpoint: 2 ResBlocks at 1024, no timestep)
        # Actually checking keys: no mid_block in checkpoint, it's all up_blocks
        # up_blocks.0 = 5 ResBlocks at 1024 is the "first" block
        self.up_blocks = []

        # Block 0: 5 ResBlocks at 1024 with timestep
        self.up_blocks.append(ResBlockGroup(1024, 5))
        # Block 1: Upsample 1024→512 via conv 1024→4096 + (2,2,2) reshape
        self.up_blocks.append(Upsampler3d(1024, 512, stride=(2, 2, 2)))
        # Block 2: 5 ResBlocks at 512 with timestep
        self.up_blocks.append(ResBlockGroup(512, 5))
        # Block 3: Upsample 512→256
        self.up_blocks.append(Upsampler3d(512, 256, stride=(2, 2, 2)))
        # Block 4: 5 ResBlocks at 256 with timestep
        self.up_blocks.append(ResBlockGroup(256, 5))
        # Block 5: Upsample 256→128
        self.up_blocks.append(Upsampler3d(256, 128, stride=(2, 2, 2)))
        # Block 6: 5 ResBlocks at 128 with timestep
        self.up_blocks.append(ResBlockGroup(128, 5))

        self.norm_out = RMSNormNoBias(128)

        # Final timestep conditioning
        self.last_time_embedder = TimestepEmbedder(256)
        self.last_scale_shift_table = mx.zeros((2, 128))

        self.conv_out = CausalConv3d(128, 48, kernel=3)

        # Timestep scale multiplier (scalar)
        self.timestep_scale_multiplier = mx.ones(())

        self.patch_size = 4
        self.patch_size_t = 1

    def __call__(self, x, temb: Optional[mx.array] = None):
        """Decode latents to video.

        Args:
            x: (B, D, H, W, 128) NDHWC latent
            temb: (B,) timestep for conditioning (e.g., 0.05)

        Returns:
            (B, D_out, H*4, W*4, 3) NDHWC video
        """
        B = x.shape[0]

        x = self.conv_in(x)

        # Scale timestep
        if temb is not None:
            temb = temb * self.timestep_scale_multiplier

        # Up blocks
        for up_block in self.up_blocks:
            x = up_block(x, temb)

        # Final norm + timestep conditioning
        x = self.norm_out(x)  # Already channels-last

        if temb is not None and self.last_time_embedder is not None:
            t_emb = self.last_time_embedder(temb)  # (B, 256)
            # Reshape for broadcasting: (B, 256) -> (B, 2, 128) + table
            t_emb = t_emb.reshape(B, 2, 128)
            t_emb = t_emb + self.last_scale_shift_table[None, :, :]
            shift = t_emb[:, 0:1, None, None, :]  # (B, 1, 1, 1, 128)
            scale = t_emb[:, 1:2, None, None, :]
            x = x * (1 + scale) + shift

        x = nn.silu(x)
        x = self.conv_out(x)

        # De-patch: (B, D, H, W, 48) -> (B, D, H*4, W*4, 3)
        B, D, H, W, C = x.shape
        p = self.patch_size
        p_t = self.patch_size_t
        x = x.reshape(B, D, H, W, 3, p_t, p, p)
        x = mx.transpose(x, axes=(0, 1, 5, 2, 7, 3, 6, 4))
        x = x.reshape(B, D * p_t, H * p, W * p, 3)
        return x


def load_vae_decoder(ckpt_path: str, dtype: mx.Dtype = mx.bfloat16) -> LTXVideoDecoder:
    """Load official LTX VAE decoder from combined safetensors checkpoint.

    Args:
        ckpt_path: Path to ltxv-2b-0.9.8-distilled.safetensors (combined checkpoint)
    """
    decoder = LTXVideoDecoder()

    raw = mx.load(ckpt_path)

    # Filter and map VAE decoder keys
    mapped = {}
    for k, v in raw.items():
        if not k.startswith("vae.decoder."):
            continue
        key = k[len("vae.decoder."):]

        # Skip encoder keys
        if key.startswith("encoder."):
            continue

        # Transpose conv3d: PyTorch (C_out, C_in, D, H, W) -> MLX (C_out, D, H, W, C_in)
        if key.endswith(".conv.weight") and v.ndim == 5:
            v = mx.transpose(v, axes=(0, 2, 3, 4, 1))

        mapped[key] = v.astype(dtype)

    decoder.load_weights(list(mapped.items()))

    import mlx.utils
    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(decoder.parameters()))
    print(f"  VAE decoder loaded: {n_params/1e6:.0f}M params, {dtype}")
    return decoder
