"""LTX-Video pipeline — pure MLX, zero heavy dependencies.

Orchestrates: T5 text encoding -> DiT denoising -> VAE decode -> video frames.
No PyTorch, no diffusers, no transformers at runtime.

Usage:
    from ltx_mlx import LTXPipeline

    # 2B (fast, ~7s at 480p)
    pipe = LTXPipeline("models/LTX")
    frames = pipe.generate("a cat sitting on a windowsill")

    # 13B (sharp, ~60s at 720p)
    pipe = LTXPipeline("models/LTX", checkpoint="LTX 13B/ltxv-13b-0.9.8-distilled.safetensors")
    frames = pipe.generate("a portrait, cinematic", height=720, width=1280)
"""

import gc
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from ltx_mlx.models.text_encoder import T5Encoder, T5Tokenizer, load_t5_encoder
from ltx_mlx.models.transformer import LTXVideoTransformer, compute_rope, load_transformer
from ltx_mlx.models.vae import LTXVideoDecoder, LTXVideoEncoder, load_vae_decoder, load_vae_encoder
from ltx_mlx.scheduler import euler_step, get_sigmas


class LTXPipeline:
    """LTX-Video generation pipeline for Apple Silicon.

    Supports both 2B and 13B models with auto-detection.

    Args:
        model_dir: Path to model directory containing:
            - text_encoder/ (T5v1.1-XXL sharded safetensors)
            - tokenizer/spiece.model
            - Combined checkpoint (.safetensors with DiT + VAE)
        checkpoint: Path to combined checkpoint relative to model_dir.
            If None, auto-detects the first .safetensors in model_dir.
        dtype: Model precision (default bf16)
    """

    VAE_TEMPORAL_COMPRESSION = 8
    VAE_SPATIAL_COMPRESSION = 32

    def __init__(self, model_dir: str, checkpoint: str = None,
                 dtype: mx.Dtype = mx.bfloat16):
        self.model_dir = Path(model_dir)
        self.dtype = dtype
        self.t5: Optional[T5Encoder] = None
        self.tokenizer: Optional[T5Tokenizer] = None
        self.transformer: Optional[LTXVideoTransformer] = None
        self.vae: Optional[LTXVideoDecoder] = None
        self.vae_encoder: Optional[LTXVideoEncoder] = None
        self._rope_cache = {}

        # Auto-detect checkpoint
        if checkpoint:
            self._ckpt = str(self.model_dir / checkpoint)
        else:
            candidates = list(self.model_dir.glob("*.safetensors"))
            if not candidates:
                candidates = list(self.model_dir.glob("**/*distilled*.safetensors"))
            if not candidates:
                raise FileNotFoundError(f"No .safetensors checkpoint found in {model_dir}")
            self._ckpt = str(candidates[0])

        self._load_all()

    def _load_all(self):
        total_t0 = time.time()

        print("Loading tokenizer...")
        t0 = time.time()
        spiece_path = self.model_dir / "tokenizer" / "spiece.model"
        self.tokenizer = T5Tokenizer(str(spiece_path), max_length=128)
        print(f"  Tokenizer: {time.time()-t0:.2f}s")

        print("Loading T5 encoder...")
        t0 = time.time()
        self.t5 = load_t5_encoder(str(self.model_dir / "text_encoder"), self.dtype)
        print(f"  T5: {time.time()-t0:.1f}s")

        print(f"Loading transformer from {Path(self._ckpt).name}...")
        t0 = time.time()
        self.transformer = load_transformer(self._ckpt, self.dtype)
        print(f"  Transformer: {time.time()-t0:.1f}s")

        print("Loading VAE decoder...")
        t0 = time.time()
        self.vae = load_vae_decoder(self._ckpt, self.dtype)
        print(f"  VAE decoder: {time.time()-t0:.1f}s")

        print("Loading VAE encoder...")
        t0 = time.time()
        self.vae_encoder = load_vae_encoder(self._ckpt, self.dtype)
        print(f"  VAE encoder: {time.time()-t0:.1f}s")

        self._load_latent_stats()

        # Detect model size for default step count
        self._is_13b = self.transformer.inner_dim == 4096
        model_name = "13B" if self._is_13b else "2B"
        print(f"\nLTX-Video {model_name} ready in {time.time()-total_t0:.1f}s")

    def _load_latent_stats(self):
        mean_path = self.model_dir / "vae_mean_of_means.npy"
        std_path = self.model_dir / "vae_std_of_means.npy"

        if mean_path.exists() and std_path.exists():
            self.latent_mean = mx.array(np.load(str(mean_path))).astype(self.dtype)
            self.latent_std = mx.array(np.load(str(std_path))).astype(self.dtype)
        else:
            raw = mx.load(self._ckpt)
            mean_key = "vae.per_channel_statistics.mean-of-means"
            std_key = "vae.per_channel_statistics.std-of-means"
            if mean_key in raw and std_key in raw:
                self.latent_mean = raw[mean_key].astype(self.dtype)
                self.latent_std = raw[std_key].astype(self.dtype)
            else:
                self.latent_mean = None
                self.latent_std = None

    def _get_rope(self, seq_len: int, num_frames: int, height: int, width: int,
                  frame_rate: int = 24):
        key = (seq_len, num_frames, height, width, frame_rate)
        if key not in self._rope_cache:
            rope_scale = (
                self.VAE_TEMPORAL_COMPRESSION / frame_rate,
                self.VAE_SPATIAL_COMPRESSION,
                self.VAE_SPATIAL_COMPRESSION,
            )
            self._rope_cache[key] = compute_rope(
                seq_len, self.transformer.inner_dim,
                num_frames, height, width,
                rope_interpolation_scale=rope_scale,
            )
        return self._rope_cache[key]

    def _encode_image(self, image: np.ndarray, lat_h: int, lat_w: int) -> mx.array:
        """Encode an image to latent space using the VAE encoder.

        Args:
            image: (H, W, 3) uint8 numpy array
            lat_h: Expected latent height
            lat_w: Expected latent width

        Returns:
            (1, lat_h * lat_w, 128) normalized latent tokens
        """
        # Preprocess: uint8 [0,255] → float [-1, 1], add batch+temporal dims
        img = image.astype(np.float32) / 255.0 * 2.0 - 1.0
        img_mx = mx.array(img[None, None, ...]).astype(self.dtype)  # (1, 1, H, W, 3) NDHWC

        # Encode
        latent = self.vae_encoder(img_mx)  # (1, 1, lat_h, lat_w, 128)
        mx.eval(latent)

        # Normalize using per-channel stats (same stats used for decoder denormalization)
        if self.latent_mean is not None:
            latent = (latent - self.latent_mean[None, None, None, None, :]) / self.latent_std[None, None, None, None, :]

        # Flatten to token form: (1, lat_h * lat_w, 128)
        latent = latent.reshape(1, lat_h * lat_w, 128)
        return latent

    def generate(
        self,
        prompt: str,
        num_frames: int = 97,
        height: int = 480,
        width: int = 768,
        num_steps: Optional[int] = None,
        timesteps: Optional[list[float]] = None,
        seed: int = 42,
        image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate video frames from text, or image+text.

        Args:
            prompt: Text description of desired video.
            num_frames: Number of output frames (default 97 = ~4s at 24fps).
            height: Output height (must be divisible by 32).
            width: Output width (must be divisible by 32).
            num_steps: Denoising steps. Default: 4 for 2B, 8 for 13B.
            timesteps: Explicit sigma schedule (bypasses dynamic shifting).
            seed: Random seed.
            image: Optional (H, W, 3) uint8 numpy array for image-to-video.
                   Image is used as the first frame conditioning.

        Returns:
            (num_frames, height, width, 3) uint8 numpy array.
        """
        total_t0 = time.time()

        # Default steps: 4 for 2B, 8 for 13B
        if num_steps is None and timesteps is None:
            num_steps = 8 if self._is_13b else 4

        # Latent dimensions
        lat_h = height // self.VAE_SPATIAL_COMPRESSION
        lat_w = width // self.VAE_SPATIAL_COMPRESSION
        lat_t = (num_frames - 1) // self.VAE_TEMPORAL_COMPRESSION + 1
        seq_len = lat_t * lat_h * lat_w

        model_tag = "13B" if self._is_13b else "2B"
        print(f"\n=== LTX-Video {model_tag} MLX | {height}x{width} | {num_frames}f | "
              f"latent {lat_t}x{lat_h}x{lat_w} = {seq_len} tokens ===\n")

        # ---- T5 encoding ----
        t0 = time.time()
        input_ids, attention_mask = self.tokenizer.encode(prompt)
        text_embeds = self.t5(input_ids, mask=attention_mask)
        mx.eval(text_embeds)
        t5_time = time.time() - t0
        print(f"  T5 encode: {t5_time:.2f}s")

        # ---- RoPE ----
        rope = self._get_rope(seq_len, lat_t, lat_h, lat_w)

        # ---- Sigma schedule ----
        if timesteps is not None:
            sigmas = np.array(list(timesteps) + [0.0], dtype=np.float32)
            n_steps = len(timesteps)
        else:
            sigmas = get_sigmas(num_steps, seq_len)
            n_steps = num_steps

        # ---- Encode conditioning image (if provided) ----
        image_latent = None
        if image is not None:
            t0_enc = time.time()
            # Resize + center-crop to target resolution (preserves aspect ratio)
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            src_w, src_h = pil_img.size
            # Scale so the shorter side covers the target, then center-crop
            scale = max(width / src_w, height / src_h)
            new_w = round(src_w * scale)
            new_h = round(src_h * scale)
            pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
            # Center crop
            left = (new_w - width) // 2
            top = (new_h - height) // 2
            pil_img = pil_img.crop((left, top, left + width, top + height))
            image_resized = np.array(pil_img)

            image_latent = self._encode_image(image_resized, lat_h, lat_w)
            mx.eval(image_latent)
            enc_time = time.time() - t0_enc
            print(f"  Image encode: {enc_time:.2f}s")

        # ---- Denoise ----
        print(f"  Denoising: {n_steps} steps...")
        t0 = time.time()

        mx.random.seed(seed)
        noise = mx.random.normal((1, seq_len, 128)).astype(self.dtype)

        # Build conditioning mask for image-to-video
        cond_mask = None
        if image_latent is not None:
            frame_size = lat_h * lat_w
            # Frame 0 = image latent (clean), other frames = noise
            latents = mx.concatenate([
                image_latent,
                noise[:, frame_size:, :] * sigmas[0],
            ], axis=1)
            # Conditioning mask: 1.0 for frame 0 (conditioned), 0.0 for rest
            cond_mask = mx.concatenate([
                mx.ones((1, frame_size), dtype=self.dtype),
                mx.zeros((1, seq_len - frame_size), dtype=self.dtype),
            ], axis=1)
        else:
            latents = noise * sigmas[0]

        for i in range(n_steps):
            s = float(sigmas[i])
            s_next = float(sigmas[i + 1])
            t_val = mx.array([s * 1000.0])

            pred = self.transformer(
                latents, t_val, text_embeds, lat_t, lat_h, lat_w,
                rope=rope, encoder_attention_mask=attention_mask,
                conditioning_mask=cond_mask,
            )

            latents = euler_step(pred, s, s_next, latents)

            if cond_mask is not None:
                # Freeze conditioned tokens: replace frame 0 with original image latent
                latents = mx.concatenate([
                    image_latent,
                    latents[:, frame_size:, :],
                ], axis=1)

            mx.eval(latents)

        denoise_time = time.time() - t0
        print(f"  DiT: {denoise_time:.1f}s ({denoise_time/n_steps:.1f}s/step)")

        # ---- Denormalize + VAE decode ----
        t0 = time.time()
        if self.latent_mean is not None:
            latents = latents * self.latent_std[None, None, :] + self.latent_mean[None, None, :]

        lat_mlx = latents.reshape(1, lat_t, lat_h, lat_w, 128)
        decode_timestep = mx.array([0.05]).astype(self.dtype)
        video_mlx = self.vae(lat_mlx, temb=decode_timestep)
        mx.eval(video_mlx)
        vae_time = time.time() - t0
        print(f"  VAE decode: {vae_time:.1f}s")

        # ---- To uint8 ----
        video_np = np.array(video_mlx)[0]
        video_np = np.clip((video_np + 1.0) / 2.0, 0.0, 1.0)
        frames = (video_np * 255).astype(np.uint8)

        del latents, video_mlx, lat_mlx, pred
        gc.collect()
        mx.clear_cache()

        total = time.time() - total_t0
        print(f"  Total: {total:.1f}s (T5 {t5_time:.1f}s + DiT {denoise_time:.1f}s + VAE {vae_time:.1f}s)")

        return frames
