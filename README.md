# ltx-mlx

LTX-Video generation optimized for Apple Silicon. Pure MLX — no PyTorch, diffusers, or transformers at runtime.

Supports **LTX-Video 2B** (fast) and **LTX-Video 13B** (high quality) with automatic model detection.

## Performance

| Model | Resolution | Frames | Steps | Time | Notes |
|-------|-----------|--------|-------|------|-------|
| 2B | 480x768 | 97 | 4 | **~7s** | Fast preview |
| 13B | 480x768 | 97 | 8 | **~56s** | Good quality |
| 13B | 720x1280 | 41 | 8 | **~65s** | Sharp, cinematic |
| 13B | 1080x1920 | 17 | 8 | **~78s** | Maximum detail |

Benchmarked on M3 Ultra (76 GPU cores, 256 GB). Times include T5 encoding + DiT denoising + VAE decode.

## Install

```bash
pip install -e .
```

Dependencies: `mlx`, `numpy`, `pillow`, `sentencepiece`, `imageio[ffmpeg]`. No PyTorch needed.

## Model Setup

Download the LTX-Video checkpoints from [Hugging Face](https://huggingface.co/Lightricks/LTX-Video):

```
models/LTX/
├── ltxv-2b-0.9.8-distilled.safetensors          # 2B model (5.9 GB)
├── LTX 13B/
│   └── ltxv-13b-0.9.8-distilled.safetensors      # 13B model (27 GB)
├── text_encoder/                                   # T5v1.1-XXL (shared)
│   ├── config.json
│   ├── model-00001-of-00004.safetensors
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   └── model.safetensors.index.json
└── tokenizer/                                      # T5 tokenizer (shared)
    ├── spiece.model
    └── tokenizer_config.json
```

Both models share the same T5 text encoder and tokenizer.

## Usage

### CLI

```bash
# 2B model — fast (~7s)
ltx-mlx --prompt "a cat sitting on a windowsill, golden hour"

# 13B model — sharp (~60s)
ltx-mlx --prompt "a portrait, cinematic" --model 13b

# 720p for maximum quality
ltx-mlx --prompt "ocean waves crashing" --model 13b --resolution 720p

# Custom resolution + settings
ltx-mlx --prompt "a dancer" --height 544 --width 960 --steps 8 --seed 7 -o dance.mp4
```

### Python API

```python
from ltx_mlx import LTXPipeline

# 2B (fast)
pipe = LTXPipeline("models/LTX")
frames = pipe.generate("a cat on a windowsill", seed=42)

# 13B (high quality)
pipe = LTXPipeline("models/LTX", checkpoint="LTX 13B/ltxv-13b-0.9.8-distilled.safetensors")
frames = pipe.generate(
    "a woman smiling, soft studio lighting, portrait",
    height=720, width=1280,
    num_frames=41,
    seed=42,
)

# Save video
import imageio
imageio.mimwrite("output.mp4", frames, fps=24, codec="libx264")
```

### Resolution Guide

| Preset | Resolution | 97 frames | 41 frames | 17 frames | Best for |
|--------|-----------|-----------|-----------|-----------|----------|
| 480p | 480x768 | 4,680 | 2,160 | 1,080 | Fast previews |
| 544p | 544x960 | 6,630 | 3,060 | 1,530 | Balanced |
| 720p | 720x1280 | 11,440 | 5,280 | 2,640 | Sharp output |
| 1080p | 1080x1920 | 25,740 | 11,880 | 5,940 | Maximum detail |

Higher resolutions need fewer frames to fit in memory. Recommended:
- **480p**: up to 97 frames (~4s)
- **720p**: up to 41 frames (~1.7s)
- **1080p**: up to 17 frames (~0.7s)

## Architecture

```
Prompt → T5v1.1-XXL Encoder → Text Embeddings
                                    ↓
Noise → DiT Transformer (28 or 48 blocks) → Denoised Latents
                                    ↓
         VAE Decoder (timestep-conditioned) → Video Frames
```

- **T5 Encoder**: 4.8B params, sentencepiece tokenizer (~0.1s)
- **DiT (2B)**: 28 layers, 2048 hidden, 32×64 heads (~1.1s/step)
- **DiT (13B)**: 48 layers, 4096 hidden, 32×128 heads (~7.0s/step)
- **VAE Decoder**: 553M params, timestep-conditioned 3D convolutions (~2.5s)
- **Scheduler**: FlowMatch Euler with dynamic sigma shifting

Zero-copy weight loading via `mx.load` (memory-mapped safetensors).

## License

Apache-2.0. Model weights are subject to [Lightricks' license](https://huggingface.co/Lightricks/LTX-Video).
