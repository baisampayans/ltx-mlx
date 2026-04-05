# LTX-Video Model Weights

## Quick Download (recommended)

```bash
# From the ltx-mlx root directory:
huggingface-cli download Lightricks/LTX-Video --local-dir models/LTX
```

This downloads everything (~40 GB). To download only what you need:

```bash
# 2B model only (~6 GB + ~10 GB text encoder)
huggingface-cli download Lightricks/LTX-Video --include "ltxv-2b-0.9.8-distilled.safetensors" "text_encoder/*" "tokenizer/*" --local-dir models/LTX

# 13B model only (~27 GB + ~10 GB text encoder)
huggingface-cli download Lightricks/LTX-Video --include "LTX 13B/*" "text_encoder/*" "tokenizer/*" --local-dir models/LTX
```

## Manual Download

Download from: https://huggingface.co/Lightricks/LTX-Video

Place files in this folder structure:

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

## License

Model weights are subject to [Lightricks' license](https://huggingface.co/Lightricks/LTX-Video).
