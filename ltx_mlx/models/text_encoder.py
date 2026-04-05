"""T5v1.1-XXL encoder — pure MLX, no transformers dependency.

Architecture (from HuggingFace config):
    24 layers, d_model=4096, d_kv=64, num_heads=64, d_ff=10240
    Gated-GELU FFN, T5 RMSNorm, relative position bias (shared from block 0)
    vocab_size=32128

Weight format (HuggingFace sharded safetensors):
    shared.weight                                                   (32128, 4096)
    encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight       (4096, 4096)
    encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight  (32, 64)
    encoder.block.{i}.layer.0.layer_norm.weight                     (4096,)
    encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight            (10240, 4096)  gate
    encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight            (10240, 4096)  up
    encoder.block.{i}.layer.1.DenseReluDense.wo.weight              (4096, 10240)  down
    encoder.block.{i}.layer.1.layer_norm.weight                     (4096,)
    encoder.final_layer_norm.weight                                 (4096,)

Uses sentencepiece tokenizer (spiece.model) — 200x faster than transformers.
"""

import json
import math
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ============================================================
# T5 RMSNorm (no bias, no mean subtraction)
# ============================================================

class T5LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.rsqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps)
        return self.weight * (x * rms).astype(self.weight.dtype)


# ============================================================
# T5 Relative Position Bias (shared from block 0)
# ============================================================

class T5RelativePositionBias(nn.Module):
    def __init__(self, num_buckets: int = 32, num_heads: int = 64):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.embedding = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _bucket(relative_position: mx.array, num_buckets: int = 32) -> mx.array:
        # Bidirectional bucketing
        half = num_buckets // 2
        ret = (relative_position > 0).astype(mx.int32) * half
        n = mx.abs(relative_position)

        max_exact = half // 2
        is_small = n < max_exact
        val_large = max_exact + (
            mx.log(n.astype(mx.float32) / max_exact)
            / math.log(half / max_exact)
            * (half - max_exact)
        ).astype(mx.int32)
        val_large = mx.minimum(val_large, mx.full(val_large.shape, half - 1))
        return ret + mx.where(is_small, n, val_large)

    def __call__(self, seq_len: int) -> mx.array:
        pos = mx.arange(seq_len)
        rel = pos[None, :] - pos[:, None]
        buckets = self._bucket(rel, self.num_buckets)
        values = self.embedding(buckets)
        # (L, L, H) -> (1, H, L, L)
        return mx.expand_dims(values.transpose(2, 0, 1), axis=0)


# ============================================================
# T5 Attention (no bias on projections, unscaled)
# ============================================================

class T5Attention(nn.Module):
    def __init__(self, dim: int = 4096, num_heads: int = 64, head_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim

        self.q = nn.Linear(dim, inner, bias=False)
        self.k = nn.Linear(dim, inner, bias=False)
        self.v = nn.Linear(dim, inner, bias=False)
        self.o = nn.Linear(inner, dim, bias=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                 pos_bias: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape
        n, d = self.num_heads, self.head_dim

        q = self.q(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, L, n, d).transpose(0, 2, 1, 3)

        # T5 attention: unscaled (no /sqrt(d))
        attn = q @ k.transpose(0, 1, 3, 2)
        if pos_bias is not None:
            attn = attn + pos_bias
        if mask is not None:
            attn = attn + mask

        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o(out)


# ============================================================
# T5 Gated-GELU FFN
# ============================================================

class T5FFN(nn.Module):
    def __init__(self, dim: int = 4096, dim_ff: int = 10240):
        super().__init__()
        self.wi_0 = nn.Linear(dim, dim_ff, bias=False)  # gate
        self.wi_1 = nn.Linear(dim, dim_ff, bias=False)  # up
        self.wo = nn.Linear(dim_ff, dim, bias=False)     # down

    def __call__(self, x: mx.array) -> mx.array:
        return self.wo(nn.gelu(self.wi_0(x)) * self.wi_1(x))


# ============================================================
# T5 Encoder Block
# ============================================================

class T5Block(nn.Module):
    def __init__(self, dim: int = 4096, num_heads: int = 64,
                 head_dim: int = 64, dim_ff: int = 10240):
        super().__init__()
        self.layer_0_layer_norm = T5LayerNorm(dim)
        self.layer_0_SelfAttention = T5Attention(dim, num_heads, head_dim)
        self.layer_1_layer_norm = T5LayerNorm(dim)
        self.layer_1_DenseReluDense = T5FFN(dim, dim_ff)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None,
                 pos_bias: Optional[mx.array] = None) -> mx.array:
        h = self.layer_0_layer_norm(x)
        x = x + self.layer_0_SelfAttention(h, mask=mask, pos_bias=pos_bias)
        h = self.layer_1_layer_norm(x)
        x = x + self.layer_1_DenseReluDense(h)
        return x


# ============================================================
# Full T5v1.1-XXL Encoder
# ============================================================

class T5Encoder(nn.Module):
    """T5v1.1-XXL encoder stack (24 layers, 4.7B params).

    Uses shared relative position bias from block 0 (standard T5,
    unlike UMT5 which has per-block bias).
    """

    def __init__(self, vocab_size: int = 32128, dim: int = 4096,
                 num_heads: int = 64, head_dim: int = 64,
                 dim_ff: int = 10240, num_layers: int = 24,
                 num_buckets: int = 32):
        super().__init__()
        self.shared = nn.Embedding(vocab_size, dim)
        self.pos_bias = T5RelativePositionBias(num_buckets, num_heads)
        self.blocks = [
            T5Block(dim, num_heads, head_dim, dim_ff)
            for _ in range(num_layers)
        ]
        self.final_layer_norm = T5LayerNorm(dim)

    def __call__(self, input_ids: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = self.shared(input_ids)
        L = x.shape[1]

        # Shared position bias (computed once from block 0's embedding)
        bias = self.pos_bias(L)

        attn_mask = None
        if mask is not None:
            attn_mask = mx.where(
                mask[:, None, None, :] == 0,
                mx.array(float("-inf")),
                mx.array(0.0),
            )

        for block in self.blocks:
            x = block(x, mask=attn_mask, pos_bias=bias)

        return self.final_layer_norm(x)


# ============================================================
# Weight loading + tokenizer
# ============================================================

def load_t5_encoder(model_dir: str, dtype: mx.Dtype = mx.bfloat16) -> T5Encoder:
    """Load T5v1.1-XXL encoder from HuggingFace sharded safetensors.

    Uses mx.load for zero-copy memory-mapped loading (~0.5s vs 3s with PyTorch).

    Args:
        model_dir: Path to text_encoder directory containing:
            - model-00001-of-00004.safetensors ... model-00004-of-00004.safetensors
            - model.safetensors.index.json
            - config.json
    """
    model_dir = Path(model_dir)

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    model = T5Encoder(
        vocab_size=config["vocab_size"],
        dim=config["d_model"],
        num_heads=config["num_heads"],
        head_dim=config["d_kv"],
        dim_ff=config["d_ff"],
        num_layers=config["num_layers"],
    )

    # Load all shards via mx.load (zero-copy safetensors)
    raw_weights = {}
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        raw_weights.update(mx.load(str(shard)))

    # Map HuggingFace keys to our model keys
    mapped = {}
    for hf_key, tensor in raw_weights.items():
        mlx_key = _map_hf_key(hf_key)
        if mlx_key is not None:
            mapped[mlx_key] = tensor.astype(dtype)

    import mlx.utils
    model.update(mlx.utils.tree_unflatten(list(mapped.items())))
    mx.eval(model.parameters())

    import mlx.utils
    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    print(f"  T5 encoder loaded: {n_params/1e9:.1f}B params, {dtype}")
    return model


def _map_hf_key(hf_key: str) -> Optional[str]:
    """Map HuggingFace T5 encoder key to our MLX model key.

    HF format:
        shared.weight -> shared.weight
        encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
            -> blocks.{i}.layer_0_SelfAttention.{q,k,v,o}.weight
        encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight
            -> pos_bias.embedding.weight
        encoder.block.{i}.layer.0.layer_norm.weight
            -> blocks.{i}.layer_0_layer_norm.weight
        encoder.block.{i}.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight
            -> blocks.{i}.layer_1_DenseReluDense.{wi_0,wi_1,wo}.weight
        encoder.block.{i}.layer.1.layer_norm.weight
            -> blocks.{i}.layer_1_layer_norm.weight
        encoder.final_layer_norm.weight -> final_layer_norm.weight
    """
    if hf_key == "shared.weight":
        return "shared.weight"

    if not hf_key.startswith("encoder."):
        return None

    key = hf_key[len("encoder."):]

    # Relative attention bias (only in block 0, shared)
    if "relative_attention_bias.weight" in key:
        return "pos_bias.embedding.weight"

    # Final layer norm
    if key == "final_layer_norm.weight":
        return "final_layer_norm.weight"

    # Block keys: block.{i}.layer.{j}.XXX -> blocks.{i}.layer_{j}_XXX
    if key.startswith("block."):
        # block.0.layer.0.SelfAttention.q.weight -> blocks.0.layer_0_SelfAttention.q.weight
        parts = key.split(".")
        # parts: ['block', '0', 'layer', '0', 'SelfAttention', 'q', 'weight']
        # or:    ['block', '0', 'layer', '0', 'layer_norm', 'weight']
        # or:    ['block', '0', 'layer', '1', 'DenseReluDense', 'wi_0', 'weight']
        block_idx = parts[1]
        layer_idx = parts[3]
        rest = ".".join(parts[4:])

        return f"blocks.{block_idx}.layer_{layer_idx}_{rest}"

    return None


class T5Tokenizer:
    """T5 tokenizer using sentencepiece directly.

    200x faster load than transformers.AutoTokenizer (~15ms vs 3s).
    """

    def __init__(self, model_path: str, max_length: int = 128):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.max_length = max_length
        self.pad_id = 0
        self.eos_id = 1

    def encode(self, text: str) -> tuple[mx.array, mx.array]:
        """Tokenize and return (input_ids, attention_mask) with padding.

        Returns:
            input_ids: (1, max_length) int32
            attention_mask: (1, max_length) int32
        """
        ids = self.sp.Encode(text)
        # Add EOS token
        ids = ids + [self.eos_id]
        # Truncate
        ids = ids[:self.max_length]
        seq_len = len(ids)
        # Pad
        pad_len = self.max_length - seq_len
        input_ids = ids + [self.pad_id] * pad_len
        mask = [1] * seq_len + [0] * pad_len

        return (
            mx.array([input_ids], dtype=mx.int32),
            mx.array([mask], dtype=mx.int32),
        )
