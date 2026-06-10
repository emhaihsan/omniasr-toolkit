"""MLX implementation of the omnilingual-asr LLM model (wav2vec2 encoder + Llama decoder).

EXPERIMENTAL: mirrors the fairseq2 reference architecture (LLM_ASR_LID variant,
encoder_stacking=1, non-streaming). Verify parity with omniasr_mlx/verify.py
before relying on outputs.

Decoder input syntax (same as ``Wav2Vec2LlamaModel.create_default_syntax``):
    [audio embeddings] [<lid marker> <lang id>] <bos> -> autoregressive text -> <eos>
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# wav2vec2 feature extractor (layer-norm variant, as in large_lv60k / 1B / 3B / 7B)
# ---------------------------------------------------------------------------


class ConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel, stride=stride, bias=False)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.stride = stride
        self.kernel = kernel

    def __call__(self, x: mx.array) -> mx.array:  # x: (B, T, C)
        x = self.conv(x)
        x = self.layer_norm(x)
        return nn.gelu(x)


class FeatureExtractor(nn.Module):
    def __init__(self, layer_specs: list[list[int]]):
        super().__init__()
        layers = []
        in_dim = 1
        for out_dim, kernel, stride in layer_specs:
            layers.append(ConvLayer(in_dim, out_dim, kernel, stride))
            in_dim = out_dim
        self.layers = layers

    def __call__(self, waveform: mx.array) -> mx.array:  # (B, T) -> (B, T', 512)
        x = waveform[..., None]
        for layer in self.layers:
            x = layer(x)
        return x

    def output_length(self, n_samples: int) -> int:
        length = n_samples
        for layer in self.layers:
            length = (length - layer.kernel) // layer.stride + 1
        return length


class Frontend(nn.Module):
    """post-extract LayerNorm -> 512->D projection -> conv positional encoding."""

    def __init__(self, feature_dim: int, model_dim: int, pos_kernel: int, pos_groups: int):
        super().__init__()
        self.post_extract_layer_norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, model_dim)
        self.pos_conv = nn.Conv1d(
            model_dim, model_dim, pos_kernel, padding=pos_kernel // 2, groups=pos_groups
        )
        self.pos_kernel = pos_kernel

    def __call__(self, features: mx.array) -> mx.array:  # (B, T, 512) -> (B, T, D)
        x = self.post_extract_layer_norm(features)
        x = self.proj(x)
        pos = self.pos_conv(x)
        if self.pos_kernel % 2 == 0:
            pos = pos[:, :-1]  # remove extra frame from even kernel padding
        pos = nn.gelu(pos)
        return x + pos


# ---------------------------------------------------------------------------
# Transformer encoder (pre-norm, GELU FFN)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, bias: bool = True):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.output_proj = nn.Linear(dim, dim, bias=bias)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, T, D = x.shape
        h = self.heads
        q = self.q_proj(x).reshape(B, T, h, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, h, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, h, -1).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(q.shape[-1]), mask=mask
        )
        return self.output_proj(out.transpose(0, 2, 1, 3).reshape(B, T, D))


class EncoderFFN(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, inner_dim: int):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn = EncoderFFN(dim, inner_dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Encoder(nn.Module):
    def __init__(self, dim: int, num_layers: int, heads: int, inner_dim: int):
        super().__init__()
        self.layers = [EncoderLayer(dim, heads, inner_dim) for _ in range(num_layers)]
        self.final_norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Llama decoder (RMSNorm, RoPE, SwiGLU)
# ---------------------------------------------------------------------------


class DecoderAttention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, rope_theta: float):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.head_dim * kv_heads, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim * kv_heads, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)
        # fairseq2 LLaMA uses the reference (interleaved/complex) RoPE: traditional=True
        self.rope = nn.RoPE(self.head_dim, traditional=True, base=rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, T, D = x.shape
        q = self.q_proj(x).reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.kv_heads, -1).transpose(0, 2, 1, 3)

        offset = cache[0].shape[2] if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        new_cache = (k, v)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.head_dim), mask=mask
        )
        out = self.output_proj(out.transpose(0, 2, 1, 3).reshape(B, T, D))
        return out, new_cache


class DecoderFFN(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inner_dim, bias=False)
        self.up_proj = nn.Linear(dim, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, inner_dim: int, rope_theta: float):
        super().__init__()
        self.attn = DecoderAttention(dim, heads, kv_heads, rope_theta)
        self.attn_norm = nn.RMSNorm(dim)
        self.ffn = DecoderFFN(dim, inner_dim)
        self.ffn_norm = nn.RMSNorm(dim)

    def __call__(self, x, mask=None, cache=None):
        attn_out, new_cache = self.attn(self.attn_norm(x), mask=mask, cache=cache)
        x = x + attn_out
        return x + self.ffn(self.ffn_norm(x)), new_cache


class Decoder(nn.Module):
    def __init__(self, dim, num_layers, heads, kv_heads, inner_dim, rope_theta):
        super().__init__()
        self.layers = [
            DecoderLayer(dim, heads, kv_heads, inner_dim, rope_theta) for _ in range(num_layers)
        ]
        self.final_norm = nn.RMSNorm(dim)

    def __call__(self, x, mask=None, cache=None):
        if cache is None:
            cache = [None] * len(self.layers)
        new_cache = []
        for layer, layer_cache in zip(self.layers, cache):
            x, c = layer(x, mask=mask, cache=layer_cache)
            new_cache.append(c)
        return self.final_norm(x), new_cache


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


@dataclass
class OmniASRConfig:
    raw: dict

    @classmethod
    def load(cls, model_dir: str | Path) -> "OmniASRConfig":
        with open(Path(model_dir) / "config.json") as f:
            return cls(json.load(f))


class OmniASRModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        fe = config["feature_extractor"]
        enc = config["encoder"]
        dec = config["decoder"]

        self.feature_extractor = FeatureExtractor(fe["layers"])
        self.frontend = Frontend(
            fe["layers"][-1][0], enc["model_dim"], enc["pos_conv_kernel"], enc["pos_conv_groups"]
        )
        self.encoder = Encoder(
            enc["model_dim"], enc["num_layers"], enc["num_heads"], enc["ffn_inner_dim"]
        )
        self.encoder_proj = nn.Linear(enc["model_dim"], dec["model_dim"])
        self.text_embed = nn.Embedding(
            config["vocab_size"] + config["n_special_tokens"], dec["model_dim"]
        )
        if config.get("has_lang_embeddings"):
            # lookup table size is stored implicitly in the weights; placeholder dims
            self.lang_embed = nn.Embedding(1, dec["model_dim"])
        self.decoder = Decoder(
            dec["model_dim"], dec["num_layers"], dec["num_heads"],
            dec["num_kv_heads"], dec["ffn_inner_dim"], dec["rope_theta"],
        )
        self.final_proj = nn.Linear(dec["model_dim"], config["vocab_size"], bias=False)

        self.vocab_size = config["vocab_size"]
        self.bos_idx = config["bos_idx"]
        self.eos_idx = config["eos_idx"]
        self.lid_marker = config["vocab_size"]  # special token: vocab_size + 0

    # --- audio ---
    def encode_audio(self, waveform: mx.array) -> mx.array:
        """(B, T) normalized waveform -> (B, T', decoder_dim)."""
        feats = self.feature_extractor(waveform)
        x = self.frontend(feats)
        x = self.encoder(x)
        return self.encoder_proj(x)

    # --- decoding ---
    def build_context(self, audio_embeds: mx.array, lang_id: int | None) -> mx.array:
        """Concatenates [audio] [<lid> <lang>] <bos> embeddings -> decoder prefix."""
        B = audio_embeds.shape[0]
        parts = [audio_embeds]
        if lang_id is not None and "lang_embed" in self:
            lid = self.text_embed(mx.full((B, 1), self.lid_marker, dtype=mx.int32))
            lang = self.lang_embed(mx.full((B, 1), lang_id, dtype=mx.int32))
            parts += [lid, lang]
        bos = self.text_embed(mx.full((B, 1), self.bos_idx, dtype=mx.int32))
        parts.append(bos)
        return mx.concatenate(parts, axis=1)

    def generate(
        self,
        waveform: mx.array,
        lang_id: int | None = None,
        max_tokens: int = 512,
    ) -> list[int]:
        """Greedy decoding for a single utterance. waveform: (T,) normalized."""
        audio_embeds = self.encode_audio(waveform[None])
        x = self.build_context(audio_embeds, lang_id)

        # prefill (full causal attention over the prefix)
        T = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(x.dtype)
        h, cache = self.decoder(x, mask=mask)
        logits = self.final_proj(h[:, -1:])

        tokens: list[int] = []
        for _ in range(max_tokens):
            next_token = int(mx.argmax(logits[0, -1]))
            if next_token == self.eos_idx:
                break
            tokens.append(next_token)
            emb = self.text_embed(mx.array([[next_token]], dtype=mx.int32))
            h, cache = self.decoder(emb, mask=None, cache=cache)
            logits = self.final_proj(h[:, -1:])
        return tokens

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "OmniASRModel":
        model_dir = Path(model_dir)
        config = OmniASRConfig.load(model_dir).raw
        model = cls(config)

        weights = mx.load(str(model_dir / "model.safetensors"))
        # MLX Conv1d expects (out, kernel, in); torch Conv1d is (out, in, kernel)
        fixed = {}
        for k, v in weights.items():
            if ("conv.weight" in k or "pos_conv.weight" in k) and v.ndim == 3:
                v = v.transpose(0, 2, 1)
            fixed[k] = v

        # Re-create lang_embed with the right size before loading
        if config.get("has_lang_embeddings") and "lang_embed.weight" in fixed:
            n_langs, dim = fixed["lang_embed.weight"].shape
            model.lang_embed = nn.Embedding(n_langs, dim)

        model.load_weights(list(fixed.items()), strict=True)
        mx.eval(model.parameters())
        return model


def normalize_waveform(waveform) -> mx.array:
    """Zero-mean unit-variance, same as the reference pipeline."""
    x = mx.array(waveform, dtype=mx.float32)
    mean = mx.mean(x)
    var = mx.var(x)
    return (x - mean) / mx.sqrt(var + 1e-5)
