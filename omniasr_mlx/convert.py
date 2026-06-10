"""Convert an omnilingual-asr LLM checkpoint (.pt) to MLX safetensors.

Produces a HuggingFace-ready folder:
    model.safetensors   # weights, MLX naming
    config.json         # architecture config inferred from weight shapes
    tokenizer.model     # char sentencepiece tokenizer (copied if provided)
    README.md

Usage:
    python -m omniasr_mlx.convert \
        --checkpoint ~/.cache/fairseq2/assets/omniASR-LLM-300M-v2.pt \
        --output-dir omniASR-LLM-300M-v2-mlx \
        [--tokenizer ~/.cache/fairseq2/assets/omniASR_tokenizer_written_v2.model] \
        [--dtype float16]

The mapping is regex-based and fails loudly on unmapped keys so that
mismatches against future fairseq2 versions are caught immediately.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

# fairseq2 key -> mlx key rewrite rules (applied in order, first match wins)
RENAME_RULES: list[tuple[str, str]] = [
    # --- wav2vec2 conv feature extractor ---
    (r"^encoder_frontend\.feature_extractor\.layers\.(\d+)\.conv\.(weight|bias)$",
     r"feature_extractor.layers.\1.conv.\2"),
    (r"^encoder_frontend\.feature_extractor\.layers\.(\d+)\.layer_norm\.(weight|bias)$",
     r"feature_extractor.layers.\1.layer_norm.\2"),
    (r"^encoder_frontend\.feature_extractor\.layers\.(\d+)\.group_norm\.(weight|bias)$",
     r"feature_extractor.layers.\1.group_norm.\2"),
    # --- frontend: post-extract norm + projection + conv positional encoder ---
    (r"^encoder_frontend\.post_extract_layer_norm\.(weight|bias)$", r"frontend.post_extract_layer_norm.\1"),
    (r"^encoder_frontend\.layer_norm\.(weight|bias)$", r"frontend.post_extract_layer_norm.\1"),
    (r"^encoder_frontend\.model_dim_proj\.(weight|bias)$", r"frontend.proj.\1"),
    (r"^encoder_frontend\.pos_encoder\.conv\.(weight|bias)$", r"frontend.pos_conv.\1"),
    (r"^encoder_frontend\.pos_encoder\.conv\.parametrizations\.weight\.(original0|original1)$",
     r"frontend.pos_conv.weight_\1"),  # weight-norm halves, folded below
    (r"^encoder_frontend\.pos_encoder\.conv\.weight_(g|v)$", r"frontend.pos_conv.weight_norm_\1"),
    # --- transformer encoder ---
    (r"^encoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|output_proj)\.(weight|bias)$",
     r"encoder.layers.\1.attn.\2.\3"),
    (r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$",
     r"encoder.layers.\1.attn_norm.\2"),
    (r"^encoder\.layers\.(\d+)\.ffn\.inner_proj\.(weight|bias)$", r"encoder.layers.\1.ffn.fc1.\2"),
    (r"^encoder\.layers\.(\d+)\.ffn\.output_proj\.(weight|bias)$", r"encoder.layers.\1.ffn.fc2.\2"),
    (r"^encoder\.layers\.(\d+)\.ffn_layer_norm\.(weight|bias)$", r"encoder.layers.\1.ffn_norm.\2"),
    (r"^encoder\.layer_norm\.(weight|bias)$", r"encoder.final_norm.\1"),
    # --- bridge / embeddings ---
    (r"^encoder_proj\.(weight|bias)$", r"encoder_proj.\1"),
    (r"^text_frontend\.weight$", r"text_embed.weight"),
    (r"^lang_embeddings\.weight$", r"lang_embed.weight"),
    (r"^final_proj\.weight$", r"final_proj.weight"),
    # --- llama decoder ---
    (r"^llama_decoder\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|output_proj)\.weight$",
     r"decoder.layers.\1.attn.\2.weight"),
    (r"^llama_decoder\.layers\.(\d+)\.self_attn_layer_norm\.weight$",
     r"decoder.layers.\1.attn_norm.weight"),
    (r"^llama_decoder\.layers\.(\d+)\.ffn\.gate_proj\.weight$", r"decoder.layers.\1.ffn.gate_proj.weight"),
    (r"^llama_decoder\.layers\.(\d+)\.ffn\.inner_proj\.weight$", r"decoder.layers.\1.ffn.up_proj.weight"),
    (r"^llama_decoder\.layers\.(\d+)\.ffn\.output_proj\.weight$", r"decoder.layers.\1.ffn.down_proj.weight"),
    (r"^llama_decoder\.layer_norm\.weight$", r"decoder.final_norm.weight"),
]

SKIP_PATTERNS = [
    r"^masker\.",            # training-only feature masker
    r"num_batches_tracked",
]


def rename_key(key: str) -> str | None:
    for pat in SKIP_PATTERNS:
        if re.search(pat, key):
            return None
    for pat, repl in RENAME_RULES:
        if re.match(pat, key):
            return re.sub(pat, repl, key)
    raise KeyError(f"Unmapped checkpoint key: {key}")


def fold_weight_norm(weights: dict) -> dict:
    """Folds pos_conv weight-norm (g, v) into a plain conv weight."""
    pairs = [
        ("frontend.pos_conv.weight_original0", "frontend.pos_conv.weight_original1"),
        ("frontend.pos_conv.weight_norm_g", "frontend.pos_conv.weight_norm_v"),
    ]
    for g_key, v_key in pairs:
        if g_key in weights and v_key in weights:
            g, v = weights.pop(g_key), weights.pop(v_key)
            norm = v.norm(dim=(0, 1), keepdim=True)
            weights["frontend.pos_conv.weight"] = g * v / norm
    return weights


def infer_config(weights: dict) -> dict:
    """Derives the architecture config from weight shapes."""
    enc_layers = 1 + max(
        int(m.group(1)) for k in weights if (m := re.match(r"encoder\.layers\.(\d+)\.", k))
    )
    dec_layers = 1 + max(
        int(m.group(1)) for k in weights if (m := re.match(r"decoder\.layers\.(\d+)\.", k))
    )
    conv_layers = 1 + max(
        int(m.group(1)) for k in weights if (m := re.match(r"feature_extractor\.layers\.(\d+)\.", k))
    )
    encoder_dim = weights["encoder.layers.0.attn.q_proj.weight"].shape[0]
    decoder_dim = weights["decoder.layers.0.attn.q_proj.weight"].shape[0]
    vocab_size, _ = weights["final_proj.weight"].shape
    n_embeddings = weights["text_embed.weight"].shape[0]

    pos_conv = weights.get("frontend.pos_conv.weight")

    return {
        "model_type": "omniasr_llm",
        "feature_extractor": {
            "num_layers": conv_layers,
            # standard wav2vec2: (dim, kernel, stride)
            "layers": [[512, 10, 5]] + [[512, 3, 2]] * 4 + [[512, 2, 2]] * 2,
        },
        "encoder": {
            "model_dim": int(encoder_dim),
            "num_layers": int(enc_layers),
            "num_heads": 16,
            "ffn_inner_dim": int(weights["encoder.layers.0.ffn.fc1.weight"].shape[0]),
            "pos_conv_kernel": int(pos_conv.shape[-1]) if pos_conv is not None else 128,
            "pos_conv_groups": 16,
        },
        "decoder": {
            "model_dim": int(decoder_dim),
            "num_layers": int(dec_layers),
            "num_heads": 8,
            "num_kv_heads": 8,
            "ffn_inner_dim": int(weights["decoder.layers.0.ffn.gate_proj.weight"].shape[0]),
            "rope_theta": 10000.0,
        },
        "vocab_size": int(vocab_size),
        "n_special_tokens": int(n_embeddings - vocab_size),
        "bos_idx": 0,
        "pad_idx": 1,
        "eos_idx": 2,
        "unk_idx": 3,
        "has_lang_embeddings": "lang_embed.weight" in weights,
        "sample_rate": 16000,
    }


def convert(checkpoint: str, output_dir: str, tokenizer: str | None, dtype: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[convert] loading {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]

    weights: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        new_key = rename_key(key)
        if new_key is not None:
            weights[new_key] = tensor.to(torch.float32)
    weights = fold_weight_norm(weights)

    config = infer_config(weights)
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[convert] config: encoder {config['encoder']['num_layers']}L/"
          f"{config['encoder']['model_dim']}d, decoder {config['decoder']['num_layers']}L/"
          f"{config['decoder']['model_dim']}d, vocab {config['vocab_size']}")

    import mlx.core as mx

    np_dtype = {"float16": np.float16, "float32": np.float32, "bfloat16": None}[dtype]
    mlx_weights = {}
    for k, v in weights.items():
        arr = v.numpy()
        a = mx.array(arr)
        if dtype == "bfloat16":
            a = a.astype(mx.bfloat16)
        elif np_dtype is not None and ("norm" not in k):
            a = a.astype(getattr(mx, dtype))
        mlx_weights[k] = a
    mx.save_safetensors(str(out / "model.safetensors"), mlx_weights)
    print(f"[convert] saved {out / 'model.safetensors'}")

    if tokenizer:
        shutil.copy(tokenizer, out / "tokenizer.model")
        print(f"[convert] copied tokenizer -> {out / 'tokenizer.model'}")

    # Language -> embedding index mapping (for LID conditioning)
    try:
        import pyarrow.parquet as pq
        from omnilingual_asr.models.wav2vec2_llama.factory import LANG_LOOKUP_TABLE_PATH

        table = pq.read_table(LANG_LOOKUP_TABLE_PATH).to_pylist()
        lang_map = {row["lang"].lower(): row["index"] + 1 for row in table}
        with open(out / "lang_map.json", "w") as f:
            json.dump(lang_map, f)
        print(f"[convert] wrote lang_map.json ({len(lang_map)} languages)")
    except Exception as e:  # omnilingual_asr not installed — optional
        print(f"[convert] skipped lang_map.json ({e})")

    readme = out / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Omnilingual ASR — MLX (experimental)\n\n"
            "Converted from [facebookresearch/omnilingual-asr]"
            "(https://github.com/facebookresearch/omnilingual-asr).\n\n"
            "```bash\npip install mlx sentencepiece soundfile\n"
            "python -m omniasr_mlx.generate --model-dir . --audio sample.wav --lang ind_Latn\n```\n"
        )
    print("[convert] done — folder siap di-upload: huggingface-cli upload <repo> " + str(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    convert(args.checkpoint, args.output_dir, args.tokenizer, args.dtype)


if __name__ == "__main__":
    main()
