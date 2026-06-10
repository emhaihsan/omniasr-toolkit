"""Quick inference with a LoRA-fine-tuned omnilingual-asr model.

Usage:
    python -m omniasr_ft.infer \
        --model-card omniASR_LLM_300M_v2 \
        --adapter outputs/omniasr_lora/lora_final.pt \
        --audio sample.wav --lang ind_Latn
"""

from __future__ import annotations

import argparse

import torch
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.models.hub import load_model


def load_finetuned_pipeline(
    model_card: str,
    adapter_path: str | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
):
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    from omniasr_ft.lora import inject_lora, load_lora_state_dict, merge_lora

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_card, device=torch.device(device), dtype=dtype)
    tokenizer = load_tokenizer(model_card)

    if adapter_path is not None:
        inject_lora(model, r=lora_r, alpha=lora_alpha, dropout=0.0)
        load_lora_state_dict(model, torch.load(adapter_path, map_location=device, weights_only=True))
        merge_lora(model)

    return ASRInferencePipeline(
        model_card=None, model=model, tokenizer=tokenizer, device=device, dtype=dtype
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-card", default="omniASR_LLM_300M_v2")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--lang", default=None)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    pipeline = load_finetuned_pipeline(
        args.model_card, args.adapter, lora_r=args.lora_r, lora_alpha=args.lora_alpha
    )
    langs = [args.lang] * len(args.audio) if args.lang else None
    for path, text in zip(args.audio, pipeline.transcribe(args.audio, lang=langs)):
        print(f"{path}: {text}")


if __name__ == "__main__":
    main()
