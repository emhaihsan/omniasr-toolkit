"""Parity check: PyTorch (fairseq2) vs MLX implementation.

Requires both omnilingual-asr and mlx installed (Apple Silicon). Compares the
transcription of the official pipeline against the MLX port on the same audio.

Usage:
    python -m omniasr_mlx.verify --model-card omniASR_LLM_300M_v2 \
        --model-dir omniASR-LLM-300M-v2-mlx --audio sample.wav --lang ind_Latn
"""

from __future__ import annotations

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-card", default="omniASR_LLM_300M_v2")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--lang", default=None)
    args = parser.parse_args()

    # --- Reference (PyTorch, CPU fp32) ---
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    pipeline = ASRInferencePipeline(
        model_card=args.model_card, device="cpu", dtype=torch.float32
    )
    ref = pipeline.transcribe([args.audio], lang=[args.lang] if args.lang else None)[0]
    print(f"PyTorch : {ref}")

    # --- MLX ---
    from omniasr_mlx.generate import transcribe

    mlx_out = transcribe(args.model_dir, [args.audio], args.lang)[0]
    print(f"MLX     : {mlx_out}")

    match = ref.strip() == mlx_out.strip()
    print(f"\nMatch: {match}")
    if not match:
        print("Catatan: beda kecil bisa muncul karena beam search (PyTorch) vs greedy (MLX).")


if __name__ == "__main__":
    main()
