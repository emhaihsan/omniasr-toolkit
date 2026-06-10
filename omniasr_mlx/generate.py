"""Transcribe audio with the MLX port of omnilingual-asr.

Usage:
    python -m omniasr_mlx.generate \
        --model-dir omniASR-LLM-300M-v2-mlx \
        --audio sample.wav --lang ind_Latn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx

from omniasr_mlx.model import OmniASRModel, normalize_waveform

SAMPLE_RATE = 16_000


def load_audio(path: str) -> mx.array:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)  # mono
    if sr != SAMPLE_RATE:
        import numpy as np

        # simple polyphase-free resample via torch if available, else scipy
        try:
            import torch
            import torchaudio.functional as AF

            data = AF.resample(torch.from_numpy(data), sr, SAMPLE_RATE).numpy()
        except ImportError:
            from scipy.signal import resample_poly

            from math import gcd

            g = gcd(sr, SAMPLE_RATE)
            data = resample_poly(data, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return normalize_waveform(data)


def transcribe(model_dir: str, audio_paths: list[str], lang: str | None, max_tokens: int = 512):
    model_dir_p = Path(model_dir)
    model = OmniASRModel.from_pretrained(model_dir_p)

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=str(model_dir_p / "tokenizer.model"))

    lang_id = None
    if lang is not None:
        lang_map_path = model_dir_p / "lang_map.json"
        if lang_map_path.exists():
            lang_map = json.loads(lang_map_path.read_text())
            lang_id = lang_map.get(lang.lower())
            if lang_id is None:
                print(f"[warn] lang {lang!r} not in lang_map.json; decoding without LID")
        else:
            print("[warn] lang_map.json missing; decoding without LID")

    results = []
    for path in audio_paths:
        waveform = load_audio(path)
        tokens = model.generate(waveform, lang_id=lang_id, max_tokens=max_tokens)
        results.append(sp.decode(tokens))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--lang", default=None, help="e.g. ind_Latn, eng_Latn")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    for path, text in zip(args.audio, transcribe(args.model_dir, args.audio, args.lang, args.max_tokens)):
        print(f"{path}: {text}")


if __name__ == "__main__":
    main()
