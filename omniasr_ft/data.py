"""Data utilities: HuggingFace datasets -> fairseq2 ``Seq2SeqBatch``.

Replicates the preprocessing of the official inference pipeline / ASR task:
mono -> resample 16 kHz -> layer-norm normalization, and char tokenization of
the target text. Each batch carries ``example={"lang": [...]}`` which the
LID-conditioned LLM models require during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torch
import torchaudio.functional as AF
from fairseq2.datasets.batch import Seq2SeqBatch
from torch.nn.functional import layer_norm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

SAMPLE_RATE = 16_000


def prepare_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """mono float32 16kHz, zero mean / unit variance (same as official pipeline)."""
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.dim() == 2:
        # (channels, time) or (time, channels) -> mono
        if waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.mean(dim=1)
    if sample_rate != SAMPLE_RATE:
        waveform = AF.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
    return layer_norm(waveform, waveform.shape)


@dataclass
class AsrCollator:
    """Collates examples of the form ``{"audio": {"array", "sampling_rate"}, "text": str, "lang": str}``."""

    token_encoder: Callable[[str], torch.Tensor]
    pad_idx: int
    lang: str | None = None  # default lang if examples have none

    def __call__(self, examples: List[Dict[str, Any]]) -> Seq2SeqBatch:
        wavs, texts, langs = [], [], []
        for ex in examples:
            audio = ex["audio"]
            wavs.append(prepare_waveform(torch.as_tensor(audio["array"]), int(audio["sampling_rate"])))
            texts.append(self.token_encoder(ex["text"]))
            langs.append(ex.get("lang") or self.lang)

        src = pad_sequence(wavs, batch_first=True, padding_value=0.0)
        tgt = pad_sequence(texts, batch_first=True, padding_value=self.pad_idx)

        example: Dict[str, Any] = {}
        if any(l is not None for l in langs):
            example["lang"] = langs

        return Seq2SeqBatch(
            source_seqs=src,
            source_seq_lens=[len(w) for w in wavs],
            target_seqs=tgt,
            target_seq_lens=[len(t) for t in texts],
            example=example,
        )


def normalize_hf_example(
    ex: Dict[str, Any],
    text_column: str = "text",
    audio_column: str = "audio",
    lang: str | None = None,
) -> Dict[str, Any]:
    """Maps an arbitrary HF dataset row to the collator schema."""
    return {
        "audio": ex[audio_column],
        "text": ex[text_column],
        "lang": ex.get("lang") or lang,
    }


def filter_by_duration(
    ex: Dict[str, Any],
    audio_column: str = "audio",
    min_secs: float = 1.0,
    max_secs: float = 30.0,
) -> bool:
    audio = ex[audio_column]
    dur = len(audio["array"]) / audio["sampling_rate"]
    return min_secs <= dur <= max_secs


def build_hf_dataloader(
    dataset,
    token_encoder: Callable[[str], torch.Tensor],
    pad_idx: int,
    batch_size: int = 4,
    text_column: str = "text",
    audio_column: str = "audio",
    lang: str | None = None,
    min_secs: float = 1.0,
    max_secs: float = 30.0,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """Wraps a (non-streaming) HF dataset into a DataLoader yielding ``Seq2SeqBatch``."""
    dataset = dataset.filter(
        lambda ex: filter_by_duration(ex, audio_column, min_secs, max_secs)
    )

    collator = AsrCollator(token_encoder=token_encoder, pad_idx=pad_idx, lang=lang)

    def collate(rows: List[Dict[str, Any]]) -> Seq2SeqBatch:
        rows = [normalize_hf_example(r, text_column, audio_column, lang) for r in rows]
        return collator(rows)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=True,
    )
