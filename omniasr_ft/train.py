"""LoRA fine-tuning for omnilingual-asr LLM models on limited hardware.

Bypasses the heavy fairseq2 recipe/FSDP infrastructure: loads the model from
its asset card, injects LoRA into the Llama decoder, and runs a plain PyTorch
training loop. Works on a single GPU (Colab T4/A100) and Apple Silicon (MPS,
experimental).

Usage:
    python -m omniasr_ft.train --config configs/llm_300m_lora.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import torch
import yaml
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.datasets.batch import Seq2SeqBatch
from fairseq2.models.hub import load_model

from omniasr_ft.data import build_hf_dataloader
from omniasr_ft.lora import (
    DEFAULT_TARGET_PATTERNS,
    inject_lora,
    lora_state_dict,
    mark_only_lora_as_trainable,
)


@dataclass
class FinetuneConfig:
    # Model
    model_card: str = "omniASR_LLM_300M_v2"

    # Data (HuggingFace datasets)
    dataset: str = "mozilla-foundation/common_voice_17_0"
    dataset_config: str | None = "id"
    train_split: str = "train"
    valid_split: str | None = "validation"
    text_column: str = "sentence"
    audio_column: str = "audio"
    lang: str = "ind_Latn"  # omnilingual lang code, used for LID conditioning
    max_train_samples: int | None = None
    max_valid_samples: int | None = 200
    min_secs: float = 1.0
    max_secs: float = 30.0

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_TARGET_PATTERNS))
    extra_trainable: List[str] = field(default_factory=list)  # e.g. ["final_proj", "encoder_proj"]

    # Optimization
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    grad_clip: float = 1.0
    seed: int = 42

    # Runtime
    device: str = "auto"  # auto | cuda | mps | cpu
    output_dir: str = "outputs/omniasr_lora"
    save_every: int = 250
    log_every: int = 10
    eval_every: int = 250
    num_workers: int = 2

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FinetuneConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None  # fp32 on mps/cpu for safety


def batch_to_device(batch: Seq2SeqBatch, device: torch.device) -> Seq2SeqBatch:
    return Seq2SeqBatch(
        source_seqs=batch.source_seqs.to(device),
        source_seq_lens=batch.source_seq_lens,
        target_seqs=batch.target_seqs.to(device),
        target_seq_lens=batch.target_seq_lens,
        example=batch.example,
    )


def lr_lambda_factory(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return f


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, max_batches: int = 50) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch_to_device(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = model(batch)
        total += float(loss) / batch.source_seqs.size(0)
        count += 1
    model.train()
    return total / max(1, count)


def train(config: FinetuneConfig):
    from datasets import load_dataset

    torch.manual_seed(config.seed)
    device = pick_device(config.device)
    amp_dtype = autocast_dtype(device)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[omniasr-ft] device={device} amp={amp_dtype} model={config.model_card}")

    # --- Model & tokenizer (weights stay fp32, autocast handles compute dtype) ---
    model = load_model(config.model_card, device=device, dtype=torch.float32)
    tokenizer = load_tokenizer(config.model_card)
    token_encoder = tokenizer.create_encoder()
    pad_idx = tokenizer.vocab_info.pad_idx

    # --- LoRA ---
    wrapped = inject_lora(
        model,
        target_patterns=config.target_patterns,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    n_trainable = mark_only_lora_as_trainable(model, extra_trainable=config.extra_trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[omniasr-ft] LoRA on {len(wrapped)} modules | trainable {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.2f}%)")

    # --- Data ---
    def load_split(split: str, limit: int | None):
        ds = load_dataset(config.dataset, config.dataset_config, split=split)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        return ds

    train_ds = load_split(config.train_split, config.max_train_samples)
    train_loader = build_hf_dataloader(
        train_ds, token_encoder, pad_idx,
        batch_size=config.batch_size,
        text_column=config.text_column,
        audio_column=config.audio_column,
        lang=config.lang,
        min_secs=config.min_secs,
        max_secs=config.max_secs,
        num_workers=config.num_workers,
    )

    valid_loader = None
    if config.valid_split:
        valid_ds = load_split(config.valid_split, config.max_valid_samples)
        valid_loader = build_hf_dataloader(
            valid_ds, token_encoder, pad_idx,
            batch_size=config.batch_size,
            text_column=config.text_column,
            audio_column=config.audio_column,
            lang=config.lang,
            min_secs=config.min_secs,
            max_secs=config.max_secs,
            num_workers=config.num_workers,
            shuffle=False,
        )

    # --- Optimizer ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(config.warmup_steps, config.max_steps)
    )
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    # --- Loop ---
    model.train()
    step, accum, t0 = 0, 0, time.time()
    running_loss = 0.0
    data_iter = iter(train_loader)

    while step < config.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = batch_to_device(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = model(batch) / batch.source_seqs.size(0)

        scaler.scale(loss / config.grad_accum).backward()
        running_loss += float(loss)
        accum += 1

        if accum == config.grad_accum:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            accum = 0
            step += 1

            if step % config.log_every == 0:
                avg = running_loss / (config.log_every * config.grad_accum)
                speed = step / (time.time() - t0)
                print(f"step {step:>6}/{config.max_steps} | loss {avg:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e} | {speed:.2f} steps/s")
                running_loss = 0.0

            if valid_loader is not None and step % config.eval_every == 0:
                val = evaluate(model, valid_loader, device, amp_dtype)
                print(f"step {step:>6} | valid loss {val:.4f}")

            if step % config.save_every == 0 or step == config.max_steps:
                ckpt = out_dir / f"lora_step{step}.pt"
                torch.save(lora_state_dict(model), ckpt)
                print(f"[omniasr-ft] saved {ckpt}")

    # Final artifacts
    torch.save(lora_state_dict(model), out_dir / "lora_final.pt")
    with open(out_dir / "finetune_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)
    print(f"[omniasr-ft] done. Adapter: {out_dir / 'lora_final.pt'}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune omnilingual-asr")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    train(FinetuneConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
