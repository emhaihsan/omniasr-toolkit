"""Export a fine-tuned model: merge LoRA, save a fairseq2-compatible checkpoint,
generate an asset card, and optionally upload everything to HuggingFace Hub.

Usage:
    python -m omniasr_ft.export \
        --model-card omniASR_LLM_300M_v2 \
        --adapter outputs/omniasr_lora/lora_final.pt \
        --name omniASR_LLM_300M_v2_id \
        --output-dir outputs/export \
        [--push-to-hub username/omniASR-LLM-300M-id]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from fairseq2.models.hub import load_model

CARD_TEMPLATE = """\
name: {name}
model_family: wav2vec2_llama
model_arch: {arch}
checkpoint: "file://{checkpoint}"
tokenizer_ref: omniASR_tokenizer_written_v2
"""

# model_card -> model_arch (see omnilingual_asr/cards/models/*.yaml)
ARCH_BY_CARD = {
    "omniASR_LLM_300M": "300m",
    "omniASR_LLM_1B": "1b",
    "omniASR_LLM_3B": "3b",
    "omniASR_LLM_7B": "7b",
    "omniASR_LLM_300M_v2": "300m_v2",
    "omniASR_LLM_1B_v2": "1b_v2",
    "omniASR_LLM_3B_v2": "3b_v2",
    "omniASR_LLM_7B_v2": "7b_v2",
}


def export(
    model_card: str,
    adapter_path: str | Path,
    name: str,
    output_dir: str | Path,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_patterns=None,
) -> Path:
    from omniasr_ft.lora import (
        DEFAULT_TARGET_PATTERNS,
        inject_lora,
        load_lora_state_dict,
        merge_lora,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] loading base model {model_card} (CPU, fp32)")
    model = load_model(model_card, device=torch.device("cpu"), dtype=torch.float32)

    print(f"[export] applying adapter {adapter_path}")
    inject_lora(
        model,
        target_patterns=target_patterns or DEFAULT_TARGET_PATTERNS,
        r=lora_r,
        alpha=lora_alpha,
        dropout=0.0,
    )
    load_lora_state_dict(model, torch.load(adapter_path, map_location="cpu", weights_only=True))
    merge_lora(model)

    ckpt_path = (out_dir / f"{name}.pt").resolve()
    print(f"[export] saving merged checkpoint -> {ckpt_path}")
    torch.save({"model": model.state_dict()}, ckpt_path)

    arch = ARCH_BY_CARD.get(model_card)
    if arch is None:
        raise ValueError(f"Unknown model card {model_card}; add it to ARCH_BY_CARD")

    card_path = out_dir / f"{name}.yaml"
    card_path.write_text(CARD_TEMPLATE.format(name=name, arch=arch, checkpoint=ckpt_path))
    print(f"[export] asset card -> {card_path}")
    print(
        "[export] to use locally: copy the YAML to ~/.config/fairseq2/assets/ then\n"
        f"          ASRInferencePipeline(model_card={name!r})"
    )
    return ckpt_path


def push_to_hub(repo_id: str, output_dir: str | Path, name: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    out_dir = Path(output_dir)
    for fname in (f"{name}.pt", f"{name}.yaml"):
        path = out_dir / fname
        if path.exists():
            print(f"[export] uploading {path} -> {repo_id}")
            api.upload_file(path_or_fileobj=str(path), path_in_repo=fname, repo_id=repo_id)
    readme = out_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {name}\n\nFine-tuned [omnilingual-asr](https://github.com/facebookresearch/omnilingual-asr) "
            f"checkpoint.\n\n## Usage\n\n```bash\npip install omnilingual-asr huggingface_hub\n```\n\n"
            f"```python\nfrom huggingface_hub import hf_hub_download\n"
            f"ckpt = hf_hub_download('{repo_id}', '{name}.pt')\n"
            f"card = hf_hub_download('{repo_id}', '{name}.yaml')\n"
            f"# Edit the YAML 'checkpoint:' line to point at the downloaded .pt,\n"
            f"# copy it to ~/.config/fairseq2/assets/, then:\n"
            f"from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline\n"
            f"pipeline = ASRInferencePipeline(model_card='{name}')\n```\n"
        )
    api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md", repo_id=repo_id)
    print(f"[export] done: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned omnilingual-asr model")
    parser.add_argument("--model-card", default="omniASR_LLM_300M_v2")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", default="outputs/export")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--push-to-hub", default=None, help="HF repo id, e.g. user/model")
    args = parser.parse_args()

    export(args.model_card, args.adapter, args.name, args.output_dir,
           lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    if args.push_to_hub:
        push_to_hub(args.push_to_hub, args.output_dir, args.name)


if __name__ == "__main__":
    main()
