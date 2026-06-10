# omniasr-toolkit

Toolkit ringan untuk [Meta Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) di hardware terbatas:

1. **`omniasr_ft/`** — Fine-tune LoRA standalone (tanpa infra FSDP/cluster fairseq2). Muat di Colab T4 gratis.
2. **`notebooks/finetune_omniasr_colab.ipynb`** — Notebook Colab siap jalan: train → eval WER → export → upload HF.
3. **`omniasr_mlx/`** — Konversi checkpoint ke MLX (Apple Silicon) + inference greedy, siap upload ke HuggingFace. **Eksperimental.**

## Install

```bash
pip install -e .            # butuh Python >= 3.10
# Mac: brew install libsndfile
# MLX (hanya Apple Silicon): pip install -e ".[mlx]" sentencepiece
```

## 1. Fine-tune (LoRA)

Model rekomendasi untuk resource terbatas: `omniASR_LLM_300M_v2` (~6 GB download, training LoRA muat di T4 16 GB).

```bash
python -m omniasr_ft.train --config configs/llm_300m_lora.yaml
```

Atau dari Python:

```python
from omniasr_ft.train import FinetuneConfig, train
model, tokenizer = train(FinetuneConfig(dataset=..., lang="ind_Latn", max_steps=1000))
```

Konsep: model di-load dari asset card fairseq2, semua weight dibekukan, LoRA disuntik ke proyeksi attention decoder Llama (`q/k/v/output_proj`). Loss langsung dari `Wav2Vec2LlamaModel.forward` (sama persis dengan recipe resmi `workflows/recipes/wav2vec2/asr`).

Format dataset: HuggingFace dataset dengan kolom audio (`{"array", "sampling_rate"}`) dan kolom teks. `lang` memakai kode omnilingual (`ind_Latn`, `jav_Latn`, dst — lihat `omnilingual_asr.models.wav2vec2_llama.lang_ids`).

### Inference dengan adapter

```bash
python -m omniasr_ft.infer --adapter outputs/omniasr_lora_id/lora_final.pt \
    --audio test.wav --lang ind_Latn
```

### Export + upload ke HuggingFace

```bash
python -m omniasr_ft.export \
    --adapter outputs/omniasr_lora_id/lora_final.pt \
    --name omniASR_LLM_300M_v2_id \
    --push-to-hub username/omniASR-LLM-300M-id
```

Menghasilkan checkpoint merged `.pt` (kompatibel fairseq2) + asset card YAML. Pengguna lain tinggal menaruh YAML di `~/.config/fairseq2/assets/` lalu `ASRInferencePipeline(model_card="omniASR_LLM_300M_v2_id")`.

## 2. Notebook Colab

Buka `notebooks/finetune_omniasr_colab.ipynb` di Colab (runtime T4). Push repo ini ke GitHub-mu dan isi variabel `TOOLKIT_REPO` di cell pertama.

## 3. MLX (Apple Silicon) — eksperimental

```bash
# 1. Download checkpoint (otomatis kalau sudah pernah pakai pipeline PyTorch)
#    ~/.cache/fairseq2/assets/...

# 2. Konversi
python -m omniasr_mlx.convert \
    --checkpoint ~/.cache/fairseq2/assets/<hash>/omniASR-LLM-300M-v2.pt \
    --tokenizer ~/.cache/fairseq2/assets/<hash>/omniASR_tokenizer_written_v2.model \
    --output-dir omniASR-LLM-300M-v2-mlx --dtype float16

# 3. Transcribe
python -m omniasr_mlx.generate --model-dir omniASR-LLM-300M-v2-mlx \
    --audio sample.wav --lang ind_Latn

# 4. Cek paritas vs PyTorch (penting!)
python -m omniasr_mlx.verify --model-dir omniASR-LLM-300M-v2-mlx \
    --audio sample.wav --lang ind_Latn

# 5. Upload ke HF
huggingface-cli upload username/omniASR-LLM-300M-v2-mlx omniASR-LLM-300M-v2-mlx
```

**Status MLX**: arsitektur (wav2vec2 encoder + Llama decoder + syntax LID) sudah diport, tapi belum terverifikasi terhadap checkpoint asli — converter sengaja gagal keras pada key yang tidak dikenal. Jalankan `verify.py` dan laporkan key yang unmapped bila ada.

## Keterbatasan yang diketahui

- **Unsloth tidak didukung**: arsitektur OmniASR bukan model HF Transformers.
- LoRA fine-tune hanya untuk varian `omniASR_LLM_*` (bukan `Unlimited`/streaming, bukan ZS).
- MLX port: non-streaming, greedy decoding (tanpa beam search), `encoder_stacking=1`.
- Training di MPS (Mac) eksperimental — fairseq2 belum resmi support MPS; jalur paling aman adalah Colab.
