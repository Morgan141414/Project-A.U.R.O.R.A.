# Project A.U.R.O.R.A.

Self-contained local neural-network training engine for experiments with decoder-only Transformers.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_training.py
```

The first run creates `data/corpus.txt` and `data/tokenizer.json`, trains for two epochs, writes JSONL metrics to `logs/metrics.jsonl`, and saves `checkpoints/last.pt` plus the best validation checkpoint at `checkpoints/best.pt`.

The base model is a compact PyTorch decoder-only Transformer with causal multi-head self-attention, RoPE, pre-layer normalization, residual MLP blocks, tied embeddings, cosine learning-rate decay, gradient clipping, and a local character tokenizer. Device selection prefers Apple MPS, then CUDA, then CPU.

For Hugging Face fine-tuning, import `prepare_lora_model` from `models.transformer`. It lazily loads `transformers` and `peft`, so the local smoke test does not need a model download.

## Local interface and directed learning

Start the Scarlet Cyber-Glass application:

```bash
source .venv/bin/activate
python app.py
```

The browser opens automatically. If it does not, open `http://127.0.0.1:8000` manually. You can also run `python start_aurora.py`. The Learning Console accepts a topic and source material (notes, documentation, or code), stores it under `data/learning/`, then starts a background training run. The chat uses the newest local checkpoint. This is incremental training, not instant general intelligence: a scratch character model can demonstrate the pipeline but will not write code like ChatGPT. For serious coding quality, replace the inference backend with a pretrained code model and use this same topic/material pipeline for RAG or LoRA fine-tuning.

All project data, environments, checkpoints, logs, and artifacts are intended to remain under this directory.

## Pretrained code-model fine-tuning

Prepare a JSONL file inside `data/` with either `{"text": "..."}` or `{"instruction": "...", "output": "..."}` records, then run:

```bash
python training/deep_finetune.py \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --data data/code_train.jsonl \
  --output checkpoints/lora-code
```

This uses LoRA adapters rather than updating every base-model weight, FP32 on MPS for reliability, AdamW, 5% warmup, cosine decay, accumulation of eight micro-batches, gradient clipping, validation loss, and early stopping. On a 16 GB M4, start with a 0.5B model and sequence length 512; do not assume QLoRA/bitsandbytes or FP16 are stable on MPS without a separate device probe. The pretrained model and adapter should be kept inside this project by setting `HF_HOME` to a project-local cache before downloading.
