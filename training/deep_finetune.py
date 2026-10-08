"""LoRA fine-tuning for a pretrained causal code model on Apple Silicon or CPU.

Input JSONL records may contain ``text`` or ``instruction`` + ``output`` fields.
This uses a frozen pretrained model and trains only low-rank adapters, which is
the practical deep-learning path for a 16 GB Apple Silicon machine.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.hardware import get_device


class TokenDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length: int):
        self.items = []
        for row in rows:
            text = row.get("text")
            if not text:
                text = f"### Instruction\n{row.get('instruction', '')}\n### Response\n{row.get('output', '')}"
            encoded = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            labels = [token if mask else -100 for token, mask in zip(encoded["input_ids"], encoded["attention_mask"])]
            self.items.append({"input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long), "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 4:
        raise ValueError("Provide at least 4 JSONL examples for train/validation splitting")
    return rows


def evaluate(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def run(args) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(args.project_root).resolve()
    rows = load_rows(Path(args.data))
    random.shuffle(rows)
    split = max(1, int(len(rows) * 0.9))
    train_rows, val_rows = rows[:split], rows[split:] or rows[-1:]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_set = TokenDataset(train_rows, tokenizer, args.max_length)
    val_set = TokenDataset(val_rows, tokenizer, args.max_length)
    device = get_device()
    # float32 is intentionally used on MPS: it is slower than experimental mixed
    # precision but avoids silent unsupported-op fallbacks and NaNs on small Macs.
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.config.use_cache = False
    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM, target_modules=args.target_modules.split(","))
    model = get_peft_model(model, lora).to(device)
    model.print_trainable_parameters()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    accum = max(1, args.gradient_accumulation_steps)
    total_updates = max(1, math.ceil(len(train_loader) / accum) * args.epochs)
    warmup = max(1, int(total_updates * args.warmup_ratio))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_updates - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); running = 0.0
        progress = tqdm(train_loader, desc=f"deep epoch {epoch}/{args.epochs}", unit="batch")
        for index, batch in enumerate(progress):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / accum).backward(); running += float(loss.item())
            if (index + 1) % accum == 0 or index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{loss.item():.3f}", device=str(device))
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch} train_loss={running / max(1, len(train_loader)):.4f} val_loss={val_loss:.4f} ppl={math.exp(min(val_loss, 20)):.2f}")
        if val_loss < best:
            best = val_loss; patience_left = args.patience
            model.save_pretrained(output); tokenizer.save_pretrained(output)
            (output / "metrics.json").write_text(json.dumps({"epoch": epoch, "val_loss": val_loss, "device": str(device), "method": "LoRA"}, indent=2), encoding="utf-8")
        else:
            patience_left -= 1
            if patience_left < 0:
                break
    return output


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune a pretrained code model locally")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--data", required=True, help="JSONL with text or instruction/output records")
    parser.add_argument("--output", default="checkpoints/lora-code")
    parser.add_argument("--epochs", type=int, default=1); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8); parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4); parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=8); parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--max-grad-norm", type=float, default=1.0); parser.add_argument("--patience", type=int, default=2); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); print(f"Starting LoRA fine-tuning on {get_device()} with {args.model}"); print(f"Saved adapter: {run(args)}")


if __name__ == "__main__":
    main()
