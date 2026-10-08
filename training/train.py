"""Autonomous training loop for the local decoder-only Transformer."""

from __future__ import annotations

import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.tokenizer import CharacterTokenizer
from models.transformer import DecoderOnlyTransformer, TransformerConfig
from training.dataset_loader import build_datasets, load_corpus
from utils.hardware import get_device, memory_usage
from utils.logger import MetricsLogger


def evaluate(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in loader:
            _, loss = model(inputs.to(device), targets.to(device))
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def train(config: dict) -> Path:
    root = Path(config["root"]).resolve()
    device = get_device()
    torch.manual_seed(config["seed"])
    corpus_path = Path(config.get("corpus_path", root / "data" / "corpus.txt"))
    text = load_corpus(corpus_path.parent, corpus_path.name)
    tokenizer = CharacterTokenizer(text=text)
    tokenizer.save(root / "data" / "tokenizer.json")
    token_ids = tokenizer.encode(text)
    train_set, val_set = build_datasets(token_ids, config["block_size"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

    model_config = TransformerConfig(vocab_size=tokenizer.vocab_size, block_size=config["block_size"], n_layer=config["n_layer"], n_head=config["n_head"], n_embd=config["n_embd"], dropout=config["dropout"])
    model = DecoderOnlyTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    grad_accum = max(1, int(config.get("gradient_accumulation_steps", 1)))
    updates_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_updates = max(1, config["epochs"] * updates_per_epoch)
    warmup_steps = max(1, int(total_updates * float(config.get("warmup_ratio", 0.05))))
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    logger = MetricsLogger(root / "logs" / "metrics.jsonl")
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    best_val = float("inf")
    stale_epochs = 0
    patience = max(0, int(config.get("early_stopping_patience", 0)))
    print(f"A.U.R.O.R.A. device={device} train_windows={len(train_set)} val_windows={len(val_set)} vocab={tokenizer.vocab_size}")

    for epoch in range(1, config["epochs"] + 1):
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config['epochs']}", unit="batch")
        total_loss = 0.0
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (inputs, targets) in enumerate(progress):
            _, loss = model(inputs.to(device), targets.to(device))
            (loss / grad_accum).backward()
            if (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.3f}", ppl=f"{math.exp(min(loss.item(), 20)):.1f}", mem=memory_usage(device))
        train_loss = total_loss / max(1, len(train_loader))
        val_loss = evaluate(model, val_loader, device)
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_perplexity": math.exp(min(val_loss, 20)), "seconds": time.time() - started, "device": str(device)}
        logger.log(**metrics)
        state = {"model": model.state_dict(), "config": model_config.__dict__, "tokenizer": str(root / "data" / "tokenizer.json"), "epoch": epoch, "val_loss": val_loss, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "training": {"gradient_accumulation_steps": grad_accum, "warmup_steps": warmup_steps}}
        torch.save(state, checkpoints / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            torch.save(state, checkpoints / "best.pt")
        else:
            stale_epochs += 1
            if patience and stale_epochs >= patience:
                print(f"early_stopping=patience_{patience}")
                break
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_ppl={metrics['val_perplexity']:.2f}")
    return checkpoints / "best.pt"
