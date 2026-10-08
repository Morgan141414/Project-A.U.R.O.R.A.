"""Topic-directed incremental learning facade used by the local UI."""

from __future__ import annotations

import re
from pathlib import Path

from training.train import train


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48] or "topic"


def train_topic(root: str | Path, topic: str, materials: str, epochs: int = 1) -> Path:
    """Persist user-provided study material and train on it.

    The topic is an instruction label, not magic knowledge. Useful learning requires
    quality notes, documentation, code, or dataset text in ``materials``.
    """
    root = Path(root).resolve()
    if len(topic.strip()) < 3:
        raise ValueError("Topic must be at least 3 characters")
    if len(materials.strip()) < 100:
        raise ValueError("Add at least 100 characters of source material or notes")
    folder = root / "data" / "learning" / slugify(topic)
    folder.mkdir(parents=True, exist_ok=True)
    corpus = folder / "corpus.txt"
    corpus.write_text(f"TOPIC: {topic.strip()}\n\n{materials.strip()}\n", encoding="utf-8")
    return train({
        "root": str(root), "corpus_path": str(corpus), "epochs": max(1, min(int(epochs), 10)),
        "batch_size": 16, "block_size": 64, "n_layer": 2, "n_head": 4,
        "n_embd": 128, "learning_rate": 3e-4, "dropout": 0.1, "seed": 42,
        "gradient_accumulation_steps": 2, "warmup_ratio": 0.05, "early_stopping_patience": 2,
    })
