"""Local corpus loading and next-token language-model batches."""

from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import Dataset


DEFAULT_CORPUS = """A small autonomous learning engine begins with a clear signal.\nThe model reads a sequence, predicts the next character, and improves through feedback.\nReliable experiments keep data, checkpoints, and logs together.\n""" * 30


class TextWindowDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, self.tokens.numel() - self.block_size)

    def __getitem__(self, index: int):
        x = self.tokens[index:index + self.block_size]
        y = self.tokens[index + 1:index + self.block_size + 1]
        return x, y


def load_corpus(data_dir: str | Path, filename: str = "corpus.txt") -> str:
    path = Path(data_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(DEFAULT_CORPUS, encoding="utf-8")
    text = path.read_text(encoding="utf-8").strip()
    if len(text) < 100:
        raise ValueError("Corpus must contain at least 100 characters")
    return text


def build_datasets(token_ids: list[int], block_size: int, val_fraction: float = 0.1):
    split = max(block_size + 1, int(len(token_ids) * (1 - val_fraction)))
    split = min(split, len(token_ids) - block_size - 1)
    return TextWindowDataset(token_ids[:split], block_size), TextWindowDataset(token_ids[split:], block_size)
