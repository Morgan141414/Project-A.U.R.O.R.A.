"""A deterministic, dependency-free character tokenizer for local training."""

from __future__ import annotations

import json
from pathlib import Path


class CharacterTokenizer:
    """Small tokenizer suitable for smoke tests and tiny local corpora."""

    def __init__(self, text: str | None = None, vocab: list[str] | None = None):
        if vocab is None:
            if text is None:
                raise ValueError("Provide either text or vocab")
            vocab = sorted(set(text))
        self.special_tokens = ["<pad>", "<unk>"]
        self.itos = self.special_tokens + list(vocab)
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        self.pad_token_id = self.stoi["<pad>"]
        self.unk_token_id = self.stoi["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(char, self.unk_token_id) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i >= 2 and i < len(self.itos))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocab": self.itos[2:]}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(vocab=payload["vocab"])
