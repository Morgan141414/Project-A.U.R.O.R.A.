"""CLI entry point for a small autonomous smoke-training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from training.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Project A.U.R.O.R.A.'s local decoder-only Transformer")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    best = train({**vars(args), "root": str(root)})
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
