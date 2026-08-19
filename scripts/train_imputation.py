"""Minimal training entrypoint for LD-aware imputation."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from ldattention.tasks.imputation import LDAwareImputationModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_toy_batch(
    batch_size: int,
    seq_len: int,
    missing_rate: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.randint(0, 2, (batch_size, seq_len, 2), dtype=torch.float32)
    positions = torch.linspace(0, 1, seq_len).view(1, seq_len, 1).repeat(batch_size, 1, 1)

    # Generate simple labels from genotype state and hide some values as missing.
    summed = features.sum(dim=-1)
    labels = summed.long().clamp(max=2)
    missing_mask = torch.rand(batch_size, seq_len) < missing_rate
    labels[missing_mask] = -1
    return features, positions, labels


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    model = LDAwareImputationModel(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        features, positions, labels = generate_toy_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            missing_rate=args.missing_rate,
        )
        features = features.to(device)
        positions = positions.to(device)
        labels = labels.to(device)

        logits, _ = model(features, positions)
        valid = labels != -1
        loss = F.cross_entropy(logits[valid], labels[valid])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds[valid] == labels[valid]).float().mean().item()
        print(f"epoch={epoch} loss={loss.item():.4f} acc={acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LD-aware imputation model (toy data).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing_rate", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
