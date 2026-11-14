from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import torch
import torch.nn as nn

from data import build_dataloaders
from model import CharacterAwareLM, perplexity


def evaluate(model: nn.Module, loader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for chars, labels in loader:
            chars = chars.to(device)
            labels = labels.to(device)
            logits = model(chars)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_tokens += labels.size(0)

    return total_loss / max(total_tokens, 1), perplexity(total_loss / max(total_tokens, 1))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, valid_loader, dataset = build_dataloaders(
        args.train_file,
        args.valid_file,
        batch_size=args.batch_size,
        max_chars=args.max_chars,
    )

    vocab_size = len(dataset.word_to_idx)
    char_vocab_size = len(dataset.char_to_idx)

    model = CharacterAwareLM(
        char_vocab_size=char_vocab_size,
        char_embed_dim=args.char_embed_dim,
        conv_channels=args.conv_channels,
        hidden_dim=args.hidden_dim,
        vocab_size=vocab_size,
        max_chars=args.max_chars,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        for chars, labels in train_loader:
            chars = chars.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(chars)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)

        avg_loss = epoch_loss / max(len(train_loader.dataset), 1)
        train_ppl = perplexity(avg_loss)

        if valid_loader is not None:
            val_loss, val_ppl = evaluate(model, valid_loader, device)
            print(
                f"Epoch {epoch}/{args.epochs} | train_loss={avg_loss:.4f} | train_ppl={train_ppl:.4f} | "
                f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.4f} | time={time.time() - start:.2f}s"
            )
        else:
            print(f"Epoch {epoch}/{args.epochs} | train_loss={avg_loss:.4f} | train_ppl={train_ppl:.4f} | time={time.time() - start:.2f}s")

        torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_epoch_{epoch}.pt"))

    print(f"Training complete. Model saved in {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a character-aware neural language model in PyTorch.")
    parser.add_argument("--train-file", type=str, required=True, help="Path to the training corpus.")
    parser.add_argument("--valid-file", type=str, default=None, help="Optional validation corpus.")
    parser.add_argument("--save-dir", type=str, default="runs/char_lm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--char-embed-dim", type=int, default=16)
    parser.add_argument("--conv-channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-chars", type=int, default=20)
    parser.add_argument("--use-cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
