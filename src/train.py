from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from model import build_word_vocab, load_ud_ewt


class WordCharDataset(Dataset):
    """Simple dataset for word-level character-aware language modeling."""

    def __init__(self, corpus_path: str, max_chars: int = 20, min_word_freq: int = 1):
        self.word_to_idx, self.idx_to_word, self.char_to_idx, self.idx_to_char, self.char_matrix, self.labels = load_ud_ewt(
            corpus_path, max_chars=max_chars, min_word_freq=min_word_freq
        )
        self.max_chars = max_chars

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        chars = torch.tensor(self.char_matrix[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return chars, label


def collate_batch(batch):
    chars, labels = zip(*batch)
    chars = torch.stack(chars)
    labels = torch.stack(labels)
    return chars, labels


def build_dataloaders(train_path: str, valid_path: Optional[str], batch_size: int = 32, max_chars: int = 20):
    train_ds = WordCharDataset(train_path, max_chars=max_chars)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    valid_loader = None
    if valid_path:
        valid_ds = WordCharDataset(valid_path, max_chars=max_chars)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, valid_loader, train_ds


def main():
    parser = argparse.ArgumentParser(description="Dataset utilities for the character-aware LM.")
    parser.add_argument("--train-file", type=str, required=True, help="Path to the train corpus.")
    parser.add_argument("--valid-file", type=str, default=None, help="Optional validation corpus.")
    parser.add_argument("--max-chars", type=int, default=20)
    args = parser.parse_args()

    train_loader, valid_loader, ds = build_dataloaders(args.train_file, args.valid_file, max_chars=args.max_chars)
    print(f"Train batches: {len(train_loader)}")
    if valid_loader:
        print(f"Validation batches: {len(valid_loader)}")
    print(f"Vocabulary size: {len(ds.word_to_idx)}")


if __name__ == "__main__":
    main()
