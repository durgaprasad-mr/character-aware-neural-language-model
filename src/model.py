from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterEmbedding(nn.Module):
    """Maps a sequence of character IDs to character embeddings."""

    def __init__(self, vocab_size: int, embed_dim: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: [batch, seq_len, max_chars]
        return self.embedding(char_ids)


class CharConvBlock(nn.Module):
    """Convolution block to capture character n-gram patterns."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, char_embed_dim, seq_len]
        x = self.conv(x)
        x = self.relu(x)
        return x


class Highway(nn.Module):
    """Highway network used to refine character-derived representations."""

    def __init__(self, dim: int):
        super().__init__()
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.sigmoid(self.gate(x))
        h = self.relu(self.transform(x))
        return t * h + (1 - t) * x


class CharacterAwareLM(nn.Module):
    """Character-aware neural language model with convolution + highway blocks."""

    def __init__(
        self,
        char_vocab_size: int,
        char_embed_dim: int = 16,
        conv_channels: int = 64,
        hidden_dim: int = 128,
        vocab_size: int = 5000,
        max_chars: int = 20,
        num_layers: int = 2,
    ):
        super().__init__()
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.max_chars = max_chars
        self.vocab_size = vocab_size

        self.char_embedding = CharacterEmbedding(char_vocab_size, char_embed_dim)
        self.char_conv1 = CharConvBlock(char_embed_dim, conv_channels, kernel_size=5)
        self.char_conv2 = CharConvBlock(conv_channels, conv_channels, kernel_size=3)
        self.highway1 = Highway(conv_channels)
        self.highway2 = Highway(conv_channels)

        self.word_proj = nn.Linear(conv_channels, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode_word(self, word_chars: torch.Tensor) -> torch.Tensor:
        """Encode a batch of token character sequences into a single fixed-size representation."""
        # word_chars: [batch, max_chars]
        x = self.char_embedding(word_chars)  # [batch, max_chars, char_embed_dim]
        x = x.transpose(1, 2)  # [batch, char_embed_dim, max_chars]

        x = self.char_conv1(x)
        x = self.char_conv2(x)
        x = x.max(dim=-1).values  # [batch, conv_channels]

        x = self.highway1(x)
        x = self.highway2(x)
        x = self.word_proj(x)
        return x

    def forward(self, word_chars: torch.Tensor) -> torch.Tensor:
        # word_chars: [batch, max_chars]
        h = self.encode_word(word_chars)
        logits = self.output(h)
        return logits


class CharAwareLMWithContext(nn.Module):
    """Simple context-aware variant that consumes previous tokens for next-word prediction."""

    def __init__(
        self,
        char_vocab_size: int,
        vocab_size: int,
        char_embed_dim: int = 16,
        hidden_dim: int = 128,
        max_chars: int = 20,
        context_size: int = 3,
    ):
        super().__init__()
        self.context_size = context_size
        self.char_lm = CharacterAwareLM(
            char_vocab_size=char_vocab_size,
            char_embed_dim=char_embed_dim,
            conv_channels=64,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            max_chars=max_chars,
        )

        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context_words: torch.Tensor, context_char_matrix: torch.Tensor) -> torch.Tensor:
        """
        context_words: [batch, context_size]
        context_char_matrix: [batch, context_size, max_chars]
        """
        batch_size, seq_len = context_words.shape
        encoded = []
        for i in range(seq_len):
            token_chars = context_char_matrix[:, i, :]
            h = self.char_lm.encode_word(token_chars)
            encoded.append(h)

        encoded = torch.stack(encoded, dim=1)  # [batch, seq_len, hidden_dim]
        _, hidden = self.rnn(encoded)
        h = hidden[-1]
        return self.classifier(h)


def build_vocab(tokens: List[str]) -> Tuple[dict, dict]:
    """Build a simple character vocabulary and word vocab from a token list."""
    char_set = set("".join(tokens))
    char_to_idx = {ch: i + 1 for i, ch in enumerate(sorted(char_set))}
    char_to_idx["<pad>"] = 0
    idx_to_char = {v: k for k, v in char_to_idx.items()}

    word_to_idx = {tok: i for i, tok in enumerate(sorted(set(tokens)))}
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    return char_to_idx, idx_to_char, word_to_idx, idx_to_word


def encode_word_chars(word: str, char_to_idx: dict, max_chars: int) -> List[int]:
    chars = [char_to_idx.get(ch, char_to_idx.get("<unk>", 0)) for ch in word]
    if len(chars) > max_chars:
        chars = chars[:max_chars]
    chars = chars + [char_to_idx.get("<pad>", 0)] * (max_chars - len(chars))
    return chars


def perplexity(loss: float) -> float:
    return math.exp(loss)
