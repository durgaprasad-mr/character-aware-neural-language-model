from __future__ import annotations

import os
from collections import Counter
from typing import List, Tuple


def tokenize_text(text: str) -> List[str]:
    """Basic tokenization for corpus preprocessing."""
    return text.lower().replace("\n", " ").split()


def iter_sentences(path: str) -> List[List[str]]:
    """Read a text corpus and split into sentences."""
    sentences: List[List[str]] = []
    if not os.path.exists(path):
        return sentences

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = tokenize_text(line)
            if tokens:
                sentences.append(tokens)
    return sentences


def build_word_vocab(sentences: List[List[str]], min_freq: int = 1, unk_token: str = "<unk>") -> Tuple[dict, dict]:
    wc = Counter(tok for sent in sentences for tok in sent)
    vocab = {token: idx for idx, (token, count) in enumerate(sorted(wc.items(), key=lambda x: (-x[1], x[0])) if False else [])}

    filtered = [token for token, count in wc.items() if count >= min_freq]
    filtered = sorted(filtered)
    word_to_idx = {token: idx + 1 for idx, token in enumerate(filtered)}
    word_to_idx[unk_token] = 0
    idx_to_word = {idx: token for token, idx in word_to_idx.items()}
    return word_to_idx, idx_to_word


def build_char_vocab(sentences: List[List[str]], unk_token: str = "<unk>") -> Tuple[dict, dict]:
    chars = set()
    for sent in sentences:
        for token in sent:
            chars.update(token)
    chars = sorted(chars)
    char_to_idx = {ch: idx + 1 for idx, ch in enumerate(chars)}
    char_to_idx[unk_token] = 0
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char


def prepare_corpus(path: str, max_chars: int = 20, min_word_freq: int = 1) -> Tuple[dict, dict, dict, dict, List[List[int]], List[int]]:
    """Prepare sentence data into a simple, model-friendly format."""
    sentences = iter_sentences(path)
    if not sentences:
        raise FileNotFoundError(f"No corpus data found at {path}")

    word_to_idx, idx_to_word = build_word_vocab(sentences, min_freq=min_word_freq)
    char_to_idx, idx_to_char = build_char_vocab(sentences)

    encoded_sentences: List[List[int]] = []
    for sent in sentences:
        encoded = [word_to_idx.get(tok, word_to_idx["<unk>"]) for tok in sent]
        encoded_sentences.append(encoded)

    # Build per-word character matrices for the first token of each sentence as a lightweight example.
    char_matrix: List[List[int]] = []
    for sent in sentences:
        if not sent:
            continue
        first_word = sent[0]
        token_chars = []
        for ch in first_word[:max_chars]:
            token_chars.append(char_to_idx.get(ch, char_to_idx["<unk>"]))
        token_chars += [0] * (max_chars - len(token_chars))
        char_matrix.append(token_chars)

    labels = [word_to_idx.get(sent[0], word_to_idx["<unk>"]) for sent in sentences if sent]
    return word_to_idx, idx_to_word, char_to_idx, idx_to_char, char_matrix, labels


def load_ud_ewt(path: str, max_chars: int = 20, min_word_freq: int = 1):
    """Helper wrapper for UD EWT or any text corpus with one sentence per line."""
    return prepare_corpus(path, max_chars=max_chars, min_word_freq=min_word_freq)
