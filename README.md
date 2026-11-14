# Character-Aware Neural Language Model

A compact PyTorch implementation of a character-aware neural language model inspired by the character-to-word modeling approach used in modern neural language modeling research. The project is designed to work with the UD EWT corpus and includes a training pipeline, data utilities, and evaluation for perplexity-based benchmarking.

## Highlights

- Character-level token encoding
- Convolutional character encoder
- Highway network refinement
- Word-level LM prediction head
- UD EWT compatible preprocessing
- PyTorch implementation with evaluation loop

## Repository structure

- `src/model.py`: character-aware model definition
- `src/data.py`: tokenization and dataset loading utilities
- `src/train.py`: training and validation pipeline
- `README.md`: usage and project notes

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Prepare a UTF-8 corpus file with one sentence per line using standard whitespace tokenization. For example, a file called `data/ewt_train.txt`:

```bash
mkdir -p data
# add UD EWT training sentences here or replace with your corpus file
```

Then train:

```bash
python src/train.py \
  --train-file data/ewt_train.txt \
  --valid-file data/ewt_valid.txt \
  --save-dir runs/char_lm \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 1e-3
```

You can also use a small in-memory toy dataset:

```bash
python src/train.py --train-file data/mini_corpus.txt --valid-file data/mini_corpus.txt --epochs 5
```

## Model notes

This implementation follows the common research pattern:

1. Character embeddings are produced for each token.
2. A convolution block extracts local character patterns.
3. A highway layer refines the representation.
4. A final projection predicts the next word from the learned character-conditioned representation.

This design is especially useful when modeling rare and out-of-vocabulary words, which is a key motivation for character-aware language modeling.

## Recommended corpus

Use the UD English-EWT dataset or another sentence-tokenized English corpus. The parser in `src/data.py` can be adapted to the UD EWT directory layout if you have a full annotated treebank available.

## License

This project is intended as a research and educational implementation.
