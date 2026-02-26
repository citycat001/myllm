# Architecture

The model is character-level: each unique character in the training data maps to an integer index. There are no subword tokenizers.

## Data Flow

`data/input.txt` → `train.py` (tokenize, train, save) → `bigram_model.pt` → `generate.py` (load, generate)

## Components

- **model.py** — `BigramLanguageModel(nn.Module)`: a single `nn.Embedding(vocab_size, vocab_size)` table. Each token looks up a row of logits for the next token. Includes `forward()` (returns logits + loss) and `generate()` (autoregressive sampling).
- **train.py** — Loads `data/input.txt`, builds char↔int vocab mappings (`stoi`/`itos`), splits 90/10 train/val, trains with AdamW for 10000 steps, saves checkpoint.
- **generate.py** — CLI that loads a checkpoint and generates text. The checkpoint bundles `model_state_dict`, `vocab_size`, `stoi`, and `itos`.

## Key Details

- Training data is 《三国演义》(~1.8MB, Chinese classical literature)
- The model auto-selects CUDA if available, otherwise CPU
