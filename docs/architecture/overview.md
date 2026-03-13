# Architecture

The project implements multiple character-level language models with a shared interface. Each unique character in the training data maps to an integer index. There are no subword tokenizers.

## Model Hierarchy

```
BaseLanguageModel (nn.Module)        ← 共享接口：forward() + generate()
├── BigramLanguageModel              ← 只看前 1 个字，纯 Embedding 查表
└── SelfAttentionLanguageModel       ← 能看前 block_size 个字，含自注意力机制
    └── Head                         ← 单头自注意力（Q/K/V + 因果遮罩）
```

## Data Flow

`data/input.txt` → `train.py` (tokenize, train, save) → `{model_type}_model.pt` → `generate.py` (load, generate)

## Components

- **model.py** — All model classes:
  - `BaseLanguageModel`: Abstract base with shared `generate()` method
  - `BigramLanguageModel`: A single `nn.Embedding(vocab_size, vocab_size)` table. Each token looks up a row of logits for the next token.
  - `Head`: Single self-attention head with Q/K/V projections and causal mask.
  - `SelfAttentionLanguageModel`: Token embedding + position embedding → self-attention → linear output. Uses `n_embd=64` dimensional embeddings and `block_size=256` context window.
  - `MODEL_REGISTRY`: Dict mapping model type names to classes.
- **train.py** — Loads `data/input.txt`, builds char↔int vocab mappings (`stoi`/`itos`), splits 90/10 train/val, trains with AdamW, saves checkpoint. Supports `--model-type` to select between models.
- **generate.py** — CLI that loads a checkpoint, auto-detects model type from saved `model_type` field, and generates text.

## Key Details

- Training data is 《三国演义》(~1.8MB, Chinese classical literature)
- The model auto-selects CUDA if available, otherwise CPU
- Checkpoints include `model_type` and `config` for automatic model reconstruction
