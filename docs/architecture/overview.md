# Architecture

The project implements multiple character-level language models with a shared interface and a pluggable architecture. Each unique character in the training data maps to an integer index via a `CharTokenizer`. Tokenizers, embeddings, and processing blocks are all designed as independent, composable plugins.

## Model Hierarchy

```
BaseLanguageModel (nn.Module)        ← 共享接口：forward() + generate()
├── BigramLanguageModel              ← 只看前 1 个字，纯 Embedding 查表
├── SelfAttentionLanguageModel       ← 能看前 block_size 个字，含自注意力机制
│   └── Head                         ← 单头自注意力（Q/K/V + 因果遮罩）
└── AssembledModel                   ← 积木式模型，通过 Block 列表自由组装
    ├── AttentionBlock               ← 注意力插件（LayerNorm + 注意力 + 残差）
    │   └── MultiHeadAttention       ← 统一使用多头注意力（n_head=1 也走此路径）
    │       └── Head × n_head        ← 多个独立的注意力头
    └── FFNBlock                     ← 前馈网络插件（LayerNorm + FFN + 残差）
        └── FeedForward              ← 两层 MLP（展开→ReLU→压缩）
```

Block 组装示例：
- `["attention"]` (n_head=4) → 纯多头注意力
- `["attention", "ffn"]` (n_head=1) → 单头注意力 + FFN
- `["attention", "ffn"]` (n_head=4) → 多头注意力 + FFN（标准 Transformer Block）

## Data Flow

`data/input.txt` → `train.py` (tokenize, train, save) → `{model_type}_model.pt` → `generate.py` (load, generate)

## Components

- **tokenizer.py** — Tokenizer plugins:
  - `CharTokenizer`: Character-level tokenizer (one char = one token). Supports `encode`/`decode`/`to_dict`/`from_dict`.
  - `load_tokenizer()`: Restores a tokenizer from checkpoint data.
  - `TOKENIZER_REGISTRY`: Dict mapping tokenizer type names to classes.
- **model.py** — All model classes and components:
  - `BaseLanguageModel`: Abstract base with shared `generate()` method
  - `BigramLanguageModel`: A single `nn.Embedding(vocab_size, vocab_size)` table. Each token looks up a row of logits for the next token.
  - `Head`: Single self-attention head with Q/K/V projections and causal mask.
  - `SelfAttentionLanguageModel`: Token embedding + position embedding → self-attention → linear output. Uses `n_embd=64` dimensional embeddings and `block_size=256` context window.
  - `TokenEmbedding`: Pure token embedding (token index → vector). No position information.
  - `TokenPositionEmbedding`: Token + position embedding (token index → vector with position).
  - `build_embedding()`: Factory function that creates Embedding plugins from type name.
  - `FeedForward`: Two-layer MLP (expand 4x → ReLU → compress back). Independent of attention.
  - `MultiHeadAttention`: Multiple `Head` instances in parallel + projection layer.
  - `AttentionBlock`: Plug-in block wrapping LayerNorm + attention (single/multi-head) + residual connection.
  - `FFNBlock`: Plug-in block wrapping LayerNorm + FeedForward + residual connection.
  - `AssembledModel`: Takes a list of Block instances and chains them between embedding and output layers.
  - `build_blocks()`: Factory function that creates Block instances from a list of names.
  - `MODEL_REGISTRY`: Dict mapping model type names to classes.
- **train.py** — Loads `data/input.txt`, builds char↔int vocab mappings (`stoi`/`itos`), splits 90/10 train/val, trains with AdamW, saves checkpoint. Supports `--model-type` to select between models.
- **generate.py** — CLI that loads a checkpoint, auto-detects model type from saved `model_type` field, and generates text.

## Key Details

- Training data is 《三国演义》(~1.8MB, Chinese classical literature)
- The model auto-selects CUDA if available, otherwise CPU
- Checkpoints include `model_type` and `config` for automatic model reconstruction
