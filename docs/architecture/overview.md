# Architecture

The project implements multiple character-level language models with a shared interface and a pluggable architecture. Each unique character in the training data maps to an integer index via a `CharTokenizer`. Tokenizers, embeddings, and processing blocks are all designed as independent, composable plugins.

## Model Hierarchy

```
BaseLanguageModel (nn.Module)        ← 共享接口：forward() + generate()
├── BigramLanguageModel              ← 只看前 1 个字，纯 Embedding 查表
├── SelfAttentionLanguageModel       ← 能看前 block_size 个字，含自注意力机制
│   └── Head                         ← 单头自注意力（Q/K/V + 因果遮罩）
└── AssembledModel                   ← 积木式模型，通过 Block 列表自由组装
    └── Block × N                    ← 通用积木（固定流程 + 可插拔算法组件）
        ├── op = MultiHeadAttention  ← 注意力算法（含 Dropout）
        │       └── Head × n_head    ← 多个独立的注意力头
        └── op = FeedForward         ← 前馈网络算法（含 Dropout）
```

Block 组装示例（通过 build_op 工厂创建不同算法组件）：
- `["attention"]` (n_head=4) → 1 个注意力积木
- `["attention", "ffn"]` (n_head=1) → 注意力 + FFN（1层）
- `["attention", "ffn"]` (n_head=4) → 注意力 + FFN（1层）
- `["attention", "ffn"]` (n_head=6, n_layer=6, dropout=0.2) → Mini-GPT（6层 × 2积木 = 12个 Block）

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
  - `Block`: Generic building block with fixed workflow (LayerNorm → op → residual). The `op` is a pluggable algorithm component.
  - `AssembledModel`: Takes a list of Block instances and chains them between embedding and output layers.
  - `build_op()`: Algorithm component factory. Creates pluggable ops ("attention" → MultiHeadAttention, "ffn" → FeedForward).
  - `build_blocks()`: Block assembly factory. Creates Block list from config, supports `n_layer` for multi-layer stacking and `dropout`.
  - `MODEL_REGISTRY`: Dict mapping model type names to classes.
- **train.py** — Loads `data/input.txt`, builds char↔int vocab mappings (`stoi`/`itos`), splits 90/10 train/val, trains with AdamW, saves checkpoint. Supports `--model-type` to select between models.
- **generate.py** — CLI that loads a checkpoint, auto-detects model type from saved `model_type` field, and generates text.

## Key Details

- Training data is 《三国演义》(~1.8MB, Chinese classical literature)
- The model auto-selects CUDA if available, otherwise CPU
- Checkpoints include `model_type` and `config` for automatic model reconstruction
