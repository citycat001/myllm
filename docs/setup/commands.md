# Commands

## Training

```bash
# 训练 Bigram 模型（默认，保存到 bigram_model.pt）
uv run python train.py

# 训练 Self-Attention 模型（保存到 attention_model.pt）
uv run python train.py --model-type attention
```

## Text Generation

```bash
# 用 Bigram 模型生成文本（默认）
uv run python generate.py
uv run python generate.py --prompt "却说曹操" --length 200

# 用 Self-Attention 模型生成文本
uv run python generate.py --model attention_model.pt
uv run python generate.py --model attention_model.pt --prompt "却说曹操" --length 200
```

## Package Management

Package management uses `uv` (not pip). Add dependencies with `uv add <package>`.
