# Commands

## Training

```bash
# 训练 Bigram 模型（默认，保存到 bigram_model.pt）
uv run python train.py

# 训练 Self-Attention 模型（保存到 attention_model.pt）
uv run python train.py --model-type attention

# 训练组装式模型（Block 列表组装）
uv run python train.py --model-type attention_ffn   # 单头注意力 + FFN
uv run python train.py --model-type multihead       # 4头注意力（无 FFN）
uv run python train.py --model-type multihead_ffn   # 4头注意力 + FFN
```

## Text Generation

```bash
# 用 Bigram 模型生成文本（默认）
uv run python generate.py
uv run python generate.py --prompt "却说曹操" --length 200

# 用 Self-Attention 模型生成文本
uv run python generate.py --model attention_model.pt
uv run python generate.py --model attention_model.pt --prompt "却说曹操" --length 200

# 用组装式模型生成文本
uv run python generate.py --model attention_ffn_model.pt
uv run python generate.py --model multihead_model.pt
uv run python generate.py --model multihead_ffn_model.pt --prompt "却说曹操" --length 200
```

## Package Management

Package management uses `uv` (not pip). Add dependencies with `uv add <package>`.
