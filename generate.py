"""
用训练好的模型生成文本。

加载之前训练保存的模型文件，给它一个开头（或者不给），
让它自己往下写。支持所有类型的模型（bigram、attention、组装式模型等）。

用法：
    uv run python generate.py                                        # 加载 bigram 模型，从头写
    uv run python generate.py --model attention_model.pt             # 加载 attention 模型
    uv run python generate.py --model multihead_ffn_model.pt         # 加载多头+FFN 模型
    uv run python generate.py --prompt "却说曹操" --length 200       # 从"却说曹操"接着写 200 字
"""

import argparse
import os
import torch
from model import MODEL_REGISTRY, build_embedding, build_blocks
from tokenizer import load_tokenizer, CharTokenizer


def main():
    # ======================== 解析命令行参数 ========================
    parser = argparse.ArgumentParser(description="用训练好的模型生成文本")
    parser.add_argument("--model", default="bigram_model.pt",
                        help="模型文件路径（默认：bigram_model.pt）")
    parser.add_argument("--prompt", default="",
                        help="生成的开头文字（不填就从头开始写）")
    parser.add_argument("--length", type=int, default=500,
                        help="要生成多少个字（默认：500）")
    args = parser.parse_args()

    # ======================== 加载模型文件 ========================
    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}")
        print("Run 'uv run python train.py' first to train the model.")
        return

    # weights_only=False：因为文件里不光有模型参数，还有分词器数据。
    checkpoint = torch.load(model_path, weights_only=False)
    vocab_size = checkpoint["vocab_size"]

    # 恢复分词器：优先用新格式（tokenizer 字典），兼容旧格式（stoi/itos）
    if "tokenizer" in checkpoint:
        tokenizer = load_tokenizer(checkpoint["tokenizer"])
    else:
        # 兼容旧版本的 checkpoint
        tokenizer = CharTokenizer(checkpoint["stoi"], checkpoint["itos"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ======================== 重建模型 ========================
    # 从 checkpoint 中读取模型类型和配置参数，自动选择正确的模型类。
    # 这样不管训练时用的是哪种模型，加载时都能正确重建。
    model_type = checkpoint.get("model_type", "bigram")
    config = checkpoint.get("config", {"vocab_size": vocab_size})

    ModelClass = MODEL_REGISTRY[model_type]
    if "block_names" in config:
        # 组装式模型：根据配置重建嵌入插件和 Block 列表
        block_names = config.pop("block_names")
        embedding_type = config.pop("embedding_type")
        n_head = config.pop("n_head")
        config["embedding"] = build_embedding(
            embedding_type, config["vocab_size"], config["n_embd"], config["block_size"]
        )
        config["blocks"] = build_blocks(
            block_names, config["n_embd"], n_head, config["block_size"]
        )
    model = ModelClass(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # 切到"使用模式"（关闭训练专用功能）

    print(f"Loaded {model_type} model from {args.model}")

    # ======================== 准备开头文字 ========================
    if args.prompt:
        context = torch.tensor(
            [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
        )
    else:
        # 没给开头 → 从编号 0 的字符开始，形状 (1, 1)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # ======================== 生成文本 ========================
    generated = model.generate(context, max_new_tokens=args.length)
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
