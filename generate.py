"""
从训练好的 Bigram 语言模型生成文本。

加载保存的 checkpoint，从头或从给定的提示词开始生成新文本。

用法：
    uv run python generate.py
    uv run python generate.py --prompt "却说曹操" --length 200
    uv run python generate.py --model bigram_model.pt --length 500
"""

import argparse
import os
import torch
from model import BigramLanguageModel


def main():
    # ======================== 解析命令行参数 ========================
    # argparse 提供干净的 CLI 接口，支持 --help、默认值和类型检查。
    parser = argparse.ArgumentParser(description="从训练好的 Bigram 模型生成文本")
    parser.add_argument("--model", default="bigram_model.pt",
                        help="模型 checkpoint 路径（默认：bigram_model.pt）")
    parser.add_argument("--prompt", default="",
                        help="生成的起始文本（为空则从头开始生成）")
    parser.add_argument("--length", type=int, default=500,
                        help="要生成的字符数（默认：500）")
    args = parser.parse_args()

    # ======================== 加载 Checkpoint ========================
    # checkpoint 包含我们需要的一切：
    #   - model_state_dict：训练好的权重
    #   - vocab_size：用于重建模型架构
    #   - stoi/itos：分词器映射（这样就不需要原始文本文件了）
    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}")
        print("Run 'uv run python train.py' first to train the model.")
        return

    # weights_only=False 因为我们的 checkpoint 包含 Python 字典（stoi/itos），
    # 不仅仅是张量权重。因为文件是我们自己保存的，所以这是安全的。
    checkpoint = torch.load(model_path, weights_only=False)
    vocab_size = checkpoint["vocab_size"]
    stoi = checkpoint["stoi"]  # 字符 → 整数 映射
    itos = checkpoint["itos"]  # 整数 → 字符 映射

    # 从保存的映射重建 encode/decode 函数。
    decode = lambda l: "".join([itos[i] for i in l])  # [1038, 2893] → "曹操"
    encode = lambda s: [stoi[c] for c in s]            # "曹操" → [1038, 2893]

    # 有 GPU 就用 GPU，否则用 CPU。
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ======================== 重建模型 ========================
    # 创建一个相同架构的新模型，然后加载训练好的权重。
    # model.eval() 切换到推理模式 —— 关闭 dropout 和 batchnorm
    #（我们的 Bigram 模型没有这些层，但这是好习惯）。
    model = BigramLanguageModel(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ======================== 准备起始上下文 ========================
    if args.prompt:
        # 将用户输入的提示词编码为 token 索引。
        # 用 [...] 包一层创建 batch 维度：形状 (1, len(prompt))
        context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    else:
        # 没有提示词 —— 从单个零 token 开始（通常是换行或空格）。
        # 形状 (1, 1)：batch 大小 1，序列长度 1。
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # ======================== 生成文本 ========================
    # model.generate() 自回归地生成 max_new_tokens 个字符：
    #   预测下一个字 → 采样 → 拼接 → 重复
    # 输出包含原始上下文 + 所有生成的 token。
    generated = model.generate(context, max_new_tokens=args.length)

    # generated 形状是 (1, 上下文长度 + length)。取第一个（唯一的）batch，
    # 转为 Python 整数列表，再解码回字符串。
    print(decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
