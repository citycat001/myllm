"""
用训练好的模型生成文本。

加载之前训练保存的模型文件，给它一个开头（或者不给），
让它自己往下写。

用法：
    uv run python generate.py                              # 从头开始写
    uv run python generate.py --prompt "却说曹操" --length 200  # 从"却说曹操"接着写
"""

import argparse
import os
import torch
from model import BigramLanguageModel


def main():
    # ======================== 解析命令行参数 ========================
    # 让用户可以通过命令行指定开头文字、生成长度等。
    parser = argparse.ArgumentParser(description="用训练好的 Bigram 模型生成文本")
    parser.add_argument("--model", default="bigram_model.pt",
                        help="模型文件路径（默认：bigram_model.pt）")
    parser.add_argument("--prompt", default="",
                        help="生成的开头文字（不填就从头开始写）")
    parser.add_argument("--length", type=int, default=500,
                        help="要生成多少个字（默认：500）")
    args = parser.parse_args()

    # ======================== 加载模型文件 ========================
    # 模型文件里存了四样东西（训练时保存的）：
    #   - 训练好的参数（那张大表）
    #   - 词表大小（多少个不同字符）
    #   - 字→数字 和 数字→字 的对照表
    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}")
        print("Run 'uv run python train.py' first to train the model.")
        return

    # weights_only=False：因为文件里不光有模型参数，还有字典（对照表）。
    # 默认模式只能加载纯参数，加这个选项才能加载字典。
    checkpoint = torch.load(model_path, weights_only=False)
    vocab_size = checkpoint["vocab_size"]
    stoi = checkpoint["stoi"]  # 字 → 数字
    itos = checkpoint["itos"]  # 数字 → 字

    # 重建编码/解码函数
    decode = lambda l: "".join([itos[i] for i in l])  # [1038, 2893] → "曹操"
    encode = lambda s: [stoi[c] for c in s]            # "曹操" → [1038, 2893]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ======================== 重建模型 ========================
    # 先建一个空模型（结构和训练时一样），再把训练好的参数填进去。
    # model.eval() = 切到"使用模式"（关闭训练专用功能）。
    model = BigramLanguageModel(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ======================== 准备开头文字 ========================
    if args.prompt:
        # 用户给了开头，比如"却说曹操" → 转成数字 [xx, xx, xx, xx]
        # 外面套一层列表是因为模型要求输入有 batch 维度：(1, 字数)
        context = torch.tensor([encode(args.prompt)], dtype=torch.long, device=device)
    else:
        # 没给开头 → 从编号 0 的字符开始（通常是换行或空格）
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # ======================== 生成文本 ========================
    # 模型会从开头接着写，每次写一个字，写够指定的长度。
    generated = model.generate(context, max_new_tokens=args.length)

    # generated 是数字序列，用 decode 变回文字再打印出来。
    print(decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
