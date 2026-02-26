"""
Bigram 语言模型的训练脚本。

完整训练流程：
  1. 加载原始文本数据（三国演义）
  2. 构建字符级分词器（字符 ↔ 整数 映射）
  3. 编码文本并划分训练集/验证集
  4. 用 AdamW 优化器训练模型
  5. 保存模型 checkpoint
  6. 生成样本文本验证效果

运行：uv run python train.py
"""

import os
import torch
from model import BigramLanguageModel

# ======================== 超参数 ========================
# 以下参数控制训练行为，每个选择都有对应的原因。

# BATCH_SIZE：每次训练迭代并行处理的独立文本片段数量。
# 64 在梯度稳定性（不会太嘈杂）和 CPU 内存（不会太重）之间取得平衡。
# 太小（如 1）= 梯度噪声大，更新不稳定；太大（如 1024）= CPU 上很慢，收益递减。
BATCH_SIZE = 64

# BLOCK_SIZE：每条训练样本的长度（字符数）。
# 对于 Bigram 模型来说这个参数影响不大（它只看前 1 个字符），
# 但每条样本可以提供 BLOCK_SIZE 个训练对，所以 8 是高效的选择。
# 后续加入注意力机制后，这个参数会变得至关重要。
BLOCK_SIZE = 8

# MAX_STEPS：总训练迭代次数。
# 设为 10000 是因为中文词表（4742 个字符）远大于英文（65 个），
# 模型需要更多步数来学习所有字符间的转移概率。
MAX_STEPS = 10000

# EVAL_INTERVAL：每隔多少步评估一次训练集/验证集的 loss。
# 1000 步评估一次，整个训练过程产生约 10 个数据点 —— 足够观察趋势，
# 又不会因为频繁评估（每次要跑 200 个 batch 的前向传播）拖慢训练。
EVAL_INTERVAL = 1000

# EVAL_ITERS：评估 loss 时取平均的随机 batch 数量。
# 单个 batch 的 loss 波动很大（因为是随机采样的）。
# 平均 200 个 batch 能得到稳定、有代表性的 loss 估计。
EVAL_ITERS = 200

# LEARNING_RATE：参数更新的步长大小。
# 1e-2（0.01）比较激进，但对我们的简单模型（只有一张嵌入表）来说没问题。
# 对于更深的模型（Transformer），需要用更小的学习率（如 1e-3 或 3e-4），
# 因为复杂的 loss 曲面更难用大步长来导航。
LEARNING_RATE = 1e-2

# DEVICE：有 GPU 就用 GPU，否则用 CPU。
# torch.cuda.is_available() 检测是否有 NVIDIA GPU + CUDA 驱动。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =================================================================


# ======================== 加载数据 ========================
# 读取原始文本文件（三国演义，约 1.8MB，约 60 万字符）。
data_path = os.path.join(os.path.dirname(__file__), "data", "input.txt")
with open(data_path, "r") as f:
    text = f.read()


# ======================== 字符级分词器 ========================
# 构建最简单的分词器：一个字符 = 一个 token。
# 不需要任何外部库 —— 只是字符和整数之间的映射。

# 提取所有不重复的字符并排序，排序是为了保证每次运行的映射一致。
chars = sorted(set(text))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")

# stoi（string-to-index）：每个字符映射到一个唯一整数。
# itos（index-to-string）：反向映射，用于把数字解码回文字。
# 例如：stoi["曹"] = 1038，itos[1038] = "曹"
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# encode：字符串 → 整数列表。  "曹操" → [1038, 2893]
# decode：整数列表 → 字符串。  [1038, 2893] → "曹操"
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])


# ======================== 编码数据集 ========================
# 将整篇文本转换为一维的 token 索引张量。
# dtype=torch.long（int64），因为 PyTorch 的 Embedding 层要求整数索引。
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Dataset: {len(data):,} tokens")


# ======================== 训练集 / 验证集划分 ========================
# 90% 用于训练，10% 用于验证。
# 验证集是模型从未训练过的数据 —— 它能告诉我们模型是真的学会了规律（泛化），
# 还是只是在背诵训练数据（过拟合）。
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")


def get_batch(split: str):
    """
    生成一个随机的训练 batch。

    每个样本是一对 (x, y)：
      x = 连续 BLOCK_SIZE 个字符（输入）
      y = 同样的序列向右移一位（目标）

    例如文本"却说曹操引兵追赶"，BLOCK_SIZE=8 时：
      x = [却, 说, 曹, 操, 引, 兵, 追, 赶]
      y = [说, 曹, 操, 引, 兵, 追, 赶, 关]
    这提供了 8 个训练对：(却→说)、(说→曹)、(曹→操)……

    返回：
        x: (BATCH_SIZE, BLOCK_SIZE) 输入张量
        y: (BATCH_SIZE, BLOCK_SIZE) 目标张量
    """
    d = train_data if split == "train" else val_data

    # 在文本中随机选取 BATCH_SIZE 个起始位置。
    # torch.randint 确保不会越过数据末尾。
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))

    # 从每个起始位置截取连续 BLOCK_SIZE 个字符。
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])

    # 目标是同样的窗口向右移一位 —— 即每个位置的"正确下一个字符"。
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])

    # 移动到目标设备（CPU 或 GPU）。
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()  # 评估时不需要计算梯度 —— 节省内存
def estimate_loss(model):
    """
    估算训练集和验证集的平均 loss。

    采样 EVAL_ITERS 个随机 batch 并取平均，
    这比只看一个 batch 要稳定得多（单个 batch 因随机采样波动很大）。

    会将模型切换到 eval 模式（关闭 dropout/batchnorm 等），
    评估完后切回 train 模式。
    """
    model.eval()  # 切换到评估模式
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()  # .item() 从张量中提取标量值
        out[split] = losses.mean().item()
    model.train()  # 切回训练模式
    return out


# ======================== 创建模型 ========================
# 实例化 Bigram 模型并移到目标设备。
# 对于中文数据集：vocab_size=4742，所以嵌入表是 4742×4742
# = 约 2250 万参数。整个模型就只有这一张表。
model = BigramLanguageModel(vocab_size).to(DEVICE)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")
print(f"Device: {DEVICE}")
print()


# ======================== 优化器 ========================
# AdamW = Adam + 修正的权重衰减。
# 为什么选 AdamW 而不是普通 SGD？
#   - Adam 为每个参数维护独立的动量（平滑噪声梯度）
#   - Adam 为每个参数自适应调整学习率（频繁更新的参数自动减速）
#   - AdamW 修正了权重衰减与自适应学习率的交互方式
#   - GPT 等现代 LLM 的训练都使用 AdamW，它几乎是默认选择
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# ======================== 训练循环 ========================
# 所有深度学习训练都遵循的核心循环：
#   1. 采样一批数据
#   2. 前向传播：计算预测值和 loss
#   3. 反向传播：计算梯度（每个参数应该怎么调整）
#   4. 优化器更新：根据梯度更新参数
print("Training...")
for step in range(MAX_STEPS):

    # 定期评估训练集和验证集的 loss，监控训练进度。
    # 最后一步也评估，看最终 loss。
    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss(model)
        print(f"  step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    # 第 1 步：获取一个随机 batch 的训练数据。
    xb, yb = get_batch("train")

    # 第 2 步：前向传播 —— 把输入喂进模型，得到预测值和 loss。
    _, loss = model(xb, yb)

    # 第 3a 步：清零上一步的梯度。
    # set_to_none=True 比填零稍快 —— 它直接释放梯度张量，而不是写零。
    optimizer.zero_grad(set_to_none=True)

    # 第 3b 步：反向传播 —— 计算 loss 对每个参数的梯度。
    # PyTorch 的 autograd 引擎会沿着计算图反向遍历。
    loss.backward()

    # 第 4 步：根据计算出的梯度更新所有参数。
    # 每个参数沿着减小 loss 的方向移动一小步。
    optimizer.step()

print("\nTraining complete!")


# ======================== 保存 Checkpoint ========================
# 保存后续加载模型所需的全部信息：
#   - model_state_dict：学到的权重（4742×4742 的嵌入表）
#   - vocab_size：重建模型架构时需要
#   - stoi/itos：分词器映射，这样 generate.py 就不需要原始文本文件
#
# 我们保存 state_dict（仅权重）而不是整个模型对象，因为：
#   - 与代码解耦 —— 即使重命名了类，加载仍然有效
#   - 这是 PyTorch 推荐的做法，可移植性更好
save_path = os.path.join(os.path.dirname(__file__), "bigram_model.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos,
}, save_path)
print(f"Model saved to {save_path}")


# ======================== 快速生成样本 ========================
# 从 token 0（通常是换行/空格）开始生成 300 个字符。
# 这是一个健全性检查 —— 输出大部分是乱码，
# 因为 Bigram 模型对超过单个字符的上下文毫无理解能力。
print("\n--- Sample generation (300 chars) ---\n")
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # 从 token 0 开始
generated = model.generate(context, max_new_tokens=300)
print(decode(generated[0].tolist()))  # 将 token 索引转换回字符
