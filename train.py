"""
语言模型的训练脚本。

支持训练不同类型的模型：
  - bigram:    只看前一个字猜下一个字（最简单的基线模型）
  - attention: 加入自注意力机制，能看到更多上下文

用法：
  uv run python train.py                        # 默认训练 bigram 模型
  uv run python train.py --model-type attention  # 训练 attention 模型

训练流程（不管哪个模型都一样）：
  1. 读取《三国演义》全文
  2. 给每个字编号（分词）
  3. 把文本分成训练集和验证集
  4. 让模型反复猜"下一个字是什么"，猜错了就纠正（训练）
  5. 把学到的结果保存成文件
  6. 试着让模型写一段话，看看效果
"""

import argparse
import os
import torch
from model import MODEL_REGISTRY

# ======================== 命令行参数 ========================
parser = argparse.ArgumentParser(description="训练语言模型")
parser.add_argument(
    "--model-type",
    choices=list(MODEL_REGISTRY.keys()),
    default="bigram",
    help="模型类型（默认：bigram）",
)
args = parser.parse_args()
model_type = args.model_type


# ======================== 超参数 ========================
# 超参数 = 训练前人为设定的参数，不是模型自己学出来的。
# 它们决定了"怎么训练"，就像做菜时的火候和时间。
#
# 不同的模型需要不同的超参数 —— 好比小火炖汤和大火爆炒用的火候不同。

if model_type == "bigram":
    # ----- Bigram 模型的超参数 -----
    BATCH_SIZE = 64       # 每次训练同时看 64 条文本片段
    BLOCK_SIZE = 8        # 每条片段 8 个字（Bigram 只看前 1 个字，这个值不太重要）
    MAX_STEPS = 10000     # 总共训练 10000 轮
    LEARNING_RATE = 1e-2  # 学习率 0.01（简单模型可以用大步子走）
    N_EMBD = None         # Bigram 不需要这个参数
else:
    # ----- Self-Attention 模型的超参数 -----
    BATCH_SIZE = 32       # 每次训练同时看 32 条（模型更大，一批不能太多，不然内存不够）
    BLOCK_SIZE = 256      # 每条片段 256 个字（注意力模型能利用更长的上下文）
    MAX_STEPS = 10000     # 总共训练 10000 轮
    LEARNING_RATE = 1e-3  # 学习率 0.001（复杂模型需要更小的步子，不然容易"走偏"）
    N_EMBD = 64           # 每个字用 64 维的向量表示

# 通用超参数
EVAL_INTERVAL = 1000  # 每 1000 轮看一次成绩
EVAL_ITERS = 200      # 看成绩时用 200 组题算平均分
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Model type: {model_type}")
print(f"Hyperparameters: batch_size={BATCH_SIZE}, block_size={BLOCK_SIZE}, "
      f"max_steps={MAX_STEPS}, lr={LEARNING_RATE}")


# ======================== 加载数据 ========================
# 读取《三国演义》全文，约 60 万字。
data_path = os.path.join(os.path.dirname(__file__), "data", "input.txt")
with open(data_path, "r") as f:
    text = f.read()


# ======================== 字符级分词器 ========================
# 电脑不认字，只认数字。所以第一步是给每个字编个号。
#
# 做法很简单：
#   1. 找出文本里所有不重复的字符（排序保证每次结果一样）
#   2. 按顺序编号：第 0 个字符编号 0，第 1 个编号 1……
#   3. 制作两张对照表：字→数字（stoi），数字→字（itos）
#
# 这是最原始的分词方式：一个字 = 一个 token。
# 后面会学更聪明的分词方法（BPE），能把常见的词组合并成一个 token。

chars = sorted(set(text))
vocab_size = len(chars)  # 三国演义里共有 4742 个不同字符
print(f"Vocabulary size: {vocab_size} unique characters")

stoi = {ch: i for i, ch in enumerate(chars)}  # "曹" → 1038
itos = {i: ch for i, ch in enumerate(chars)}  # 1038 → "曹"

encode = lambda s: [stoi[c] for c in s]        # "曹操" → [1038, 2893]
decode = lambda l: "".join([itos[i] for i in l])  # [1038, 2893] → "曹操"


# ======================== 编码数据集 ========================
# 把整篇文章从文字变成数字序列，存成 PyTorch 张量。
# dtype=torch.long 表示用整数（因为编号是整数）。
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Dataset: {len(data):,} tokens")


# ======================== 训练集 / 验证集划分 ========================
# 把数据分成两份：
#   训练集（前 90%）= 模型用来学习的"教材"
#   验证集（后 10%）= 模型没见过的"考试题"
#
# 为什么要分？如果只看教材上的成绩，模型可能只是在"死记硬背"（过拟合），
# 而不是真正学会了规律。用没见过的考试题才能检验真实水平。
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")


def get_batch(split: str):
    """
    从文本中随机截取一批训练片段。

    打个比方：
      文本 = "却说曹操引兵追赶关公到城下"
      随机选个起点，截 BLOCK_SIZE 个字：
        输入 x = [却, 说, 曹, 操, 引, 兵, 追, 赶]
        答案 y = [说, 曹, 操, 引, 兵, 追, 赶, 关]
      y 就是 x 往后挪一位 —— 每个位置的"正确的下一个字"。
      这样一条样本就提供了 BLOCK_SIZE 个练习。

    一次截 BATCH_SIZE 条这样的片段，打包在一起并行处理。
    """
    d = train_data if split == "train" else val_data

    # 随机选 BATCH_SIZE 个起始位置（不超出文本末尾）
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))

    # 从每个起始位置截取 BLOCK_SIZE 个连续字符作为输入
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])

    # 答案 = 输入向右移一位（每个位置的"正确下一个字"）
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])

    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()  # 考试时不需要算梯度（不用"记纠错笔记"），关掉省内存
def estimate_loss(model):
    """
    给模型做一次"模拟考试"，看看当前水平如何。

    注意：这里虽然也用了 cross_entropy 算 loss，但和训练时的意义完全不同：
      - 训练时：算 loss → backward() 算梯度 → step() 更新参数（"做题→对答案→纠正"）
      - 评估时：算 loss → 只是把分数记下来，不做 backward，不做 step（"考试→看分数→不改卷"）
    cross_entropy 在这里只是一把"尺子"——训练时用它量完会动手纠正，评估时用它量完只是看看。

    做法：随机抽 200 组题，每组算一个分数，最后取平均。
    这样比只看一组题更稳定、更有代表性。
    """
    model.eval()  # 切到"考试模式"（关闭训练专用的功能，如 Dropout）
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            # 这里调用 model(xb, yb) 会算 cross_entropy loss，
            # 但注意：我们只是读取 loss 的数值（.item()），
            # 没有调用 loss.backward()，所以模型参数不会被改动。
            _, loss = model(xb, yb)
            losses[k] = loss.item()  # .item() 把张量里的数字取出来变成普通 Python 数字
        out[split] = losses.mean().item()
    model.train()  # 考完了，切回"学习模式"继续训练
    return out


# ======================== 创建模型 ========================
ModelClass = MODEL_REGISTRY[model_type]

if model_type == "bigram":
    model = ModelClass(vocab_size, block_size=BLOCK_SIZE).to(DEVICE)
else:
    model = ModelClass(vocab_size, n_embd=N_EMBD, block_size=BLOCK_SIZE).to(DEVICE)

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")
print(f"Device: {DEVICE}")
print()


# ======================== 优化器 ========================
# 优化器的职责：根据"猜错的方向"来调整模型参数，让下次猜得更准。
#
# 我们用 AdamW，它是目前训练神经网络最常用的优化器（GPT 也用它）。
# 相比最基础的 SGD（每次固定步长调参），AdamW 更聪明：
#   - 它会记住之前的调整方向（有"惯性"，不容易被一次的错误带偏）
#   - 它会给每个参数自动调节步长（有的参数需要大步走，有的需要小步挪）
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# ======================== 训练循环 ========================
# 训练的本质：让模型不断地做"猜下一个字"的练习，猜错了就纠正。
# 重复 MAX_STEPS 次，每次的步骤：
#   1. 随机抽一批练习题
#   2. 让模型猜答案，算出猜错了多少（loss）
#   3. 算出"每个参数应该往哪个方向调"（梯度）
#   4. 按照梯度的方向微调参数
print("Training...")
for step in range(MAX_STEPS):

    # 每隔 EVAL_INTERVAL 轮看一次训练成绩和考试成绩
    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss(model)
        print(f"  step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    # 第 1 步：随机抽一批练习题
    xb, yb = get_batch("train")

    # 第 2 步：让模型猜，算出猜错了多少（loss 越大 = 猜得越差）
    _, loss = model(xb, yb)

    # 第 3 步：清除上一轮的调整记录（不然会累加，越来越乱）
    # set_to_none=True 比填零更快 —— 直接丢掉旧记录，而不是逐个归零
    optimizer.zero_grad(set_to_none=True)

    # 第 4 步：反向传播 —— PyTorch 自动算出"每个参数该往哪调、调多少"
    # 好比考完试后对答案，知道哪里错了、该怎么改
    loss.backward()

    # 第 5 步：按照算出的方向，微调每个参数
    optimizer.step()

print("\nTraining complete!")


# ======================== 保存模型 ========================
# 把训练好的模型存到文件里，这样下次可以直接加载，不用重新训练。
# 除了模型参数之外，还保存了：
#   - model_type：模型类型（bigram 或 attention），加载时需要知道用哪个类
#   - config：模型配置参数，加载时需要用它重建模型结构
#   - 字→数字的对照表（stoi）和 数字→字的对照表（itos）
#
# 只保存参数（state_dict）而不是整个模型，好处是：
# 即使后来改了代码里的类名，保存的文件照样能用。

# 模型配置：加载时需要这些参数来重建模型
config = {"vocab_size": vocab_size, "block_size": BLOCK_SIZE}
if model_type != "bigram":
    config["n_embd"] = N_EMBD

save_filename = f"{model_type}_model.pt"
save_path = os.path.join(os.path.dirname(__file__), save_filename)
torch.save({
    "model_state_dict": model.state_dict(),
    "model_type": model_type,
    "config": config,
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos,
}, save_path)
print(f"Model saved to {save_path}")


# ======================== 试着写一段话 ========================
# 让训练好的模型从头生成 300 个字，看看效果。
print("\n--- Sample generation (300 chars) ---\n")
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)  # 从编号 0 的字开始
generated = model.generate(context, max_new_tokens=300)
print(decode(generated[0].tolist()))  # 把数字序列变回文字
