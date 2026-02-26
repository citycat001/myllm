# 从零手搓大语言模型（一）：用三国演义训练你的第一个语言模型

> 这是「从零手搓 LLM」系列的第一篇。我们从最简单的 Bigram（二元）语言模型开始，用《三国演义》全文作为训练数据，在 CPU 上跑通从训练到生成的完整流程。代码不到 200 行，不需要 GPU。

## 什么是语言模型？

语言模型的本质就一句话：**给定前面的文字，预测下一个字是什么。**

当你在手机输入法里打了"今天天气"，它帮你联想出"真好"——这就是一个语言模型在工作。ChatGPT 也是同样的原理，只不过它用了更大的模型、更多的数据、更复杂的结构。

今天我们搭建的是最最最简单的版本：**Bigram 模型**——只看当前这一个字，就预测下一个字。

## 为什么从 Bigram 开始？

你可能会问：只看一个字能有什么用？生成的文字肯定不像话啊？

没错。Bigram 模型的生成质量很差，但这恰恰是它的教学价值：

1. **它足够简单**——整个模型只有一张查找表，没有任何花哨的结构，让你把注意力放在理解流程上
2. **它跑通了完整的 pipeline**——分词、训练、损失计算、反向传播、文本生成，一个不少
3. **它是后续改进的基线**——后面我们加注意力机制、加 Transformer 层，每一步都能对比 Bigram 看到明确的提升

## 先看全貌：一个 LLM 由什么组成？

在动手写代码之前，我们先从最高的抽象层理解大语言模型。不管是 GPT、LLaMA 还是 DeepSeek，所有 LLM 都可以拆成以下几个核心积木：

```
"却说曹操"  ──→  [ Tokenization ]  ──→  [2187, 541, 1038, 2893]
   原始文本        分词：文字变数字            Token 序列

                        ↓

[2187, 541, 1038, 2893]  ──→  [ Embedding ]  ──→  [[0.12, -0.5, ...], ...]
      Token 序列               嵌入：数字变向量          向量序列

                        ↓

[[0.12, -0.5, ...], ...]  ──→  [ Transformer ]  ──→  [[0.87, 0.23, ...], ...]
      向量序列                   模型核心：理解上下文          新的向量序列
                          ┌─────────────────────┐
                          │  Self-Attention      │  ← 让每个字"看到"其他字
                          │  Feed-Forward (FFN)  │  ← 对每个位置做非线性变换
                          │  Layer Norm          │  ← 稳定训练
                          │  × N 层堆叠          │  ← 层数越多，理解越深
                          └─────────────────────┘

                        ↓

[[0.87, 0.23, ...], ...]  ──→  [ Linear + Softmax ]  ──→  "引" (概率最高)
      新的向量序列               输出层：向量变概率              预测下一个字
```

### 六个核心概念

| 概念 | 一句话解释 | 本篇涉及？ |
|------|-----------|:----------:|
| **Tokenization** | 把文字切成数字序列，是所有 NLP 的第一步 | ✅ 字符级分词 |
| **Embedding** | 把离散的 token 编号映射为连续的向量，让模型能做数学运算 | ✅ nn.Embedding |
| **Self-Attention** | 让序列中的每个位置能"看到"并"关注"其他位置，是 Transformer 的灵魂 | ❌ 下一步 |
| **Feed-Forward Network (FFN)** | 对每个位置独立做非线性变换，增加模型的表达能力 | ❌ 下一步 |
| **Loss & Backpropagation** | 衡量预测有多差（Loss），然后反向计算每个参数该怎么调（梯度） | ✅ cross_entropy + backward |
| **Autoregressive Generation** | 逐字生成：预测一个字 → 拼到输入后面 → 再预测下一个字，循环往复 | ✅ generate() |

今天的 Bigram 模型跳过了 Attention 和 FFN（所以它不是 Transformer），但**其余四个概念全部用到了**。这意味着后续升级时，我们只需要在 Embedding 和输出层之间"插入" Transformer 结构，其他部分可以原封不动地复用。

---

## 全景路线图：从 Bigram 到 GPT

理解了核心积木后，我们来看完整的训练流程：

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM 训练全景路线图                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ① 数据准备                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ 收集原始  │ →  │ 清洗/去重 │ →  │ 分词编码  │                  │
│  │ 文本语料  │    │ 预处理    │    │Tokenizer │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│                                                                 │
│  ② 模型架构                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ Embedding │ →  │ Attention │ →  │Transformer│                 │
│  │ 词嵌入层  │    │ 注意力机制│    │ 多层堆叠  │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│                                                                 │
│  ③ 训练过程                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ 前向传播  │ →  │ 计算损失  │ →  │ 反向传播  │                  │
│  │ + 预测    │    │ Loss     │    │ + 更新参数│                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│                                                                 │
│  ④ 生成推理                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ 输入提示  │ →  │ 自回归采样│ →  │ 输出文本  │                  │
│  │ Prompt    │    │ 逐字生成  │    │ Response │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│                                                                 │
│  ⑤ 对齐优化（进阶）                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ SFT      │ →  │ RLHF/DPO │ →  │ 评估测试  │                  │
│  │ 指令微调  │    │ 人类偏好  │    │Benchmark │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 我们的学习计划

| 步骤 | 内容 | 对应路线图 | 状态 |
|------|------|-----------|------|
| **第 1 步 👈 本篇** | **Bigram 模型** | ① 字符级分词 + ② Embedding + ③ 完整训练 + ④ 自回归生成 | **已完成** |
| 第 2 步 | Self-Attention | ② 加入注意力机制，模型能"看到"前文多个字符 | 待做 |
| 第 3 步 | Multi-Head Attention + FFN | ② 多头注意力 + 前馈网络，构成完整 Transformer Block | 待做 |
| 第 4 步 | Mini-GPT | ② 多层 Transformer 堆叠，真正的"小型 GPT" | 待做 |
| 第 5 步 | BPE 分词器 | ① 从字符级升级到子词级分词 | 待做 |
| 第 6 步 | 在自定义数据上微调 | ⑤ 让模型学会特定领域的表达 | 待做 |

可以看到，虽然今天的 Bigram 模型极其简单，但它**已经走通了 ①②③④ 四个大阶段**。后续每一步都是在这个骨架上升级某个模块，而不是推倒重来。这就是为什么我们从 Bigram 开始——它是麻雀虽小、五脏俱全的最小完整实现。

---

## 项目结构

```
myllm/
├── data/input.txt      # 训练数据：《三国演义》全文（~1.8MB）
├── model.py            # 模型定义
├── train.py            # 训练脚本
├── generate.py         # 文本生成脚本
└── bigram_model.pt     # 训练好的模型文件
```

---

## 第一步：字符级分词器

在喂给模型之前，我们要先把文字变成数字。最简单的方式就是**字符级分词**：每个字符（包括汉字、标点、空格）分配一个唯一的整数编号。

```python
chars = sorted(set(text))       # 从文本中提取所有不重复的字符
vocab_size = len(chars)          # 词表大小

stoi = {ch: i for i, ch in enumerate(chars)}  # 字符 → 数字
itos = {i: ch for i, ch in enumerate(chars)}  # 数字 → 字符

encode = lambda s: [stoi[c] for c in s]       # "曹操" → [1234, 2345]
decode = lambda l: "".join([itos[i] for i in l])  # [1234, 2345] → "曹操"
```

《三国演义》全文包含 **4,742 个不同字符**，所以我们的词表大小 `vocab_size = 4742`。

### 为什么用字符级而不是词级？

- **零依赖**：不需要任何分词库，几行代码就搞定
- **概念清晰**：一个字符就是一个 token，没有歧义
- **适合入门**：先理解原理，后面再引入 BPE 等更高级的分词方法

代价是词表相对较大（中文汉字多），但对于学习目的完全可以接受。

---

## 第二步：模型——一张查找表

Bigram 模型的核心思想极其简单：

> 对于每个字符 A，我要学会"A 后面最可能出现哪些字符"。

这本质上就是一张 **vocab_size × vocab_size** 的概率表。第 i 行存的是"字符 i 出现后，下一个字符的概率分布"。

在 PyTorch 里，我们用 `nn.Embedding` 来实现这张表：

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

### nn.Embedding 是什么？

`nn.Embedding(num_embeddings, embedding_dim)` 本质上就是一个**可学习的查找表**，内部维护一个 `(num_embeddings, embedding_dim)` 的权重矩阵。输入一个整数索引 i，它返回第 i 行的向量。

在我们的场景中：
- `num_embeddings = vocab_size = 4742`（有多少个字符）
- `embedding_dim = vocab_size = 4742`（输出一个长度为 vocab_size 的向量，代表下一个字符的 logits）

所以总参数量 = 4742 × 4742 = **22,486,564**（约 2250 万）。整个模型只有这一个参数矩阵。

### 为什么不直接用一个二维数组？

你完全可以用 numpy 数组实现同样的逻辑。使用 `nn.Embedding` 的好处是：
- 它是 PyTorch 的 `nn.Module`，自动参与梯度计算和反向传播
- 输入是整数索引，内部做的就是查表操作，比矩阵乘法更高效
- 后面升级模型时可以无缝替换

---

## 第三步：前向传播与损失函数

```python
def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)     # (B, T, C)

    loss = None
    if targets is not None:
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)      # (B*T, C)
        targets_flat = targets.view(B * T)        # (B*T,)
        loss = F.cross_entropy(logits_flat, targets_flat)

    return logits, loss
```

### 形状解释

- **B** (Batch)：一批有多少条样本
- **T** (Time)：每条样本的长度（序列长度）
- **C** (Classes)：词表大小，即每个位置要在 C 个字符中做选择

输入 `idx` 形状是 `(B, T)`——一批 B 条序列，每条长度 T。经过 Embedding 查表后得到 `(B, T, C)`——每个位置都有 C 个 logit 值，代表"下一个字符可能是什么"的原始分数。

### 为什么用 cross_entropy（交叉熵）？

交叉熵是分类问题的标准损失函数。我们的任务本质上是一个 **4742 类分类问题**：给定当前字符，从 4742 个候选字符中选出正确的下一个字符。

`F.cross_entropy` 内部做了两件事：
1. **Softmax**：把 logits 转换成概率分布（所有值加起来等于 1）
2. **负对数似然（NLL）**：计算 `-log(正确答案的概率)`

直觉上：如果模型对正确答案给了 90% 的概率，loss = -log(0.9) ≈ 0.105（很小）；如果只给了 1% 的概率，loss = -log(0.01) ≈ 4.6（很大）。模型通过最小化这个 loss 来学会给正确答案更高的概率。

### 为什么需要 view/reshape？

PyTorch 的 `F.cross_entropy` 要求输入是 `(N, C)` 形状（N 个样本，C 个类别），而我们的 logits 是三维的 `(B, T, C)`。所以需要把 B 和 T 两个维度展平成一个维度：`(B*T, C)`。targets 同理，从 `(B, T)` 展平为 `(B*T,)`。

---

## 第四步：训练超参数详解

```python
BATCH_SIZE = 64
BLOCK_SIZE = 8
MAX_STEPS = 10000
EVAL_INTERVAL = 1000
EVAL_ITERS = 200
LEARNING_RATE = 1e-2
```

逐个解释：

### BATCH_SIZE = 64

**含义**：每次训练迭代，从数据集中随机抽 64 条独立的文本片段，打包成一批一起处理。

**为什么是 64？**
- 太小（比如 1）：每次只看一条样本，梯度噪声大，参数更新不稳定
- 太大（比如 1024）：CPU 内存和计算压力大，收敛未必更快
- 64 是一个经典的平衡点：梯度估计足够稳定，同时 CPU 处理起来也很轻松

### BLOCK_SIZE = 8

**含义**：每条训练样本的长度是 8 个字符。

**为什么是 8？** 对于 Bigram 模型来说，这个参数其实"不太重要"——因为 Bigram 只看前一个字符，不管上下文有多长。设为 8 是为了：
- 每条样本可以提供 8 个训练 pair（8 个位置，每个位置都产生一个"当前字→下一个字"的学习信号）
- 为后续升级到 Attention 模型保留这个参数的位置

### MAX_STEPS = 10000

**含义**：训练 10,000 轮迭代。

**为什么是 10,000？** 我们的词表有 4,742 个字符，远大于英文数据集的 65 个字符。更大的词表意味着模型需要学习更多的字符间转移概率，所以需要更多的训练步数。从训练日志可以看到，loss 在 10,000 步时仍在缓慢下降，说明这个数字是合理的。

### EVAL_INTERVAL = 1000

**含义**：每隔 1,000 步计算一次训练集和验证集的 loss。

**为什么？** 评估需要额外计算（200 个 batch 的前向传播），有一定开销。太频繁会拖慢训练，太稀疏又看不到训练趋势。每 1,000 步评估一次，在 10,000 步的训练中恰好产生 10 个数据点，足够观察趋势了。

### EVAL_ITERS = 200

**含义**：每次评估时，随机采样 200 个 batch，取 loss 的平均值。

**为什么不只算一个 batch？** 单个 batch 的 loss 波动很大（因为每个 batch 是随机采样的），平均 200 个 batch 能得到更稳定、更有代表性的 loss 估计。

### LEARNING_RATE = 1e-2（0.01）

**含义**：每次参数更新的步长大小。

**为什么用 0.01 而不是更常见的 0.001？**
- Bigram 模型结构极其简单（只有一个嵌入表），loss landscape 也相对简单
- 词表很大（4742），需要更积极地更新才能在有限步数内收敛
- 如果学习率太小，10,000 步可能不够让 loss 充分下降

对于更复杂的模型（比如后面的 Transformer），通常要用更小的学习率（如 1e-3 或 3e-4），因为模型更深、更容易训练不稳定。

---

## 第五步：数据加载与 batch 生成

### 训练/验证集划分

```python
n = int(0.9 * len(data))
train_data = data[:n]     # 前 90%
val_data = data[n:]        # 后 10%
```

**为什么要分验证集？** 如果只看训练集 loss，模型可能只是在"背书"（过拟合），而不是真正学会了字符间的规律。验证集是模型从未训练过的数据，验证集 loss 能反映模型的真实泛化能力。

从我们的训练日志可以验证这一点：

```
step     0 | train loss 8.9392 | val loss 8.9385   # 开始时两者接近
step  9999 | train loss 3.9272 | val loss 5.5611   # 结束时 val 明显高于 train
```

val loss 高于 train loss，说明模型确实有过拟合（在训练数据上表现更好）。这对于 Bigram 模型是正常的——2250 万参数对于一个只靠查表的模型来说偏多，容易记住训练数据的统计特征。

### 随机 batch 采样

```python
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)
```

**工作方式**：
1. `torch.randint` 随机生成 64 个起始位置
2. 从每个起始位置截取连续 8 个字符作为输入 `x`
3. 对应的 `y` 是 `x` 往后移一位——即每个位置的"正确答案"

举例：如果文本是"却说曹操引兵追赶"，BLOCK_SIZE=8：
- x = `却说曹操引兵追赶`
- y = `说曹操引兵追赶关`

每个位置都构成一个训练 pair：(却→说)、(说→曹)、(曹→操)……

### 为什么用随机采样而不是顺序遍历？

随机采样（随机梯度下降的核心思想）的好处：
- 每一步看到的数据都不一样，减少过拟合
- 不需要记录"读到哪里了"，实现更简单
- 对于 Bigram 模型来说，字符间的统计关系是全局的，不依赖出现顺序

---

## 第六步：优化器——AdamW

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
```

### 为什么选 AdamW 而不是 SGD？

**SGD（随机梯度下降）** 是最基础的优化器：`参数 = 参数 - 学习率 × 梯度`。简单粗暴，但有两个问题：
- 所有参数用同一个学习率，但有些参数可能需要更大的更新
- 容易被噪声干扰，更新方向不稳定

**Adam** 在 SGD 基础上加了两个改进：
1. **动量（Momentum）**：记住之前的梯度方向，像滚球一样有惯性，不容易被单次噪声带偏
2. **自适应学习率**：为每个参数维护独立的学习率，更新幅度大的参数自动减速

**AdamW** 是 Adam 的改进版，修正了权重衰减（weight decay）的实现方式。在现代深度学习中，AdamW 几乎是默认选择，GPT 系列模型的训练也使用 AdamW。

### optimizer.zero_grad(set_to_none=True)

每次反向传播前要清零梯度。`set_to_none=True` 比默认的 `set_to_none=False` 更高效——它直接把梯度设为 `None` 而不是填零，省了一次内存写入操作。

---

## 第七步：训练循环

```python
for step in range(MAX_STEPS):
    # 1. 定期评估
    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss(model)
        print(...)

    # 2. 采样一个 batch
    xb, yb = get_batch("train")

    # 3. 前向传播 → 计算 loss
    _, loss = model(xb, yb)

    # 4. 清零梯度
    optimizer.zero_grad(set_to_none=True)

    # 5. 反向传播（计算每个参数的梯度）
    loss.backward()

    # 6. 更新参数
    optimizer.step()
```

这就是所有深度学习训练的核心循环，不管模型多大多复杂，都是这六步。

### estimate_loss 中的 @torch.no_grad()

```python
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    ...
    model.train()
```

**为什么需要？**
- `@torch.no_grad()`：评估时不需要计算梯度，关掉梯度追踪可以节省内存和计算
- `model.eval()`：切换到评估模式（对于有 Dropout、BatchNorm 的模型会改变行为；我们的 Bigram 模型虽然没有这些层，但养成好习惯很重要）
- `model.train()`：评估完切回训练模式

---

## 第八步：文本生成

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, _ = self(idx)
        logits = logits[:, -1, :]           # 只取最后一个位置的预测
        probs = F.softmax(logits, dim=-1)   # logits → 概率分布
        idx_next = torch.multinomial(probs, num_samples=1)  # 按概率采样
        idx = torch.cat([idx, idx_next], dim=1)             # 拼接到序列末尾
    return idx
```

### 关键 API 解释

**F.softmax**：把任意实数向量（logits）转换成概率分布。公式是 `softmax(x_i) = e^(x_i) / Σe^(x_j)`。转换后所有值都在 0~1 之间，且加起来等于 1。

**torch.multinomial**：根据给定的概率分布随机采样。如果"操"的概率是 30%，"军"是 20%，"兵"是 10%……它会按这些概率随机抽一个字符。

### 为什么用采样而不是取概率最大的？

如果每次都取概率最大的字符（贪心解码），生成的文本会非常重复和无聊。随机采样引入了多样性——同样的输入可能产生不同的输出，就像人写文章时也不会每次都用同样的措辞。

后续我们还会介绍 **temperature**（温度）参数来控制采样的随机程度。

---

## 训练结果

```
Vocabulary size: 4742 unique characters
Model parameters: 22,486,564
Device: cpu

step     0 | train loss 8.9392 | val loss 8.9385
step  1000 | train loss 5.2032 | val loss 5.8429
step  5000 | train loss 4.0640 | val loss 5.4034
step  9999 | train loss 3.9272 | val loss 5.5611
```

### 初始 loss 为什么是 8.94？

理论上，如果模型完全随机猜测，对于 4742 个字符的均匀分布，loss = -log(1/4742) = log(4742) ≈ **8.46**。实际初始 loss 8.94 略高于理论值，因为 Embedding 的初始权重不是均匀分布（PyTorch 默认用正态分布初始化），导致初始预测比均匀分布还差一点。

### 生成样本

输入"却说曹操"，生成的文本：

> 却说曹操贼未知苏顒缃毫无言，吏，都督建庙。"玄德曰："江人也。"盖宫为将翻。城，去。岂与德便连日内书遗图敌……

可以看到：
- 出现了三国人物名（曹操、玄德、赵云）——因为模型学到了"操"后面常跟特定字符
- 有"曰"字和引号——学到了古文对话的基本格式
- 但完全不通顺——因为 Bigram 只看一个字，没有任何"理解"上下文的能力

---

## 模型保存与加载

```python
# 保存
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": vocab_size,
    "stoi": stoi,
    "itos": itos,
}, "bigram_model.pt")
```

### 为什么保存字典而不是整个模型？

PyTorch 推荐保存 `state_dict`（参数字典）而不是整个模型对象，原因是：
- 整个模型保存时会序列化类定义，如果代码结构改了可能加载失败
- `state_dict` 只保存参数数值，与代码解耦，更灵活

我们额外保存了 `vocab_size`、`stoi`、`itos`，这样 `generate.py` 加载模型时不需要重新读训练数据就能重建分词器。

---

## 总结

这一篇我们完成了：

| 组件 | 你学到了什么 |
|------|-------------|
| **字符级分词** | 最简单的分词方式：每个字符 → 一个整数 |
| **nn.Embedding** | 可学习的查找表，输入整数索引，输出一个向量 |
| **cross_entropy** | 多分类问题的标准损失函数 |
| **AdamW** | 现代深度学习的默认优化器 |
| **训练循环** | 前向 → 算 loss → 反向 → 更新，所有模型都是这个流程 |
| **自回归生成** | softmax → 采样 → 拼接 → 重复 |

Bigram 模型本身不实用，但它是理解语言模型的最佳起点。下一篇，我们会给模型加上**自注意力机制（Self-Attention）**，让它能够"看到"前面多个字符，生成质量会有质的飞跃。

---

*代码仓库：[GitHub 链接]*
*使用的技术栈：Python 3.10 + PyTorch + uv*
*训练数据：《三国演义》全文（~60 万字符）*
*训练设备：CPU（无需 GPU）*
