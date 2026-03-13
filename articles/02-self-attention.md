# 从零手搓大语言模型（二）：给模型装上"眼睛"—— Self-Attention 自注意力机制

> 这是「从零手搓 LLM」系列的第二篇。上一篇我们用 Bigram 模型跑通了完整的训练-生成流程，但模型只看前一个字就猜下一个字，生成的文本完全不通顺。今天我们给模型加上**自注意力机制（Self-Attention）**，让它能"看到"前面更多的字，理解上下文。
>
> 关于 LLM 的整体架构和六个核心概念，请参阅本系列第一篇。

## 本篇在学习计划中的位置

| 步骤 | 内容 | 状态 |
|------|------|:----:|
| 第 1 步 | Bigram 模型 | ✅ 已完成 |
| **第 2 步 👈 本篇** | **Self-Attention** | **进行中** |
| 第 3 步 | Multi-Head Attention + FFN | 待做 |
| 第 4 步 | Mini-GPT | 待做 |
| 第 5 步 | BPE 分词器 | 待做 |
| 第 6 步 | 自定义数据微调 | 待做 |

**本篇新增的核心概念：Self-Attention（自注意力）、Position Embedding（位置编码）**

---

## Bigram 的问题：只看一个字，怎么写得好文章？

上一篇我们训练了一个 Bigram 模型，它的"思考过程"是这样的：

```
看到"曹" → 查表 → "操"的概率最高 → 输出"操"
看到"操" → 查表 → 接下来猜一个字
看到"引" → 查表 → 接下来猜一个字
……
```

问题很明显：**每个字都在"独立作战"，完全不知道前面说了什么。** 就像一群人各自蒙着眼睛写接龙故事，每个人只能看前面一个人写的最后一个字——写出来的东西自然是一团糟。

要写出通顺的文字，模型得有"回头看"的能力。比如处理"兵"这个字时，如果模型能看到前面是"曹操引"，就知道这里说的是"曹操引兵"，下一个字可能是"追"或"赶"。

**Self-Attention 就是给模型装上"眼睛"，让每个字都能回头看前面的所有字。**

---

## Self-Attention 直觉：一场"圆桌会议"

想象你是一个编辑，要在一篇文章中理解每个字的含义。你让文本中的每个字都坐在一张圆桌旁"开会"：

1. **每个字提出一个问题（Query）**："我在找什么样的上下文信息？"
   - 比如"兵"可能在问："谁在带领我？是什么动作？"

2. **每个字举一块牌子（Key）**："我能提供什么信息。"
   - "曹操"的牌子写着："我是一个人物名"
   - "引"的牌子写着："我是一个动作词"

3. **每个字准备一份资料（Value）**："这是我的详细内容。"
   - 这是每个字实际包含的语义信息

4. **匹配与关注**：
   - "兵"的问题和"曹操"、"引"的牌子很匹配 → 多关注它们
   - "兵"的问题和"却说"的牌子不太匹配 → 少关注

5. **融合信息**：按照关注程度，把大家的资料加权汇总
   - "兵"最终得到的表示融合了"曹操引"的上下文 → 它"知道"自己是被曹操引领的兵

**这就是 Self-Attention 的全部。** 三个关键角色：Q（提问）、K（标签）、V（内容），通过 Q 和 K 的匹配来决定关注谁，然后按关注度加权求和 V。

---

## 代码架构升级：继承与统一接口

在动手实现之前，我们先做一个重要的架构升级。上一篇只有一个 `BigramLanguageModel`，现在我们要加第二个模型了。为了让 `train.py` 和 `generate.py` 不需要关心具体用哪个模型，我们抽出一个基类：

```python
class BaseLanguageModel(nn.Module):
    """所有语言模型的基类 —— 统一接口。"""

    block_size: int  # 模型能"看到"多远

    def forward(self, idx, targets=None):
        """子类必须实现：输入 → 预测 + 损失"""
        raise NotImplementedError

    def generate(self, idx, max_new_tokens):
        """所有模型共享的生成逻辑"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # 截取最近的上下文
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
```

这样一来：

```
BaseLanguageModel         ← 定义"规矩"：forward 和 generate
├── BigramLanguageModel   ← 查表就完事（和第一篇一样）
└── SelfAttentionLanguageModel  ← 加了注意力（本篇新增）
```

训练和生成脚本只需要换一行参数就能切换模型：

```bash
uv run python train.py                        # 训练 Bigram（默认）
uv run python train.py --model-type attention  # 训练 Self-Attention
```

这就像汽车换引擎——底盘（训练流程）不用动，只是发动机（模型）升级了。

### generate 方法的一个细节：block_size 截断

注意 `generate` 中有一行 `idx[:, -self.block_size:]`。为什么要截取？

对 Bigram 来说，这无所谓——它反正只看最后一个字。但 Self-Attention 模型有**位置编码**（后面会讲），位置编码的长度是固定的。如果输入超过了 `block_size`，位置编码就不够用了。所以要截取最近的 `block_size` 个字，保证不超范围。

就好比你的工作台只能同时摊开 256 页纸——纸再多也只看最近的 256 页。

---

## Position Embedding：让模型知道"第几个字"

在讲注意力机制的代码之前，先解决一个问题：**注意力机制本身不知道字的顺序。**

为什么？因为注意力只看"Q 和 K 的匹配程度"——它关心的是"这两个字相不相关"，不关心"谁在前谁在后"。这意味着"曹操引兵"和"兵引操曹"在注意力看来是一模一样的！

这显然不行。解决办法是给每个位置加一个"座位号"：

```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

- 第 0 个位置有一个向量
- 第 1 个位置有另一个向量
- ……第 255 个位置有又一个向量

然后把"字的向量"和"位置的向量"**相加**：

```python
tok_emb = self.token_embedding_table(idx)                         # 字 → 向量
pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # 位置 → 向量
x = tok_emb + pos_emb                                             # 叠加
```

为什么用**加法**而不是**拼接**？

- 拼接会让向量变长（n_embd + n_embd），后面所有层的参数量都要翻倍
- 加法保持维度不变，而且实验表明效果一样好
- 原始 Transformer 论文就是用的加法

加完之后，同一个"操"字出现在第 2 个位置和第 100 个位置时，向量就不同了——模型就能区分顺序了。

---

## Self-Attention Head：核心实现

现在进入本篇的重头戏——自注意力头的代码实现。

### 整体结构

```python
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)  # Q 投影
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # K 投影
        self.value = nn.Linear(n_embd, head_size, bias=False)  # V 投影
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
```

三个 `nn.Linear` 就是三个**可学习的矩阵**，分别负责从每个字的原始向量中"提炼"出 Q、K、V 三种不同的表示。

为什么不用偏置（`bias=False`）？这是 Transformer 的惯例，实验表明去掉偏置效果差不多，还能少点参数。

最后一行的 `register_buffer` 是 PyTorch 提供的一个工具，用来存放"不需要训练、但需要跟着模型走"的数据。这里存的是一个下三角矩阵（后面讲因果遮罩时会详细解释）。为什么不用普通的 `self.tril = ...`？因为 `register_buffer` 有三个好处：
- 模型移到 GPU 时，它会自动跟着一起移过去
- 保存模型时会自动包含在文件里
- 但不会被当作"需要训练的参数"——它是固定的，不参与梯度更新

### 第 1 步：计算 Q、K、V

```python
q = self.query(x)  # (B, T, head_size)  每个字的"提问"
k = self.key(x)    # (B, T, head_size)  每个字的"标签"
v = self.value(x)  # (B, T, head_size)  每个字的"内容"
```

每个字的原始向量（n_embd 维），分别经过三个不同的矩阵变换，得到 Q、K、V 三种表示。

**为什么要用矩阵变换，而不是直接拿原始向量来算注意力？**

想象每个字的原始向量是一张"身份证"，上面写满了各种信息。但在不同场景下，我们需要关注不同的信息：

- **Q 矩阵**提炼出"我在找什么"——比如"兵"这个字，Q 矩阵可能学会提取出"我需要找一个动作发起者"这个需求
- **K 矩阵**提炼出"我能提供什么"——比如"曹操"这个字，K 矩阵可能学会提取出"我是一个人物"这个标签
- **V 矩阵**提炼出"我的实际内容"——"曹操"经过 V 矩阵后，输出的是它要传递给其他字的语义信息

如果不用矩阵，直接拿原始向量互相算点积，就相当于"提问"和"回答"用的是同一套信息——这就像你问别人"你是做什么工作的？"，对方只能把整张身份证递给你，而不能针对性地回答。三个矩阵让模型能**学会**在不同角色下强调不同的特征，这是注意力机制强大的关键。

而且这三个矩阵是**训练出来的**——模型会通过大量数据自动学会"Q 该提炼什么需求、K 该提炼什么标签、V 该传递什么内容"。我们不需要手动设计规则。

### 第 2 步：计算注意力分数

```python
wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
```

这一行信息量很大，拆开看：

1. **`q @ k.transpose(-2, -1)`** —— Q 和 K 做点积
   - 结果是一个 `(T, T)` 的矩阵，`wei[i][j]` 表示"第 i 个字对第 j 个字的关注程度"
   - 点积越大，说明两个字越"匹配"

2. **`* (C ** -0.5)`** —— 除以 √head_size（缩放）
   - 这是"缩放点积注意力"（Scaled Dot-Product Attention）
   - 为什么要除？如果 head_size 很大（比如 64），点积值可能非常大（比如 100+），经过 softmax 后会变成"一个接近 1，其余接近 0"的极端分布
   - 极端分布的梯度几乎为零 → 模型学不动
   - 除以 √64 = 8 后，数值变温和，softmax 的输出更"均匀"，模型能正常学习
   - 这个技巧来自原始 Transformer 论文《Attention Is All You Need》

### 第 3 步：因果遮罩（Causal Mask）

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

**这是语言模型和普通 Transformer 的关键区别。**

语言模型的任务是"预测下一个字"。如果第 3 个位置能看到第 4、5、6 个位置的内容，就等于"考试偷看答案"——那模型学到的不是理解语言，而是"作弊"。

因果遮罩就是一个下三角矩阵：

```
位置  0  1  2  3  4
  0 [ 1  0  0  0  0 ]   ← 位置 0 只能看自己
  1 [ 1  1  0  0  0 ]   ← 位置 1 能看 0 和自己
  2 [ 1  1  1  0  0 ]   ← 位置 2 能看 0、1 和自己
  3 [ 1  1  1  1  0 ]   ← 位置 3 能看 0、1、2 和自己
  4 [ 1  1  1  1  1 ]   ← 位置 4 能看所有位置
```

上三角为 0 的地方被填成 `-inf`（负无穷），经过 softmax 后这些位置的权重变成 0 —— 等于"看不见"后面的字。这个下三角矩阵就是前面用 `register_buffer` 存的那个 `tril`——它是固定不变的，所以不需要训练。

### 第 4 步：softmax 转概率 + 加权求和

```python
wei = F.softmax(wei, dim=-1)  # (B, T, T)  分数变概率
out = wei @ v                  # (B, T, head_size)  按概率加权求和 V
```

softmax 把注意力分数转成 0~1 之间的概率（每行加起来等于 1），然后用这些概率对 V 做加权平均。

最终效果：每个字的新表示 = 它"关注"的那些字的 V 的加权平均。关注度高的字贡献多，关注度低的字贡献少。

---

## 完整模型：SelfAttentionLanguageModel

把上面的组件组装起来：

```python
class SelfAttentionLanguageModel(BaseLanguageModel):
    def __init__(self, vocab_size, n_embd=64, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)       # 字 → 向量
        self.position_embedding_table = nn.Embedding(block_size, n_embd)    # 位置 → 向量
        self.sa_head = Head(n_embd, n_embd, block_size)                     # 注意力头
        self.lm_head = nn.Linear(n_embd, vocab_size)                        # 向量 → 打分

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                            # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb                                               # (B, T, n_embd)
        x = self.sa_head(x)                                                 # (B, T, n_embd)
        logits = self.lm_head(x)                                            # (B, T, vocab_size)
        # loss 计算和 Bigram 完全一样
        ...
```

### 数据流对比：Bigram vs Self-Attention

```
Bigram:
  字 → Embedding(vocab_size, vocab_size) → logits
  每个字独立查表，互不相干

Self-Attention:
  字 → Embedding(vocab_size, 64) → +位置编码 → 注意力(融合上下文) → Linear(64, vocab_size) → logits
  每个字先"开会"交流，再做预测
```

### 参数量对比

| 模型 | 核心组件 | 参数量 |
|------|---------|--------|
| Bigram | Embedding(4742, 4742) | ~2250 万 |
| Self-Attention | Embedding(4742, 64) + Position(256, 64) + Q/K/V(64, 64)×3 + Linear(64, 4742) | ~63 万 |

有趣的事实：Self-Attention 模型的参数量只有 Bigram 的 **3%**，但因为有了注意力机制来融合上下文，效果反而更好。这说明**架构设计比单纯堆参数重要得多**。

---

## 超参数的变化

从 Bigram 升级到 Self-Attention，超参数也需要相应调整：

| 超参数 | Bigram | Self-Attention | 为什么变了？ |
|--------|--------|----------------|-------------|
| BATCH_SIZE | 64 | 32 | 模型更复杂，一批太多会内存不够 |
| BLOCK_SIZE | 8 | 256 | 注意力模型能利用更长的上下文 |
| LEARNING_RATE | 0.01 | 0.001 | 模型更复杂，步子太大容易"走偏" |
| N_EMBD | — | 64 | 每个字用 64 维向量表示 |
| MAX_STEPS | 10000 | 10000 | 暂时保持不变 |

### 为什么 BLOCK_SIZE 从 8 变成 256？

Bigram 只看前 1 个字，BLOCK_SIZE=8 只是为了"一条样本多产几个练习"。但 Self-Attention 模型能真正利用上下文——BLOCK_SIZE 越大，每个字能"看到"越多的前文。256 意味着模型能同时参考前面 256 个字来预测下一个字。

### 为什么学习率从 0.01 降到 0.001？

Self-Attention 模型比 Bigram 复杂得多（多了 Q/K/V 投影、位置编码等）。复杂模型的"loss 地形"更崎岖，学习率太大就像在山路上开快车——容易翻车。降低学习率就是"减速"，让优化过程更稳定。

---

## 训练与生成

### 训练命令

```bash
# 训练 Self-Attention 模型
uv run python train.py --model-type attention

# 生成文本
uv run python generate.py --model attention_model.pt --prompt "却说曹操"
```

### 训练脚本的变化

`train.py` 现在通过 `--model-type` 参数选择模型。内部使用 `MODEL_REGISTRY` 字典来找到对应的模型类：

```python
from model import MODEL_REGISTRY

ModelClass = MODEL_REGISTRY[model_type]  # "attention" → SelfAttentionLanguageModel
model = ModelClass(vocab_size, n_embd=N_EMBD, block_size=BLOCK_SIZE)
```

保存 checkpoint 时额外存了 `model_type` 和 `config`，这样 `generate.py` 加载时能自动识别是哪种模型、用什么参数来重建。

---

## 为什么只用一个注意力头？

你可能在其他教程里看到"多头注意力"（Multi-Head Attention）。我们这里只用了**一个头**，原因是：

1. **教学优先**：一个头就足以展示注意力机制的核心原理
2. **对比清晰**：单头 vs Bigram 的对比更纯粹，不会被其他因素干扰
3. **下一篇展开**：Multi-Head Attention 会在第三篇详细讲解

一个头就像只有一个人在"关注"上下文，多个头就像一个委员会——每个人关注不同方面（有的关注语法、有的关注语义、有的关注距离），最后汇总大家的意见。

---

## 总结

| 你学到了什么 | 一句话回顾 |
|-------------|-----------|
| **BaseLanguageModel** | 统一接口，让不同模型用同一套训练和生成流程 |
| **Position Embedding** | 给每个位置编号，让模型知道字的顺序 |
| **Q / K / V** | 提问、标签、内容 —— Self-Attention 的三个核心角色 |
| **缩放点积** | 除以 √d 防止数值过大，让 softmax 正常工作 |
| **因果遮罩** | 下三角矩阵，防止模型偷看未来的字 |
| **超参数调整** | 模型变复杂后要降低学习率、增大上下文长度 |

从 Bigram 到 Self-Attention，模型从"只看一个字的瞎子"变成了"能看到前面所有字的人"。参数量减少了 97%，但理解能力大幅提升。这就是注意力机制的威力——**不是靠记忆力（参数量），而是靠理解力（架构设计）**。

下一篇，我们会给模型加上 **Multi-Head Attention（多头注意力）+ Feed-Forward Network（前馈网络）**，构成完整的 Transformer Block。到时候模型不再只有"一个人"在关注上下文，而是一整个"委员会"，生成质量会进一步提升。敬请期待！

---

*代码仓库：[GitHub 链接]*
*使用的技术栈：Python 3.10 + PyTorch + uv*
*训练数据：《三国演义》全文（~60 万字符）*
