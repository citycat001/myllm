# 从零手搓大语言模型（四）：Mini-GPT —— 从散装积木到套装升级

> 这是「从零手搓 LLM」系列的第四篇。上一篇我们把注意力和前馈网络做成了可自由组装的积木块，但只堆了 1 层。今天我们把积木"叠高"成 6 层，加入 **Dropout 正则化**，组装出一个真正的 **Mini-GPT** —— 和 GPT-2 架构相同，只是规模小一些。
>
> 关于 LLM 的整体架构和六个核心概念，请参阅本系列第一篇。

## 本篇在学习计划中的位置

| 步骤 | 内容 | 状态 |
|------|------|:----:|
| 第 1 步 | Bigram 模型 | ✅ 已完成 |
| 第 2 步 | Self-Attention | ✅ 已完成 |
| 第 3 步 | Multi-Head Attention + FFN | ✅ 已完成 |
| **第 4 步 👈 本篇** | **Mini-GPT** | **进行中** |
| 第 5 步 | BPE 分词器 | 待做 |
| 第 6 步 | 自定义数据微调 | 待做 |

**本篇新增的核心概念：Block（通用积木 = 固定流程 + 可插拔算法）、多层堆叠（n_layer）、Dropout（正则化）**

---

## 上一篇留了什么坑？

上一篇我们造了两种积木：
- **AttentionBlock**：LayerNorm + 多头注意力 + 残差连接
- **FFNBlock**：LayerNorm + 前馈网络 + 残差连接

用它们组装出了 `multihead_ffn` 模型——4 头注意力 + FFN，但只有 **1 层**。

```
当前模型（1 层）：
  嵌入 → [AttentionBlock] → [FFNBlock] → 输出

真正的 GPT（N 层）：
  嵌入 → [Transformer层1] → [Transformer层2] → ... → [Transformer层N] → 输出
```

差距在哪？**层数**。真正的 GPT-2 有 12 层，GPT-3 有 96 层。每多一层，模型就能理解更深层次的语义。

---

## 为什么要堆叠多层？一轮会议不够用

回忆上一篇的比喻：注意力 = 开会讨论，FFN = 会后独立思考。

1 层模型 = 只开 1 轮会议。参会者（每个字）互相交流一次，然后各自思考一次，就得出结论了。

问题是：**复杂问题一轮会议讨论不清楚。**

看这个例子：

```
"却说曹操引兵追赶关公到城下叫门"
```

要理解这句话，需要逐层递进的理解：

```
第 1 层：识别基本语法
  - "曹操" 是一个人名（两个字组成一个实体）
  - "引" 是动词，"兵" 是宾语

第 2 层：理解动作关系
  - "曹操引兵" → 曹操带着士兵
  - "追赶关公" → 追赶的目标是关公

第 3 层：把握事件脉络
  - "引兵追赶" → 军事追击行动
  - "到城下" → 追到了城池旁边

第 4 层：理解整体语境
  - "叫门" → 到了城下后要求开门
  - 结合全句 → 这是一个军事追击后围城的场景

第 5-6 层：预测下一步
  - 综合所有信息 → 下一句可能是守城方的反应
```

每一层建立在上一层的理解之上。第 1 层识别出"曹操"是人名后，第 2 层才能理解"曹操引兵"是主谓宾关系；第 2 层理解了"追赶关公"后，第 3 层才能把握整个追击事件。

**这就是为什么要叠多层——每一层做一次"开会+思考"，逐层加深理解。**

---

## 从散装积木到套装积木

### 散装的问题

上一篇的 `AttentionBlock` 和 `FFNBlock` 是**散装积木**——各自独立，自由搭配。这对学习和实验很有用，但真正构建多层模型时，有个问题：散装积木没有 Dropout。

什么是 Dropout？一句话：**训练时随机"关掉"一部分神经元，防止模型死记硬背。**

### Dropout：考试时随机遮住笔记

想象一个学生在备考。如果每次做题都可以看全部笔记，他可能会偷懒——不去真正理解知识，而是死记"第 3 页第 5 行是答案"。这就是**过拟合**：模型把训练数据背下来了，遇到新数据就傻眼。

Dropout 的做法就像：**每次做题时，随机遮住 20% 的笔记**。

```
完整的笔记：[知识A, 知识B, 知识C, 知识D, 知识E]

第 1 次做题：[知识A, ████, 知识C, ████, 知识E]  ← B和D被遮住
第 2 次做题：[████, 知识B, ████, 知识D, 知识E]  ← A和C被遮住
第 3 次做题：[知识A, 知识B, 知识C, ████, ████]  ← D和E被遮住
```

每次被遮住的部分不同，学生被迫**真正理解每个知识点**，而不是依赖某几个固定的"锚点"。这样训练出来的模型泛化能力更强——遇到没见过的文本也能应对。

**关键：Dropout 只在训练时生效。** 考试时（推理/生成），所有笔记都可以看——`model.eval()` 会自动关闭 Dropout。

### 代码中的 Dropout

在 PyTorch 中，Dropout 就是一行代码：

```python
self.dropout = nn.Dropout(0.2)  # 随机丢弃 20% 的值

# 训练时：随机把一些值变成 0
# 推理时：什么都不做（nn.Dropout 自动检测 train/eval 模式）
```

### Dropout 放在哪里？

在标准的 GPT-2 架构中，Dropout 分布在三个地方：

```
注意力头内部：
  Q·K → softmax → [Dropout] → 加权求和V     ← 随机忽略一些字的注意力

多头注意力的投影后：
  拼接 → 投影层 → [Dropout]                  ← 随机丢弃一些合并后的信息

前馈网络的输出：
  展开 → ReLU → 压缩 → [Dropout]             ← 随机丢弃一些思考结论
```

每个 Dropout 都在做同一件事：**随机丢弃一些信息，迫使模型不要过度依赖任何单一的特征或连接。**

---

## 积木重构：固定流程 + 可插拔算法

理解了多层堆叠和 Dropout 的必要性后，我们来重新设计积木架构。

### 上一篇的问题

上一篇的 `AttentionBlock` 和 `FFNBlock` 各自把 LayerNorm、核心算法、残差连接**硬编码**在一起。看起来是模块化的，但仔细一想：

```
AttentionBlock 内部：LayerNorm → MultiHeadAttention → 残差    ← 三者焊死在一起
FFNBlock 内部：      LayerNorm → FeedForward → 残差           ← 三者焊死在一起
```

**每种积木都把流程和算法绑定了。** 如果将来我们想换一种注意力算法（比如 GroupedQueryAttention），就得新建一个 Block 类。

### 新设计：积木 = 固定流程 + 可插拔算法

游戏里的武器是怎么设计的？**框架固定，模块可换。** 同一个武器框架，装上不同的核心模块，就变成不同的武器。

我们的积木也应该这样：

```
Block（通用积木框架）
├── 固定流程：LayerNorm → [算法组件] → 残差连接    ← 框架不变
└── 可插拔算法（op）：                              ← 核心模块可换
    ├── MultiHeadAttention  → 变成注意力积木
    ├── FeedForward         → 变成前馈网络积木
    └── 未来可扩展...       → 变成任何新积木
```

### 代码实现

通用积木只有 10 行代码：

```python
class Block(nn.Module):
    """通用积木 —— 固定流程 + 可插拔算法组件。"""

    def __init__(self, n_embd, op):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.op = op  # 可插拔的算法组件

    def forward(self, x):
        return x + self.op(self.ln(x))  # LayerNorm → 算法 → 残差
```

`op` 可以是任何输入输出都是 `(B, T, n_embd)` 的模块。换不同的 `op`，积木的功能就完全不同：

```python
# 注意力积木（等价于上一篇的 AttentionBlock）
Block(n_embd=384, op=MultiHeadAttention(384, 6, 64, 256, dropout=0.2))

# 前馈网络积木（等价于上一篇的 FFNBlock）
Block(n_embd=384, op=FeedForward(384, dropout=0.2))
```

### 算法组件工厂：build_op

手动创建 op 太繁琐，我们提供一个工厂函数：

```python
def build_op(name, n_embd, n_head, block_size, dropout=0.0):
    """根据名称创建可插拔的算法组件。"""
    if name == "attention":
        head_size = n_embd // n_head
        return MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
    elif name == "ffn":
        return FeedForward(n_embd, dropout)
```

**Dropout 在哪？** 在算法组件内部，不在积木流程里：
- `MultiHeadAttention`：softmax 后的注意力权重 Dropout + 投影后 Dropout
- `FeedForward`：第二层线性变换后的 Dropout

这遵循了 GPT-2 的标准做法：**Dropout 在算法组件内部，不在残差路径上重复。**

### 给 Head、FeedForward、MultiHeadAttention 加 Dropout 支持

为了让 TransformerBlock 能传递 dropout 参数，我们给三个底层组件加了可选的 `dropout` 参数：

```python
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):  # ← 新增
        # ...
        self.attn_dropout = nn.Dropout(dropout)  # softmax 后的 Dropout

    def forward(self, x):
        # ...
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)  # ← 新增：随机丢弃一些注意力连接
        out = wei @ v
        return out
```

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):  # ← 新增
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),  # ← 新增：输出 Dropout
        )
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, block_size, dropout=0.0):  # ← 新增
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)  # ← 传递给每个 Head
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_head * head_size, n_embd, bias=False)
        self.proj_dropout = nn.Dropout(dropout)  # ← 新增：投影后 Dropout
```

**向后兼容**：所有 `dropout` 参数默认值都是 `0.0`。`nn.Dropout(0.0)` 不丢弃任何值，等于没有 Dropout。所以旧代码（散装积木）不传 dropout 参数，行为完全不变。

---

## n_layer：一键堆叠多层

有了通用 Block + build_op，堆叠多层就变得很简单。`build_blocks()` 工厂函数用 `n_layer` 控制重复次数：

```python
def build_blocks(block_names, n_embd, n_head, block_size,
                 n_layer=1, dropout=0.0):
    blocks = []
    for _ in range(n_layer):          # ← 重复 n_layer 次
        for name in block_names:
            op = build_op(name, n_embd, n_head, block_size, dropout)
            blocks.append(Block(n_embd, op))
    return blocks
```

`n_layer=6` + `block_names=["attention", "ffn"]` → 创建 12 个 Block（6 层 × 每层 2 个积木）。

```
第 1 层：Block(attention) → Block(ffn)    ← 开会 + 思考
第 2 层：Block(attention) → Block(ffn)    ← 开会 + 思考
 ...
第 6 层：Block(attention) → Block(ffn)    ← 开会 + 思考
```

**每层的结构相同，但权重不同。** 就像同一个会议流程开 6 轮，每轮的参会者讨论的内容和结论都不一样——因为每层有自己独立的 Q/K/V 矩阵、FFN 权重等参数，各自通过训练学到不同的东西。

---

## Mini-GPT 的完整配置

在 `train.py` 中，Mini-GPT 的配置是这样的：

```python
"mini_gpt": {
    "batch_size": 64,
    "block_size": 256,
    "max_steps": 5000,
    "lr": 3e-4,             # 更大的模型需要更小的学习率
    "n_embd": 384,          # 嵌入维度（384 / 6 头 = 每头 64 维）
    "n_head": 6,            # 6 头注意力
    "n_layer": 6,           # 6 层 Transformer
    "dropout": 0.2,         # 随机丢弃 20% 的连接
    "embedding_type": "token_position",
    "block_names": ["attention", "ffn"],  # 每层 = 注意力积木 + FFN 积木
},
```

### 超参数选择的讲解

**为什么 `n_embd=384`？** 上一篇用的是 64 维，但 6 层模型需要更大的"思考空间"。384 维能存储更丰富的语义信息。同时 384 能被 6 整除（每头 64 维），这是一个合理的搭配。

**为什么 `n_head=6`？** 每头 64 维（384/6=64），和上一篇的单头模型维度一致。6 个头意味着能同时关注 6 种不同的语义关系。

**为什么 `n_layer=6`？** 这是 nanoGPT 教程推荐的配置，对于字符级语言模型来说是一个好的起点。层数太少表达能力不够，太多训练太慢。

**为什么 `lr=3e-4`？** 更大的模型需要更小的学习率。上一篇用 1e-3，这里用 3e-4——约小了 3 倍。这是 Adam 优化器搭配中等规模模型的经典选择。

**为什么 `dropout=0.2`？** 即每次随机丢弃 20% 的连接。这是中等规模模型的常用值。太小（如 0.05）正则化效果不明显；太大（如 0.5）会导致训练信号太弱。

### 与之前模型的对比

| 模型 | n_embd | n_head | 层数 | Dropout | 参数量 |
|------|--------|--------|------|---------|--------|
| bigram | — | — | 0 | 无 | ~22.5M（大但只是查找表） |
| attention | 64 | 1 | 1 | 无 | ~316K |
| multihead_ffn | 64 | 4 | 1 | 无 | ~330K |
| **mini_gpt** | **384** | **6** | **6** | **0.2** | **~14.4M** |

Mini-GPT 的参数量是 multihead_ffn 的约 43 倍，但这些参数分布在 6 层 Transformer 中，每一层都在做有意义的信息处理——而不像 Bigram 那样只是一张死记硬背的查找表。

---

## 数据流全景：6 层 Transformer

让我们追踪一段文字在 Mini-GPT 中的完整旅程：

```
"曹操引兵追赶" → [1038, 2893, 2436, 1204, 3412, 4101]

→ Token Embedding + Position Embedding    # 字义 + 位置 → 384维向量

→ 第 1 层：Block(attention) + Block(ffn)
  ├── 6头注意力：识别基本的词组关系（"曹操" 是人名、"引兵" 是动宾）
  ├── FFN：提炼词级别的语义特征
  └── Dropout：随机丢弃 20% 的连接

→ 第 2 层：Block(attention) + Block(ffn)
  ├── 6头注意力：理解句法结构（主语-谓语-宾语）
  ├── FFN：深化句法理解
  └── Dropout

→ 第 3 层：Block(attention) + Block(ffn)
  ├── 6头注意力：建立事件关系（追赶是谁发起的、对象是谁）
  ├── FFN：综合事件信息
  └── Dropout

→ 第 4-6 层：Block(attention) + Block(ffn)
  ├── 逐层加深：从词组 → 句法 → 事件 → 语境 → 预测
  └── 每层都在上一层的理解基础上进一步抽象

→ Final LayerNorm → Linear → Softmax
  → 预测下一个字："关"（后面是"关公"）
```

注意：上面对每层功能的描述是示意性的。实际训练中，模型会自己学出每层该关注什么——我们不需要（也没法）手动指定。

---

## 如何训练和使用

### 训练

```bash
uv run python train.py --model-type mini_gpt
```

训练完成后模型保存为 `mini_gpt_model.pt`。

### 生成文本

```bash
uv run python generate.py --model mini_gpt_model.pt --prompt "却说曹操" --length 200
```

### 训练结果

> **注意**：Mini-GPT 有 1440 万参数，在 CPU 上训练较慢（每步约 10 秒）。建议使用 GPU 训练。训练结果将在后续补充。

<!-- TODO: 训练完成后补充 loss 曲线和生成样本 -->

---

## 总结

| 你学到了什么 | 一句话回顾 |
|-------------|-----------|
| **Block（通用积木）** | 固定流程（LN → op → 残差）+ 可插拔算法组件 |
| **build_op（算法工厂）** | 根据名称创建不同的算法组件，插入积木中 |
| **多层堆叠（n_layer）** | 重复 N 次 [attention, ffn]，逐层加深理解 |
| **Dropout** | 训练时随机丢弃部分连接，防止过拟合（推理时自动关闭） |
| **超参数升级** | n_embd 64→384, n_head 4→6, 加入 n_layer 和 dropout |

从 1 层积木到 6 层 Transformer，模型的架构已经和真正的 GPT-2 完全一致——只是规模小一些。这就是 **Mini-GPT**。

回顾一下我们走过的路：

```
第 1 步：Bigram      → 只看前 1 个字（查表）
第 2 步：Attention   → 能看前 256 个字（1个注意力头）
第 3 步：MultiHead   → 多角度看上下文（4个头 + FFN，1层）
第 4 步：Mini-GPT    → 逐层加深理解（6个头 + FFN + Dropout，6层）← 你在这里
```

架构已经到位，接下来的提升空间在**数据处理**上。下一篇，我们会实现 **BPE 分词器**——从"一个字 = 一个 token"升级为更智能的子词分词，让模型能更高效地理解语言。敬请期待！

---

*代码仓库：[GitHub](https://github.com/citycat001/myllm)*
*使用的技术栈：Python 3.13 + PyTorch + uv*
*训练数据：《三国演义》全文（~60 万字符）*
