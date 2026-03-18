# 从零手搓大语言模型（三）：多头注意力 + 前馈网络 —— 积木式 Transformer

> 这是「从零手搓 LLM」系列的第三篇。上一篇我们用单头自注意力让模型有了"回头看"的能力，但只有一个视角难免片面。今天我们加入**多头注意力（Multi-Head Attention）**和**前馈网络（Feed-Forward Network）**，并把它们设计成可自由组装的**积木块（Block）**，为搭建完整的 Transformer 做好准备。
>
> 关于 LLM 的整体架构和六个核心概念，请参阅本系列第一篇。

## 本篇在学习计划中的位置

| 步骤 | 内容 | 状态 |
|------|------|:----:|
| 第 1 步 | Bigram 模型 | ✅ 已完成 |
| 第 2 步 | Self-Attention | ✅ 已完成 |
| **第 3 步 👈 本篇** | **Multi-Head Attention + FFN** | **进行中** |
| 第 4 步 | Mini-GPT | 待做 |
| 第 5 步 | BPE 分词器 | 待做 |
| 第 6 步 | 自定义数据微调 | 待做 |

**本篇新增的核心概念：Multi-Head Attention（多头注意力）、Feed-Forward Network（前馈网络）、残差连接（Residual Connection）、层归一化（Layer Normalization）、Block 组装架构**

---

## 单头注意力的局限：一个人开会能讨论出什么？

上一篇的自注意力模型只有**一个注意力头**。回顾一下它的工作方式：每个字通过 Q/K/V 三个矩阵，从上下文中"收集"信息。

问题在于：**一个头只能学到一种"关注模式"。** 就像一场会议只有一个议题——如果在讨论"谁是主语"，就没法同时讨论"动作的对象是什么"。

看一个例子：

```
"曹操引兵追赶关公"
```

理解这句话需要同时关注多个维度：
- **语法关系**："曹操"是主语，"引"是动词
- **动宾搭配**："引"的对象是"兵"，"追赶"的对象是"关公"
- **远距离关联**："曹操"和"追赶"隔了好几个字，但语义上紧密相关

一个注意力头很难同时兼顾这些不同维度的关系。

**解决方案：开多场平行会议！** 每场会议讨论不同的议题，最后把各场会议的结论合并。这就是**多头注意力**。

---

## 多头注意力：从"一人独断"到"委员会决策"

### 核心思想

多头注意力的做法非常直觉：

1. 把嵌入维度**均分**给多个头（比如 64 维分给 4 个头，每头 16 维）
2. 每个头**独立**做一次完整的自注意力（各自有 Q/K/V 矩阵）
3. 把所有头的输出**拼接**起来
4. 通过一个**投影层**合并回原来的维度

```
输入 x (64维)
    ├── 头1: Q1/K1/V1 → 输出 (16维)   关注"谁是主语"
    ├── 头2: Q2/K2/V2 → 输出 (16维)   关注"动宾关系"
    ├── 头3: Q3/K3/V3 → 输出 (16维)   关注"事件链"
    └── 头4: Q4/K4/V4 → 输出 (16维)   关注"远距离关联"
         │
         └── 拼接 [16+16+16+16 = 64维] → 投影层 → 输出 (64维)
```

每个头的"思考空间"变小了（64 → 16 维），但多个头合在一起覆盖了更多角度。就像四个专家组成的委员会，虽然每个人看到的"切面"更窄，但大家的视角互补，整体判断比一个人更全面。

### 代码实现

多头注意力**直接复用**上一篇的 `Head` 类，创建多个实例：

```python
class MultiHeadAttention(nn.Module):
    """多头注意力 —— 同时从多个角度"开会"。"""

    def __init__(self, n_embd, n_head, head_size, block_size):
        super().__init__()
        # 创建 n_head 个独立的注意力头
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size) for _ in range(n_head)
        ])
        # 投影层：拼接后映射回 n_embd 维
        self.proj = nn.Linear(n_head * head_size, n_embd, bias=False)

    def forward(self, x):
        # 每个头独立处理，然后拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
```

**为什么需要投影层？** 拼接只是把信息"摆在一起"，投影层让模型学会"如何最好地合并各个头的结论"——哪些头的意见更重要，哪些可以弱化。

**参数量对比**：单头 `Head(64, 64)` 有 3 个 64×64 的矩阵 = 12,288 参数。4 头 `MultiHeadAttention(64, 4, 16)` 有 4 × 3 × (64×16) = 12,288 参数 + 投影层 64×64 = 4,096 参数。总参数稍多一点，但表达能力提升很大。

---

## 前馈网络：开完会还得自己想想

注意力层做的是**信息收集**——每个字看看上下文，把相关信息汇总过来。但汇总只是加权平均，属于**线性操作**，表达能力有限。

类比：你参加了一场圆桌会议，听取了所有人的发言。但光"听"还不够，你得回去**自己消化思考**，才能形成深入的理解。

这就是**前馈网络（Feed-Forward Network, FFN）**的作用：对每个字的向量独立做一次非线性变换。

### 结构："展开→激活→压缩"

```python
class FeedForward(nn.Module):
    """前馈网络 —— 会后每个人独立思考总结。"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 展开：64 → 256 维
            nn.ReLU(),                        # 非线性激活
            nn.Linear(4 * n_embd, n_embd),   # 压缩：256 → 64 维
        )

    def forward(self, x):
        return self.net(x)
```

**为什么先展开再压缩？** 就像写笔记——先在草稿纸上展开思路（256 维的空间有更多"余地"去发现规律），然后把要点浓缩成简洁的结论（回到 64 维）。4 倍的展开比例是 Transformer 原始论文的经验值。

**为什么需要 ReLU？** 如果只有线性层（矩阵乘法），不管叠多少层，数学上等价于一层。ReLU 引入了非线性——负数变 0，正数保持不变——这让模型能学到"有就有，没有就没有"的硬判断，大大增强了表达能力。

**FFN 的独立性**：注意 FFN 对每个字**独立**处理（不看其他字），它的输入和输出都是 `(B, T, n_embd)`。这意味着它完全不依赖注意力层，可以和**任何注意力配置**搭配使用。

---

## 两个关键基础设施：残差连接 + LayerNorm

在叠加注意力和 FFN 之后，我们面临一个实际问题：**训练不稳定**。数据经过多层变换后，数值可能爆炸或消失，梯度传不回去，模型学不动。

解决这个问题需要两个"基础设施"：

### 残差连接（Residual Connection）

核心思想极其简单：**把输入直接加到输出上**。

```python
x = x + attention(x)   # 而不是 x = attention(x)
```

就像开会讨论后，你不会完全忘掉自己原来的想法，而是把讨论得到的**新信息叠加**到原有认知上。

技术上的好处：梯度可以通过"+"号直接传回去（加法的梯度是 1），不管中间的变换多复杂，梯度都不会消失。这是深层网络能训练起来的关键。

### LayerNorm（层归一化）

每层的输入数值范围可能差别很大。LayerNorm 把每个样本的向量"拉齐"到均值 0、标准差 1 的范围：

```python
x = LayerNorm(x)  # 归一化：让数值分布稳定
```

就像考试前先统一度量衡——不管之前的数值是大是小，归一化后都在统一的范围内，后续计算更稳定。

### Pre-Norm 架构

我们采用 **Pre-Norm**（先归一化再变换），这是 GPT-2 以后的主流做法：

```python
x = x + attention(LayerNorm(x))   # Pre-Norm：先归一化，再注意力，最后加残差
x = x + ffn(LayerNorm(x))         # 同样的模式
```

比原始 Transformer 的 Post-Norm（先变换再归一化）更稳定，训练更快收敛。

---

## 积木式架构：像搭乐高一样组装模型

到目前为止我们有了四个组件：单头注意力（Head）、多头注意力（MultiHeadAttention）、前馈网络（FeedForward）、残差+LayerNorm。

关键设计决策：**不把这些组件固定焊死在一起，而是做成可插拔的积木块（Block）**。

### 为什么要做成积木？

在真实的 Transformer 中，不同的变体会用不同的组件组合：
- GPT-2：多头注意力 + FFN（标准组合）
- 有些实验：只用注意力不加 FFN
- 有些实验：用单头注意力 + FFN

我们把每个组件包装成统一接口的 Block：输入 `(B, T, n_embd)` → 输出 `(B, T, n_embd)`。只要接口一致，就能像乐高积木一样自由拼装。

### AttentionBlock：注意力积木

```python
class AttentionBlock(nn.Module):
    """注意力插件 —— LayerNorm + 注意力 + 残差连接。"""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        if n_head == 1:
            self.attn = Head(n_embd, n_embd, block_size)     # 单头
        else:
            head_size = n_embd // n_head
            self.attn = MultiHeadAttention(n_embd, n_head, head_size, block_size)  # 多头

    def forward(self, x):
        return x + self.attn(self.ln(x))   # 残差 + LayerNorm + 注意力
```

`n_head=1` 自动用单头，`n_head=4` 自动用多头。对外接口完全一样。

### FFNBlock：前馈网络积木

```python
class FFNBlock(nn.Module):
    """前馈网络插件 —— LayerNorm + FFN + 残差连接。"""

    def __init__(self, n_embd):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)

    def forward(self, x):
        return x + self.ffn(self.ln(x))    # 残差 + LayerNorm + FFN
```

FFN 积木完全独立于注意力——它不关心输入是来自单头还是多头，甚至可以单独使用。

### AssembledModel：积木组装台

有了积木块，我们需要一个"组装台"把它们串起来：

```python
class AssembledModel(BaseLanguageModel):
    """积木式模型 —— 通过 Block 列表自由组装。"""

    def __init__(self, vocab_size, n_embd, block_size, blocks):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(blocks)    # 积木列表
        self.ln_final = nn.LayerNorm(n_embd)   # 最终归一化
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:    # 依次通过每个积木
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        # ... 计算 loss
```

`AssembledModel` 自己不做任何"思考"，它只负责搭台子（embedding + 位置编码 + 输出层），中间的所有处理都交给传入的 Block 列表。

### 组装示例

```python
# 配置 1：单头注意力 + FFN
blocks = [AttentionBlock(64, n_head=1, block_size=256), FFNBlock(64)]

# 配置 2：4头注意力（无 FFN）
blocks = [AttentionBlock(64, n_head=4, block_size=256)]

# 配置 3：4头注意力 + FFN（标准 Transformer Block）
blocks = [AttentionBlock(64, n_head=4, block_size=256), FFNBlock(64)]
```

只要改变 Block 列表，就能得到完全不同的模型！

### build_blocks：积木的组装说明书

为了让 `train.py` 和 `generate.py` 都能方便地创建 Block 列表，我们提供了一个工厂函数：

```python
def build_blocks(block_names, n_embd, n_head, block_size):
    """根据名称列表创建 Block 实例。"""
    blocks = []
    for name in block_names:
        if name == "attention":
            blocks.append(AttentionBlock(n_embd, n_head, block_size))
        elif name == "ffn":
            blocks.append(FFNBlock(n_embd))
    return blocks
```

在 `train.py` 中，不同的模型配置用字典定义：

```python
MODEL_CONFIGS = {
    "attention_ffn": {
        "n_embd": 64, "n_head": 1,
        "block_names": ["attention", "ffn"],   # 单头 + FFN
    },
    "multihead": {
        "n_embd": 64, "n_head": 4,
        "block_names": ["attention"],           # 纯多头
    },
    "multihead_ffn": {
        "n_embd": 64, "n_head": 4,
        "block_names": ["attention", "ffn"],    # 多头 + FFN
    },
}
```

想要新的组合？只需在字典里加一行配置，不用写任何新的模型类。

---

## 数据流全景

让我们看看 `multihead_ffn` 配置下，一段文字经过模型的完整旅程：

```
"曹操引兵" → [1038, 2893, 2436, 1204]        # 字符级分词

→ Token Embedding + Position Embedding         # 字义 + 位置 → 64维向量
  [1038] → [0.12, -0.5, ..., 0.3] (64维)

→ AttentionBlock                               # 积木1：多头注意力
  ├── LayerNorm：归一化输入
  ├── 4个头并行做注意力：
  │   头1 关注 [曹→操] 语法关系
  │   头2 关注 [引→兵] 动宾搭配
  │   头3 关注 [曹操→引兵] 事件链
  │   头4 关注整体句式
  ├── 拼接 + 投影：合并4个头的结论
  └── 残差连接：叠加原始输入

→ FFNBlock                                     # 积木2：前馈网络
  ├── LayerNorm：归一化
  ├── 展开 64→256 → ReLU → 压缩 256→64
  └── 残差连接：叠加输入

→ Final LayerNorm → Linear → Softmax           # 输出层
  → 预测下一个字："追"
```

---

## 如何训练和使用

### 训练不同配置

```bash
# 单头注意力 + FFN（和之前的 attention 对比，看 FFN 的效果）
uv run python train.py --model-type attention_ffn

# 4头注意力（和 attention 对比，看多头的效果）
uv run python train.py --model-type multihead

# 4头注意力 + FFN（完整 Transformer Block 的雏形）
uv run python train.py --model-type multihead_ffn
```

### 生成文本

```bash
uv run python generate.py --model multihead_ffn_model.pt --prompt "却说曹操" --length 200
```

### 超参数

| 超参数 | attention（上一篇） | 新模型（本篇） | 原因 |
|--------|---------------------|----------------|------|
| BATCH_SIZE | 32 | 32 | 不变 |
| BLOCK_SIZE | 256 | 256 | 不变 |
| LEARNING_RATE | 0.001 | 0.001 | 不变（LayerNorm + 残差让训练稳定了） |
| N_EMBD | 64 | 64 | 不变 |
| N_HEAD | — | 4（多头配置） | 4 个头，每头 16 维 |

---

## 总结

| 你学到了什么 | 一句话回顾 |
|-------------|-----------|
| **MultiHeadAttention** | 多个头从不同角度做注意力，拼接后投影回原维度 |
| **FeedForward** | 两层 MLP（展开→ReLU→压缩），对每个字独立做非线性变换 |
| **残差连接** | `x + f(x)` —— 保留原始信息，解决梯度消失 |
| **LayerNorm** | 归一化数值范围，稳定训练 |
| **AttentionBlock / FFNBlock** | 统一接口的积木块，可自由组装 |
| **AssembledModel** | 积木组装台，通过 Block 列表定义模型结构 |
| **build_blocks** | 工厂函数，根据配置名称创建积木列表 |

从单头到多头，从固定结构到积木组装，模型的灵活性和表达能力都上了一个台阶。注意力负责"收集信息"，FFN 负责"消化信息"，残差和 LayerNorm 负责"稳定后勤"——这三者的组合就是 Transformer 的核心骨架。

下一篇，我们会把多层 Block 堆叠起来，构成完整的 **Mini-GPT**——真正的多层 Transformer 语言模型。到时候你会看到，只需要把积木"叠高"，模型的生成质量就能显著提升。敬请期待！

---

*代码仓库：[GitHub 链接]*
*使用的技术栈：Python 3.10 + PyTorch + uv*
*训练数据：《三国演义》全文（~60 万字符）*
