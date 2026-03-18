"""
语言模型家族 —— 从 Bigram 到可组装的 Transformer。

这个文件包含了所有模型的实现：
  - BaseLanguageModel: 所有模型的基类，定义了统一接口
  - BigramLanguageModel: 最简单的语言模型，只看前一个字猜下一个字
  - SelfAttentionLanguageModel: 加入自注意力机制，能看到更多上下文
  - AssembledModel: 积木式模型，通过 Block 列表自由组装

组件（可独立使用的积木块）：
  - TokenEmbedding: 纯 token 嵌入（字 → 向量）
  - TokenPositionEmbedding: token + 位置嵌入（字 + 位置 → 向量）
  - Head: 单个自注意力头
  - FeedForward: 前馈网络（两层 MLP）
  - MultiHeadAttention: 多头注意力（多个 Head 并行）
  - AttentionBlock: 注意力插件（LayerNorm + 注意力 + 残差连接）
  - FFNBlock: 前馈网络插件（LayerNorm + FFN + 残差连接）

所有模型共享同一套 forward() 和 generate() 接口，
这样 train.py 和 generate.py 不需要关心具体是哪个模型 —— 换模型就像换引擎，底盘不用动。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================== 基类 ========================

class BaseLanguageModel(nn.Module):
    """
    所有语言模型的基类。

    定义了两个核心接口：
      - forward(idx, targets) → (logits, loss)  训练和推理都用
      - generate(idx, max_new_tokens) → 生成的序列

    子类只需要实现自己的 __init__ 和 forward，generate 方法继承即可。
    """

    # 子类必须设置这个属性，表示模型能"看到"多远的上下文。
    # Bigram 只看 1 个字，所以理论上不限；Attention 模型受限于位置编码的长度。
    block_size: int

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        前向传播（子类必须实现）。

        参数：
            idx:     输入的字（已转成数字），形状 (B, T)
            targets: 正确答案（训练时用），生成时不传

        返回：
            logits: 对下一个字的预测打分，形状 (B, T, C)
            loss:   预测误差（训练时用），生成时为 None
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        逐字生成文本。

        过程就像一个人在写字：
          1. 看最近的上下文 → 预测下一个字的概率
          2. 按概率随机抽一个字
          3. 把新字接在后面
          4. 重复以上步骤

        注意：这里会把输入截断到 block_size 长度。
        对于 Bigram 模型来说截断没有影响（它只看最后一个字）；
        对于 Attention 模型来说，这保证了不会超出位置编码的范围。
        """
        for _ in range(max_new_tokens):
            # 截取最近的 block_size 个字作为输入
            # 好比人脑的"工作记忆"有限，只能同时关注最近的一段文字
            idx_cond = idx[:, -self.block_size:]

            # 喂给模型，拿到预测
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 只取最后一个字的预测打分

            # 把"打分"转换成"概率"
            probs = F.softmax(logits, dim=-1)

            # 按概率随机抽一个字（不是每次都选最高的，这样更有变化）
            idx_next = torch.multinomial(probs, num_samples=1)

            # 把新字拼到已有文字的后面
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ======================== Bigram 模型 ========================

class BigramLanguageModel(BaseLanguageModel):
    """
    Bigram 语言模型 —— 最简单的语言模型。

    想象一个只会看前一个字就猜下一个字的人：
    看到"曹"就猜"操"，看到"操"就猜"大"……

    核心思路：
        1. 事先准备一张大表，表里记录了"每个字后面最可能跟什么字"
        2. 输入一个字，去表里查一行，就得到了下一个字的预测
        3. 训练的过程就是不断修正这张表，让它的预测越来越准

    这张表就是所谓的 Embedding（嵌入表）。
    整个模型没有注意力机制，没有上下文理解，纯粹靠统计"谁后面跟谁"。
    """

    def __init__(self, vocab_size: int, block_size: int = 1024):
        super().__init__()
        self.block_size = block_size

        # 这张表就是整个模型的全部。
        #
        # 打个比方：假设训练文档中提取的所有词汇总共有 4742 个字，那这张表就是一个 4742 行 × 4742 列的大表格。
        # 第 i 行记录的是：看到第 i 个字之后，下一个字分别是第 0、1、2、...、4741 个字的可能性。
        #
        # nn.Embedding 做的事很简单：输入一个数字 i代表字典中的某个字，返回这个字对应的向量表中的第 i 行。
        # 它和普通数组的区别是：它能参与训练，PyTorch 会自动帮我们调整表里的数值。
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        前向传播：拿到输入的字，去查表，得到对下一个字的预测。

        参数：
            idx:     输入的字（已转成数字），形状 (B, T)
                     B = 一批处理多少条字串，T = 每条字串多少个字
                     训练时 B=64（一批 64 条样本字串），生成时 B=1（只有一条字串）。
                     PyTorch 规定输入必须带 batch 维度，就像餐厅后厨不管是
                     64 桌同时点菜还是只有 1 桌，用的都是同一套出餐流程。
            targets: 正确答案（训练时用），生成时不传

        返回：
            logits: 模型对每个字串中每个字的预测，形状 (B, T, C)
                    C = 词表大小（4742），代表"根据当前这个字预测下一个字时，对每个候选字的打分"。
                    注意：C 不是当前字本身的数学描述，而是在 4742 个候选字上的
                    概率打分数组（还没转成概率，要经过 softmax 才是真正的概率）。
            loss:   模型猜错了多少（训练时用来改进模型，生成时为 None）
        """

        # 拿每个字的编号去查表，得到这个字后面"各个字出现的可能性打分"。
        # 比如输入"曹"（编号 1038），返回一行 4742 个数字，
        # 其中"操"对应的打分可能最高，"的"对应的打分可能很低。
        # 注意：这些打分（叫 logits）还不是概率，需要后续用 softmax 转换。
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            # 计算"模型猜得有多差" —— 用交叉熵（cross entropy）来衡量。
            #
            # 训练时，T 个位置每个都在预测下一个字，然后和正确答案逐个对比：
            #   x = [却, 说, 曹, 操, 引, 兵, 追, 赶]，从文本中截取的8个字
            #   y = [说, 曹, 操, 引, 兵, 追, 赶, 关]，根据x的每个字预测的下一个字组成的新字串
            #   "却"的预测 vs 正确答案"说" → 算一个 loss
            #   "说"的预测 vs 正确答案"曹" → 算一个 loss
            #   "曹"的预测 vs 正确答案"操" → 算一个 loss
            #   ……8 个位置各算一个 loss，最后取平均。
            # 一批 64 组 × 每组 8 个位置 = 512 个"预测 vs 正确答案"的对比。
            #
            # PyTorch 的 cross_entropy 要求数据摆成特定的形状：
            # 预测值要是 (样本数, 概率打分组数)，正确答案要是 (样本数,)。
            # 所以我们把 (B, T, C) 展平成 (B*T, C)，把 (B, T) 展平成 (B*T,)。
            # 也就是把 (64, 8, 4742) → (512, 4742)，512 个预测摊平一起算。
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


# ======================== Embedding 插件 ========================

class TokenEmbedding(nn.Module):
    """
    纯 Token 嵌入 —— 最简单的嵌入方式，只把字变成向量。

    这是 Bigram 模型使用的嵌入方式：每个字查表得到一个向量，不包含位置信息。
    适用于不需要位置感知的模型（比如 Bigram 只看前一个字，位置无所谓）。

    输入：字的编号 (B, T) → 输出：向量序列 (B, T, n_embd)
    """

    def __init__(self, vocab_size: int, n_embd: int):
        """
        参数：
            vocab_size: 词表大小（有多少个不同的字）
            n_embd:     嵌入维度（每个字用多少个数字表示）
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.token_embedding_table(idx)  # (B, T, n_embd)


class TokenPositionEmbedding(nn.Module):
    """
    Token + Position 嵌入 —— 同时编码"是什么字"和"在第几个位置"。

    在 Token Embedding 的基础上加入了位置编码。
    注意力模型必须用这种嵌入，因为注意力机制本身不知道字的顺序——
    "曹操引兵"和"兵引操曹"在纯注意力看来是一样的。
    位置编码让每个位置有独特的向量，模型就能区分顺序了。

    做法：token 向量 + position 向量（逐元素相加）。
    为什么用加法而不是拼接？因为加法更简单高效，实验表明效果一样好。

    输入：字的编号 (B, T) → 输出：融合了字义和位置的向量序列 (B, T, n_embd)
    """

    def __init__(self, vocab_size: int, n_embd: int, block_size: int):
        """
        参数：
            vocab_size: 词表大小
            n_embd:     嵌入维度
            block_size: 最大上下文长度（位置编码表的行数）
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T, n_embd)，自动广播到 (B, T, n_embd)
        return tok_emb + pos_emb  # (B, T, n_embd)


# ======================== 自注意力头 ========================

class Head(nn.Module):
    """
    单个自注意力头（Self-Attention Head）。

    自注意力是让序列中的每个字都能"看到"前面所有字的机制。
    打个比方：Bigram 就像一个人只看前一个字就猜下一个字，
    而自注意力就像开会讨论 —— 每个字都能回头看前面所有字，
    然后综合判断下一个字应该是什么。

    具体做法：每个字提出三个问题（Q、K、V）：
      - Q（Query，提问）：我在找什么样的信息？
      - K（Key，标签）：我能提供什么样的信息？
      - V（Value，内容）：我实际包含的信息是什么？

    然后用 Q 和 K 的匹配程度来决定"关注谁"，最后按照关注度加权求和 V。

    举个例子：在"却说曹操引兵"中，当模型处理"兵"这个字时：
      - "兵"的 Q 可能在问：谁发起了这个动作？
      - "曹操"的 K 回应：我是一个人物名
      - "引"的 K 回应：我是一个动作
      - 经过匹配，"兵"对"曹操"和"引"的关注度最高
      - 最终"兵"的表示融合了"曹操引"的上下文信息
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        """
        参数：
            n_embd:     输入向量的维度（每个字用多少个数字表示）。
                        注意：传进来的 x 已经是 token embedding + position embedding 相加后的结果，
                        不是两个分开的向量。相加发生在 SelfAttentionLanguageModel.forward() 里。
            head_size:  这个注意力头的输出维度（Q、K、V 被投影到多大的空间）。
                        当前 head_size = n_embd = 64，所以输入输出维度一样大。
                        后面做多头注意力时，head_size 会变小（比如 4 个头，每头 64/4=16）。
            block_size: 最大上下文长度（决定因果遮罩的大小）
        """
        super().__init__()

        # Q、K、V 三个投影矩阵（对应文章中的 W_q、W_k、W_v）。
        # 它们把每个字的向量（n_embd 维）投影到 head_size 维的空间里。
        # 三个矩阵的输入都是同一个 x，但各自学到不同的变换：
        #   - query（W_q）：提取"我在找什么信息"（圆桌会议中的"提问"）
        #   - key  （W_k）：提取"我能提供什么信息"（圆桌会议中的"标签牌"）
        #   - value（W_v）：提取"我的实际语义内容"（圆桌会议中的"资料"）
        # 比喻：x 是完整简历，W_q/W_k/W_v 是三个不同的筛选器，
        #       分别从简历中挑出"你想问什么"、"你的标签"、"你的干货"。
        # 不用偏置（bias=False）是 Transformer 的惯例 —— 实验表明去掉偏置效果差不多，还能少点参数。
        self.query = nn.Linear(n_embd, head_size, bias=False)  # W_q: (n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size, bias=False)    # W_k: (n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size, bias=False)  # W_v: (n_embd, head_size)

        # 因果遮罩（Causal Mask）—— 一个下三角矩阵。
        # 作用：保证每个位置只能看到自己和前面的字，不能"偷看"后面的字。
        # 为什么？因为语言模型的任务是"预测下一个字"，如果能偷看到答案就没意义了。
        #
        # 用 register_buffer 而不是普通属性，这样：
        #   1. 它会跟着模型一起移到 GPU/CPU
        #   2. 保存模型时会自动包含它
        #   3. 但不会被当作"需要训练的参数"
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: 输入向量序列，形状 (B, T, n_embd)
               B = 批大小，T = 序列长度

        返回：
            经过注意力加权后的向量序列，形状 (B, T, head_size)
        """
        B, T, C = x.shape

        # 第 1 步：计算 Q、K、V（对应文章中 V = Embedding × W_v，Q 和 K 同理）
        # 每个字的向量 x 经过三个不同的线性变换（矩阵乘法），得到三种不同的表示：
        #   q = x × W_q → "我在找什么信息"
        #   k = x × W_k → "我能提供什么信息"
        #   v = x × W_v → "我的实际语义内容"（别的字来参考我时，拿到的就是这个）
        # 注意：x 已经融合了字义和位置信息（tok_emb + pos_emb），所以 Q/K/V 天然包含位置感知。
        q = self.query(x)  # (B, T, head_size) — 每个字的"提问"
        k = self.key(x)    # (B, T, head_size) — 每个字的"标签牌"
        v = self.value(x)  # (B, T, head_size) — 每个字的"语义资料"

        # 第 2 步：计算注意力分数 —— Q 和 K 的点积
        # 点积越大，说明两个字越"相关"。
        # 除以 sqrt(head_size) 是为了防止数值太大 ——
        # 如果不除，当 head_size 很大时，点积值会非常大，
        # 经过 softmax 后会变成"一个接近 1，其余接近 0"的极端分布，
        # 梯度几乎为零，模型学不动。这个技巧叫"缩放点积注意力"。
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)

        # 第 3 步：应用因果遮罩
        # 把上三角的位置填成负无穷，这样经过 softmax 后这些位置的权重变成 0。
        # 效果：第 i 个字只能关注第 0、1、...、i 个字，看不到后面的字。
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # 第 4 步：softmax 把分数转成概率（加起来等于 1）
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # 第 5 步：用注意力权重对 V 做加权求和
        # 每个字的新表示 = 前面所有字的 V 的加权平均，权重就是注意力分数。
        # 这样每个字就融合了上下文信息。
        out = wei @ v  # (B, T, head_size)

        return out


# ======================== 前馈网络 ========================

class FeedForward(nn.Module):
    """
    前馈网络（Feed-Forward Network）—— 给每个字的向量做一次"深加工"。

    注意力层让每个字看到了上下文（"开会讨论"），但只是做了信息汇总（加权平均）。
    前馈网络在这个基础上，对每个字独立做一次非线性变换，增加模型的表达能力。

    比喻：注意力 = 开会讨论收集信息，前馈网络 = 会后每个人独立思考、消化、总结。

    结构：两层线性变换 + 中间一个 ReLU 激活函数。
      - 第一层：把向量从 n_embd 维"展开"到 4 * n_embd 维（扩大思考空间）
      - ReLU：非线性激活（让模型能学到复杂的模式，而不只是简单的线性关系）
      - 第二层：把向量"压缩"回 n_embd 维（恢复原来的大小，方便后续处理）

    为什么要先展开再压缩？
    就像写笔记：先在草稿纸上展开思路（高维空间里有更多"余地"去发现规律），
    然后把要点浓缩成简洁的结论。4 倍的展开比例是 Transformer 论文的经验值。
    """

    def __init__(self, n_embd: int):
        """
        参数：
            n_embd: 输入和输出的向量维度
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 展开：n_embd → 4 * n_embd
            nn.ReLU(),                        # 非线性激活
            nn.Linear(4 * n_embd, n_embd),   # 压缩：4 * n_embd → n_embd
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: 输入向量序列，形状 (B, T, n_embd)
        返回：
            处理后的向量序列，形状 (B, T, n_embd)
        """
        return self.net(x)


# ======================== 多头注意力 ========================

class MultiHeadAttention(nn.Module):
    """
    多头注意力（Multi-Head Attention）—— 同时从多个角度"开会"。

    单头注意力就像一场会议只讨论一个话题；
    多头注意力就像同时开 n_head 场平行会议，每场讨论不同的话题，最后把结论合并。

    比如处理"曹操引兵追赶"时：
      - 第 1 个头可能关注"谁发起动作"（曹操 → 引）
      - 第 2 个头可能关注"动作和对象"（引 → 兵）
      - 第 3 个头可能关注"事件链"（引兵 → 追赶）
      - 第 4 个头可能关注"远距离关系"（曹操 → 追赶）

    每个头的"思考空间"变小了（head_size = n_embd / n_head），
    但多个头合在一起覆盖了更多角度，总体效果比一个大头更好。
    最后通过一个投影层把所有头的输出合并回 n_embd 维。
    """

    def __init__(self, n_embd: int, n_head: int, head_size: int, block_size: int):
        """
        参数：
            n_embd:     输入和输出的向量维度
            n_head:     注意力头的数量（同时开几场"会议"）
            head_size:  每个头的维度（= n_embd // n_head）
            block_size: 最大上下文长度
        """
        super().__init__()
        # 创建 n_head 个独立的注意力头，每个头有自己的 Q/K/V 矩阵
        # 用 nn.ModuleList 而不是普通 list，这样 PyTorch 能正确管理这些头的参数
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size) for _ in range(n_head)
        ])
        # 投影层：把 n_head 个头的输出拼接后，映射回 n_embd 维
        # 输入维度 = n_head * head_size = n_embd，输出维度 = n_embd
        # 这个投影层让模型学会"如何最好地合并各个头的结论"
        self.proj = nn.Linear(n_head * head_size, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数：
            x: 输入向量序列，形状 (B, T, n_embd)
        返回：
            多头注意力处理后的向量序列，形状 (B, T, n_embd)
        """
        # 每个头独立处理输入，各自得到 (B, T, head_size) 的输出
        # 然后在最后一个维度上拼接：n_head 个 head_size 拼成 n_embd
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_head * head_size)
        # 通过投影层合并信息
        out = self.proj(out)  # (B, T, n_embd)
        return out


# ======================== 可组装的 Block 插件 ========================

class AttentionBlock(nn.Module):
    """
    注意力插件 —— 把注意力层包装成一个标准积木块。

    包含三个部分：
      1. LayerNorm：归一化，让输入的数值稳定在合理范围（防止数值爆炸或消失）
      2. 注意力层：单头或多头，根据 n_head 自动选择
      3. 残差连接：把注意力的输出加上原始输入（x + attention(x)）

    为什么需要残差连接？
    想象你在开会讨论一个问题：讨论完后你不会完全忘掉自己原来的想法，
    而是把讨论得到的新信息"叠加"到原有认知上。残差连接就是这个"叠加"操作。
    技术上说，残差连接解决了深层网络的梯度消失问题 —— 梯度可以通过"捷径"直接传回去。

    为什么需要 LayerNorm？
    每一层的输入数值范围可能差别很大，LayerNorm 把它们"拉齐"到统一的范围，
    就像考试前先统一度量衡，这样后续计算更稳定，训练更快收敛。
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        """
        参数：
            n_embd:     向量维度
            n_head:     注意力头数（1 = 单头，>1 = 多头）
            block_size: 最大上下文长度
        """
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        # 统一使用 MultiHeadAttention，即使 n_head=1 也走同一条路径。
        # 这样单头和多头的结构完全对称（都有投影层），对比实验更公平。
        head_size = n_embd // n_head
        self.attn = MultiHeadAttention(n_embd, n_head, head_size, block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm 架构：先归一化，再做注意力，最后加上残差
        # 这是 GPT-2 以后的主流做法（比原始 Transformer 的 Post-Norm 更稳定）
        return x + self.attn(self.ln(x))


class FFNBlock(nn.Module):
    """
    前馈网络插件 —— 把 FFN 包装成一个标准积木块。

    和 AttentionBlock 一样，包含 LayerNorm + FFN + 残差连接。
    这个插件完全独立于注意力层，可以和任何 Block 自由搭配：
      - 和单头注意力搭配：[AttentionBlock(n_head=1), FFNBlock]
      - 和多头注意力搭配：[AttentionBlock(n_head=4), FFNBlock]
      - 甚至可以单独使用（虽然效果不好，但技术上可行）
    """

    def __init__(self, n_embd: int):
        """
        参数：
            n_embd: 向量维度
        """
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.ln(x))


# ======================== 组装式模型 ========================

class AssembledModel(BaseLanguageModel):
    """
    积木式语言模型 —— 通过插件自由组装的模型。

    这个模型本身不包含任何具体的 Embedding、注意力或 FFN 实现，
    所有组件都通过参数传入，像积木一样组装：
      - embedding：嵌入插件（把字变成向量）
      - blocks：处理插件列表（注意力、FFN 等，按顺序串联）

    就像一个舞台：
      - 舞台本身（AssembledModel）只提供框架（输出层）
      - 入场方式（embedding）和节目内容（blocks）都可以自由更换

    用法示例：
      # Token+Position 嵌入 + 多头注意力 + FFN
      embedding = TokenPositionEmbedding(vocab_size, 64, 256)
      blocks = [AttentionBlock(64, 4, 256), FFNBlock(64)]
      model = AssembledModel(vocab_size, 64, 256, embedding, blocks)

      # 纯 Token 嵌入 + 单头注意力（类似 Bigram 的简单嵌入）
      embedding = TokenEmbedding(vocab_size, 64)
      blocks = [AttentionBlock(64, 1, 256)]
      model = AssembledModel(vocab_size, 64, 256, embedding, blocks)
    """

    def __init__(self, vocab_size: int, n_embd: int, block_size: int,
                 embedding: nn.Module, blocks: list[nn.Module]):
        """
        参数：
            vocab_size: 词表大小
            n_embd:     嵌入维度
            block_size: 最大上下文长度
            embedding:  嵌入插件，负责把 (B, T) 的字编号变成 (B, T, n_embd) 的向量
            blocks:     Block 列表，按顺序串联执行
        """
        super().__init__()
        self.block_size = block_size

        # 嵌入插件：把字的编号变成向量
        # 可以是 TokenEmbedding（纯字向量）或 TokenPositionEmbedding（字+位置）
        self.embedding = embedding

        # 把传入的 Block 列表注册为子模块
        # nn.ModuleList 确保 PyTorch 能正确管理这些 Block 的参数
        self.blocks = nn.ModuleList(blocks)

        # 最终的 LayerNorm：在所有 Block 处理完之后做一次归一化
        self.ln_final = nn.LayerNorm(n_embd)

        # 输出层：向量 → 词表大小的打分
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        前向传播：embedding → blocks → 输出打分。

        数据流动过程：
          字 → embedding 插件 → block1 → block2 → ... → LayerNorm → 输出打分
        """
        B, T = idx.shape

        # 嵌入：把字的编号变成向量（具体做法由传入的 embedding 插件决定）
        x = self.embedding(idx)  # (B, T, n_embd)

        # 依次通过每个 Block（就像流水线上的工序）
        for block in self.blocks:
            x = block(x)

        # 最终归一化 + 输出层
        x = self.ln_final(x)           # (B, T, n_embd)
        logits = self.lm_head(x)       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


# ======================== 插件工厂函数 ========================

def build_embedding(embedding_type: str, vocab_size: int, n_embd: int,
                    block_size: int) -> nn.Module:
    """
    根据类型名称创建 Embedding 插件 —— 嵌入层的"选配单"。

    参数：
        embedding_type: 嵌入类型，"token" 或 "token_position"
        vocab_size:     词表大小
        n_embd:         嵌入维度
        block_size:     最大上下文长度（仅 token_position 需要）

    返回：
        Embedding 插件实例
    """
    if embedding_type == "token":
        return TokenEmbedding(vocab_size, n_embd)
    elif embedding_type == "token_position":
        return TokenPositionEmbedding(vocab_size, n_embd, block_size)
    else:
        raise ValueError(f"未知的 Embedding 类型: {embedding_type}")


def build_blocks(block_names: list[str], n_embd: int, n_head: int,
                 block_size: int) -> list[nn.Module]:
    """
    根据名称列表创建 Block 实例 —— 积木的"组装说明书"。

    这个函数是连接"配置"和"实际模型"的桥梁：
      - train.py 用它根据配置创建模型
      - generate.py 用它根据 checkpoint 里保存的配置重建模型

    参数：
        block_names: Block 名称列表，如 ["attention", "ffn"]
        n_embd:     向量维度
        n_head:     注意力头数
        block_size: 最大上下文长度

    返回：
        Block 实例列表
    """
    blocks = []
    for name in block_names:
        if name == "attention":
            blocks.append(AttentionBlock(n_embd, n_head, block_size))
        elif name == "ffn":
            blocks.append(FFNBlock(n_embd))
        else:
            raise ValueError(f"未知的 Block 类型: {name}")
    return blocks


# ======================== Self-Attention 语言模型 ========================

class SelfAttentionLanguageModel(BaseLanguageModel):
    """
    带自注意力的语言模型 —— 在 Bigram 的基础上加入了"看上下文"的能力。

    和 Bigram 的区别：
      - Bigram：每个字独立查表，互不相干 → 只能学到"谁后面跟谁"
      - Self-Attention：每个字先"开会"交流信息，再做预测 → 能理解上下文

    模型结构：
      1. Token Embedding：字 → 向量（和 Bigram 类似，但向量维度不再等于词表大小）
      2. Position Embedding：给每个位置编号 → 向量（让模型知道每个字在第几个位置）
      3. Self-Attention Head：让每个字能看到前面的字，融合上下文信息
      4. Linear 输出层：向量 → 词表大小的打分

    为什么需要位置编码？
    因为注意力机制本身不知道字的顺序 —— "曹操引兵"和"兵引操曹"在它看来是一样的。
    加上位置编码后，第 1 个位置和第 4 个位置的向量不同，模型就能区分顺序了。
    """

    def __init__(self, vocab_size: int, n_embd: int = 64, block_size: int = 256):
        """
        参数：
            vocab_size: 词表大小（有多少个不同的字）
            n_embd:     嵌入维度（每个字用多少个数字表示）
                        Bigram 里这个值等于 vocab_size（4742），
                        这里我们用一个小得多的值（64），因为后面有注意力层来增强表达力。
            block_size: 最大上下文长度（模型最多能"看到"多少个字）
        """
        super().__init__()
        self.block_size = block_size

        # Token Embedding：字 → 向量
        # 和 Bigram 的区别：Bigram 的 Embedding 维度 = vocab_size（直接当打分用），
        # 这里的维度 = n_embd（只是个中间表示，还要经过注意力和线性层才变成打分）。
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # Position Embedding：位置 → 向量
        # 第 0 个位置有一个向量，第 1 个位置有另一个向量……
        # 这些向量会和字的向量相加，让模型知道"这个字在第几个位置"。
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 自注意力头：让每个字能"看到"前面的字（文章中的"圆桌会议"）
        # 参数说明：Head(输入维度=64, 输出维度=64, 最大上下文长度=256)
        # 输入维度 = 输出维度 = n_embd，这是当前的简单设计。
        # Head 内部会创建 W_q、W_k、W_v 三个矩阵，各自从输入中提取不同的信息。
        self.sa_head = Head(n_embd, n_embd, block_size)

        # 输出层：把注意力处理后的向量（n_embd 维）映射回词表大小（vocab_size 维）
        # 这样每个位置就得到了对所有候选字的打分
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        前向传播：字 → 嵌入 → 加位置 → 注意力 → 输出打分。

        参数：
            idx:     输入的字，形状 (B, T)
            targets: 正确答案，形状 (B, T)，生成时不传

        返回：
            logits: 对下一个字的预测打分，形状 (B, T, vocab_size)
            loss:   预测误差，生成时为 None
        """
        B, T = idx.shape

        # 第 1 步：Token Embedding —— 查表把字变成向量
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)

        # 第 2 步：Position Embedding —— 查表把位置变成向量
        # torch.arange(T) 生成 [0, 1, 2, ..., T-1]，代表每个位置的编号
        # device=idx.device 确保位置编号和输入在同一个设备（CPU 或 GPU）上
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )  # (T, n_embd)

        # 第 3 步：字向量 + 位置向量 → 融合后的向量
        # 相加后，x 同时包含了"我是哪个字"和"我在第几个位置"两种信息。
        # 后面 Head 里的 Q/K/V 变换接收的就是这个融合后的 x，不是分开的两个向量。
        # 为什么用加法而不是拼接？因为加法更简单高效，而且实验表明效果一样好。
        # pos_emb 形状是 (T, n_embd)，会自动广播到 (B, T, n_embd)。
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # 第 4 步：自注意力 —— 让每个字"开会"，融合上下文信息
        x = self.sa_head(x)  # (B, T, n_embd)

        # 第 5 步：输出层 —— 把向量映射成词表大小的打分
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


# ======================== 模型注册表 ========================
# 方便 train.py 和 generate.py 通过名字找到对应的模型类

MODEL_REGISTRY = {
    "bigram": BigramLanguageModel,
    "attention": SelfAttentionLanguageModel,
    # 以下模型通过 AssembledModel + Block 列表组装，在 train.py 中配置
    "attention_ffn": AssembledModel,
    "multihead": AssembledModel,
    "multihead_ffn": AssembledModel,
}
