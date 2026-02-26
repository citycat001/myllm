"""
Bigram 语言模型 —— 最简单的语言模型。

本模块定义了一个字符级的二元（Bigram）模型，
仅根据当前字符预测下一个字符。
它是「从零手搓 LLM」系列的起点基线模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """
    Bigram 语言模型：仅根据当前字符预测下一个字符。

    架构：
        输入（token 索引）→ Embedding 查表 → Logits（下一个 token 的概率分布）

    整个模型就是一张 (vocab_size × vocab_size) 的嵌入表。
    第 i 行存的是"字符 i 后面最可能出现哪些字符"的原始分数。
    没有注意力机制，没有上下文窗口，没有位置编码。
    """

    def __init__(self, vocab_size: int):
        super().__init__()

        # 模型的核心：一张可学习的查找表。
        # nn.Embedding(vocab_size, vocab_size) 创建一个 (V, V) 矩阵。
        # 输入 token 索引 i，返回第 i 行 —— 一个长度为 V 的 logit 向量，
        # 每个 logit 代表下一个 token 的未归一化分数。
        #
        # 为什么用 nn.Embedding 而不是普通张量？
        #   - 它是 nn.Module，PyTorch 会自动追踪梯度
        #   - 按索引查表比矩阵乘法更高效
        #   - 后续升级为 Transformer 时可以无缝替换
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        前向传播：将 token 索引转换为下一个 token 的预测。

        参数：
            idx:     (B, T) token 索引张量
                     B = 批大小，T = 序列长度
            targets: (B, T) 真实的下一个 token，推理时传 None

        返回：
            logits: (B, T, C) 每个位置对下一个 token 的原始预测分数
                    C = vocab_size（候选字符数）
            loss:   提供 targets 时返回标量交叉熵损失，否则返回 None
        """

        # 在嵌入表中查找每个 token。
        # 输入 (B, T) 整数索引 → 输出 (B, T, C) logit 向量。
        # 每个位置独立获取预测（位置之间没有交互）。
        logits = self.token_embedding_table(idx)

        loss = None
        if targets is not None:
            # F.cross_entropy 要求输入形状为 (N, C) 和 (N,)，而不是 (B, T, C) 和 (B, T)。
            # 所以我们把 batch 和 time 两个维度展平成一个维度：
            #   logits:  (B, T, C) → (B*T, C)  —— N 个预测，每个在 C 个类别上
            #   targets: (B, T)   → (B*T,)     —— N 个真实标签
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)

            # 交叉熵 = softmax + 负对数似然。
            # 它衡量模型的预测概率分布与真实答案之间的差距。
            # loss 越低 = 预测越准。
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()  # 关闭梯度追踪 —— 生成时不需要，节省内存
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        自回归地逐字生成新 token。

        流程：预测 → 采样 → 拼接 → 重复

        参数：
            idx:            (B, T) 起始上下文（可以是单个 token）
            max_new_tokens: 要生成的字符数

        返回：
            (B, T + max_new_tokens) 张量，包含原始上下文 + 生成的 token
        """
        for _ in range(max_new_tokens):
            # 用模型获取所有位置的 logits。
            # 我们只关心最后一个位置的预测（最新的 token）。
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (B, C) —— 只保留最后一个时间步

            # 通过 softmax 将原始 logits 转换为概率分布。
            # softmax(x_i) = e^x_i / sum(e^x_j) —— 所有值变为 0~1，总和为 1。
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # 根据概率分布随机采样下一个 token。
            # 为什么用采样而不是取最大值（argmax）？采样引入多样性 ——
            # argmax 总是产生相同的确定性（且无聊/重复的）输出。
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 将新 token 拼接到序列末尾，继续生成。
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)

        return idx
