"""
Bigram 语言模型 —— 最简单的语言模型。

想象一个只会看前一个字就猜下一个字的人：
看到"曹"就猜"操"，看到"操"就猜"大"……
这就是 Bigram 模型做的事。

它是「从零手搓 LLM」系列的起点。
后续我们会逐步给它加上"看更多字"的能力（注意力机制）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """
    Bigram 语言模型。

    核心思路：
        1. 事先准备一张大表，表里记录了"每个字后面最可能跟什么字"
        2. 输入一个字，去表里查一行，就得到了下一个字的预测
        3. 训练的过程就是不断修正这张表，让它的预测越来越准

    这张表就是所谓的 Embedding（嵌入表）。
    整个模型没有注意力机制，没有上下文理解，纯粹靠统计"谁后面跟谁"。
    """

    def __init__(self, vocab_size: int):
        super().__init__()

        # 这张表就是整个模型的全部。
        #
        # 打个比方：假设词表里有 4742 个字，那这张表就是一个 4742 行 × 4742 列的大表格。
        # 第 i 行记录的是：看到第 i 个字之后，下一个字分别是第 0、1、2、...、4741 个字的可能性。
        #
        # nn.Embedding 做的事很简单：输入一个数字 i，返回第 i 行。
        # 它和普通数组的区别是：它能参与训练，PyTorch 会自动帮我们调整表里的数值。
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        前向传播：拿到输入的字，去查表，得到对下一个字的预测。

        参数：
            idx:     输入的字（已转成数字），形状 (B, T)
                     B = 一批处理多少条，T = 每条多少个字
                     训练时 B=64（一批 64 条样本），生成时 B=1（只有一条序列）。
                     PyTorch 规定输入必须带 batch 维度，就像餐厅后厨不管是
                     64 桌同时点菜还是只有 1 桌，用的都是同一套出餐流程。
            targets: 正确答案（训练时用），生成时不传

        返回：
            logits: 模型对下一个字的预测，形状 (B, T, C)
                    C = 词表大小（4742），代表"预测下一个字时，对每个候选字的打分"。
                    注意：C 不是当前字本身的特征向量，而是在 4742 个候选字上的
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
            #   x = [却, 说, 曹, 操, 引, 兵, 追, 赶]
            #   y = [说, 曹, 操, 引, 兵, 追, 赶, 关]
            #   "却"的预测 vs 正确答案"说" → 算一个 loss
            #   "说"的预测 vs 正确答案"曹" → 算一个 loss
            #   "曹"的预测 vs 正确答案"操" → 算一个 loss
            #   ……8 个位置各算一个 loss，最后取平均。
            # 一批 64 组 × 每组 8 个位置 = 512 个"预测 vs 正确答案"的对比。
            #
            # PyTorch 的 cross_entropy 要求数据摆成特定的形状：
            # 预测值要是 (样本数, 类别数)，正确答案要是 (样本数,)。
            # 所以我们把 (B, T, C) 展平成 (B*T, C)，把 (B, T) 展平成 (B*T,)。
            # 也就是把 (64, 8, 4742) → (512, 4742)，512 个预测摊平一起算。
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()  # 生成时不需要计算梯度，关掉可以省内存、跑更快
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        逐字生成文本。

        过程就像一个人在写字：
          1. 看最后一个字 → 查表得到下一个字的概率
          2. 按概率随机抽一个字（而不是每次都选概率最高的，这样更有变化）
          3. 把新字接在后面
          4. 重复以上步骤

        参数：
            idx:            起始文字（已转成数字），比如"却说"
            max_new_tokens: 要续写多少个字
        """
        for _ in range(max_new_tokens):
            # 把当前所有文字喂给模型，拿到每个位置的预测。
            # 但我们只需要最后一个位置的预测（因为我们要续写的是最后一个字后面的内容）。
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # 只取最后一个字的预测打分

            # 把"打分"转换成"概率"。
            # softmax 的作用：把一组任意数字变成 0~1 之间、加起来等于 1 的概率分布。
            # 比如 [3.2, 1.0, 0.5] → [0.78, 0.09, 0.05, ...]
            probs = F.softmax(logits, dim=-1)

            # 按概率随机抽一个字。
            # 为什么不直接选概率最高的？因为那样每次生成的结果都一模一样，很无聊。
            # 随机采样让输出更丰富多样，就像人写文章也不会每次用同样的词。
            idx_next = torch.multinomial(probs, num_samples=1)

            # 把新字拼到已有文字的后面，然后继续生成下一个字。
            # 注意：生成时 B=1，T 每轮加 1。比如输入"却说曹操"(1,4)，
            # 生成一个字后变成 (1,5)，再生成变成 (1,6)……序列越来越长。
            idx = torch.cat([idx, idx_next], dim=1)

        return idx
