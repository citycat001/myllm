"""
分词器家族 —— 把文字变成数字。

分词器（Tokenizer）是 LLM 的"翻译官"：
  - 训练时：把人类的文字翻译成模型能理解的数字序列
  - 生成时：把模型输出的数字序列翻译回人类能读的文字

这个文件包含了所有分词器的实现：
  - CharTokenizer: 字符级分词，一个字 = 一个 token（最简单的方案）

后续会加入更智能的分词方案（如 BPE），到时候只需要新增一个类，
实现同样的 encode/decode 接口，就能无缝替换。
"""


class CharTokenizer:
    """
    字符级分词器 —— 最简单的分词方案。

    做法：把文本里每个不重复的字符编个号。
      - "曹" → 1038, "操" → 2893
      - "曹操" → [1038, 2893]

    优点：实现简单，不需要额外的训练过程
    缺点：词表大（每个汉字都是一个 token）、不能识别词组

    用法：
        tokenizer = CharTokenizer.from_text("却说曹操引兵追赶")
        encoded = tokenizer.encode("曹操")      # → [1038, 2893]
        decoded = tokenizer.decode([1038, 2893]) # → "曹操"
    """

    def __init__(self, stoi: dict[str, int], itos: dict[int, str]):
        """
        参数：
            stoi: 字 → 数字 的对照表（string to integer）
            itos: 数字 → 字 的对照表（integer to string）
        """
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """
        从文本中自动建立词表。

        做法：
          1. 找出文本中所有不重复的字符
          2. 排序（保证每次结果一样）
          3. 按顺序编号

        参数：
            text: 训练文本（如《三国演义》全文）

        返回：
            构建好的 CharTokenizer 实例
        """
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi, itos)

    def encode(self, text: str) -> list[int]:
        """把文字变成数字序列。 "曹操" → [1038, 2893]"""
        return [self.stoi[c] for c in text]

    def decode(self, tokens: list[int]) -> str:
        """把数字序列变回文字。 [1038, 2893] → "曹操" """
        return "".join([self.itos[i] for i in tokens])

    def to_dict(self) -> dict:
        """导出为字典，用于保存到 checkpoint。"""
        return {"type": "char", "stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_dict(cls, data: dict) -> "CharTokenizer":
        """从字典恢复，用于从 checkpoint 加载。"""
        # itos 的 key 应该是 int，但 JSON 序列化会把 int key 变成 str，
        # 这里做一次转换确保兼容性。
        itos = {int(k): v for k, v in data["itos"].items()}
        return cls(data["stoi"], itos)


# ======================== 分词器注册表 ========================

TOKENIZER_REGISTRY = {
    "char": CharTokenizer,
}


def load_tokenizer(data: dict):
    """
    从 checkpoint 保存的字典中恢复分词器。

    参数：
        data: tokenizer.to_dict() 导出的字典

    返回：
        对应类型的分词器实例
    """
    tokenizer_type = data["type"]
    TokenizerClass = TOKENIZER_REGISTRY[tokenizer_type]
    return TokenizerClass.from_dict(data)
