# 绝对位置编码的一种
# 原始Transformer提出时采用了sinusoidal位置编码，通过使用不同频率的正弦和余弦的函数，使得模型捕获位置之间的复杂关系，且这些编码与序列中每个位置的绝对值有关
# 优势：
# 首先，正余弦函数的范围是在-1~+1，导出的位置编码与原词嵌入相加，不会使得结果偏离过远而破坏原有单词的语义信息。
# 其次，依据三角函数的基本性质，可以得知第 pos + k 个位置的编码是第 pos 个位置的编码的线性组合，这就意味着位置编码中蕴含着单词之间的距离信息。
import torch
from torch import nn
import math
import numpy as np

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]  # Simply use the tensor directly
        return x

# test
positional_encoder = PositionalEncoder(128)
input_tensor = torch.randn(1, 10, 128)  # 输入张量，形状为 (batch_size, seq_len, d_model)
output_tensor = positional_encoder(input_tensor)
print(output_tensor)