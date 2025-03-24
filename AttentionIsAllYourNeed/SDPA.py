'''
缩放点积注意力（Scaled Dot-Product Attention）早于 Transformer 被提出，受到的关注并不多，其内部只实现了 的注意力计算。
1. 输入是 query 和 key-value，注意力机制首先计算 query 与每个 key 的关联性
2. 每个关联性作为每个 value 的权重 (weight)，各个权重与 value 的乘积相加得到输出。
3. SDPA 可以被认为是 MHA 的中间步骤
'''

import torch
from torch import nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
       super().__init__()

    def forward(self, query, key, value, attention_mask = None):
        # query, key, value 的形状： (batch_size, seq_len, hidden_size)

        # 计算注意力分数
        # key.transpose(-1, -2) 将最后两个维度进行转置，以进行点积(batch_size, hidden_size, seq_len)
        # attention_scores 的形状：(batch_size, seq_len, seq_len)
        d_k = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # 添加注意力掩码（seq_len, seq_len）,掩码位置（1）的值为负无穷
        if attention_mask is not None:
            attention_scores += attention_mask * -1e9
        
        # 对注意力分数进行归一化，得到注意力概率
        attention_probs = torch.softmax(attention_scores, dim = -1) # (batch_size, seq_len, seq_len)
        print("attention_probs:", attention_probs.shape)

        # 计算注意力输出，通过注意力概率加权值
        attention_output = torch.matmul(attention_probs, value) # (batch_size, seq_len, hidden_size)

        return attention_output
    
def test_attn():
    batch_size = 128
    seq_len = 512
    hidden_size = 1024
    
    query = torch.randn(batch_size, seq_len, hidden_size)  # (batch_size, seq_len, hidden_size)
    key = torch.randn(batch_size, seq_len, hidden_size)    # (batch_size, seq_len, hidden_size)
    value = torch.randn(batch_size, seq_len, hidden_size)  # (batch_size, seq_len, hidden_size)

    sdpa = ScaledDotProductAttention()
    output = sdpa(query, key, value)
    
    print("Query shape:", query.shape)
    print("Key shape:", key.shape)
    print("Value shape:", value.shape)
    print("Output shape:", output.shape)
    
if __name__ == "__main__":
	test_attn()