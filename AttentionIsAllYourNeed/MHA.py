'''
多头注意力实现
'''
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_size // num_heads # 每个头的维度，二者必须整除
        # 初始化Q、K、V的投影矩阵，将输入词向量线性变换为Q、K、V，维度保持一致
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # 输出线性层，将拼接后的多头注意力输出变换为所需的输出维度，这里的维度爆出一致
        self.o_linear = nn.Linear(hidden_size, hidden_size)


    def forward(self, hidden_state, attention_mask = None):
        # hidden_state 形状: (batch_size, seq_len, hidden_size)
        query = self.q_linear(hidden_state) # batch_size, seq_len, hidden_size
        key = self.k_linear(hidden_state) # batch_size, seq_len, hidden_size
        value = self.v_linear(hidden_state) # batch_size, seq_len, hidden_size
        k_d = key.size(-1)

        attention_score = torch.matmul(query, key.transpose(-1,-2))/torch.sqrt(torch.tensor(k_d))

        if attention_mask is not None:
            attention_score += attention_mask * -1e9
        
        attention_result = torch.matmul(torch.softmax(attention_score, -1), value)
        
