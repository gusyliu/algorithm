'''
多头注意力实现
'''
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads # 每个头的维度，二者必须整除
        # 初始化Q、K、V的投影矩阵，将输入词向量线性变换为Q、K、V，维度保持一致
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # 输出线性层，将拼接后的多头注意力输出变换为所需的输出维度，这里的维度保持一致
        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def split_heads(self, x):
        # x 的形状： batch_size, seq_len, hidden_size
        # 获取批量大小的信息
        batch_size = x.size(0)

        # 将hidden_size分割为 num_heads 和 head_dim
        return x.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # 返回形状： batch_size, num_heads, seq_len, head_dim


    def forward(self, hidden_state, attention_mask = None):
        batch_size = hidden_state.size(0)  # 获取批量大小

        # hidden_state 形状: (batch_size, seq_len, hidden_size)
        query = self.q_linear(hidden_state) # batch_size, seq_len, hidden_size
        key = self.k_linear(hidden_state) # batch_size, seq_len, hidden_size
        value = self.v_linear(hidden_state) # batch_size, seq_len, hidden_size

        query, key, value = self.split_heads(query), self.split_heads(key), self.split_heads(value)
        # query, key, value 的形状： batch_size, num_heads, seq_len, head_dim

        k_d = key.size(-1)

        # batch_size, num_heads, seq_len, seq_len
        attention_score = torch.matmul(query, key.transpose(-1,-2))/torch.sqrt(torch.tensor(k_d))

        if attention_mask is not None:
            attention_score += attention_mask * -1e9
        
        # batch_size, num_heads, seq_len, head_dim
        output = torch.matmul(torch.softmax(attention_score, -1), value)
        
        # 对多头注意力输出进行拼接
        # output.transpose(1, 2) 将 num_heads 和 seq_len 维度转置
        # 将形状调整为 (batch_size, seq_len, hidden_size)
        output = output.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.num_heads)
        
        # 通过线性层将拼接后的输出变换为所需的输出维度
        output = self.o_linear(output)  # (batch_size, seq_len, hidden_size)
        
        return output

def test_MHA():
    batch_size = 128
    seq_len = 512
    hidden_size = 1024
    num_heads = 8
    
    # 随机生成输入数据
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)  # (batch_size, seq_len, hidden_size)
    
    # 创建多头注意力模块
    mha = MultiHeadAttention(hidden_size, num_heads)
    
    # 计算多头注意力输出
    output = mha(hidden_state)
    
    print("Input shape:", hidden_state.shape)
    print("Output shape:", output.shape)
    
if __name__ == "__main__":
	test_MHA()