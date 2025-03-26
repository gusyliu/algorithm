import torch
from torch import nn

class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()

        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads

        # 初始化 Q、K、V 投影矩阵，注意这里的 K V 比原来更小
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, self.head_dim)
        self.v_linear = nn.Linear(hidden_size, self.head_dim)

        self.o_linear = nn.Linear(hidden_size, hidden_size)

    def split_head(self, x, num_heads = None):
        # x batch_size seq_len hidden_size
        batch_size = x.size(0)
        if num_heads is None:
            num_heads = self.num_heads
        return x.reshape(batch_size, -1, num_heads, self.head_dim).transpose(1, 2)
    def forward(self, hidden_state, attention_mask = None):
        batch_size = hidden_state.size(0)

        query = self.q_linear(hidden_state) # (batch_size, seq_len, hidden_size)
        key = self.k_linear(hidden_state) # (batch_size, seq_len, head_dim)
        value = self.v_linear(hidden_state) # (batch_size, seq_len, head_dim)

        # batch_size * seq_len * head_dim * num_heads == batch_size * seq_len * hidden_size
        
        # 分割头部，K，V矩阵也要加上一个维度
        # 后两个维度统一 
        query = self.split_head(query) # (batch_size, num_heads, seq_len, head_dim)
        key = self.split_head(key, 1) # (batch_size, 1, seq_len, head_dim)
        value = self.split_head(value, 1) # (batch_size, 1, seq_len, head_dim)

        # 计算注意力分数，自动广播，(batch_size, num_heads, seq_len, seq_len)
        attention_socre = torch.softmax(torch.matmul(query, key.transpose(-1, -2))/torch.sqrt(torch.tensor(self.head_dim, dtype= torch.float32)), dim = -1)

        if attention_mask is not None:
            attention_socre += -1e9 * attention_mask
        
        output = torch.matmul(attention_socre, value).transpose(-1, -2).reshape(batch_size, -1, self.head_dim * self.num_heads)

        return self.o_linear(output)



def test_MQA():
    batch_size = 128
    seq_len = 512
    hidden_size = 1024
    num_heads = 8
    
    # 随机生成输入数据
    hidden_state = torch.randn(batch_size, seq_len, hidden_size)  # (batch_size, seq_len, hidden_size)
    
    # 创建多头注意力模块
    mha = MultiQueryAttention(hidden_size, num_heads)
    
    # 计算多头注意力输出
    output = mha(hidden_state)
    
    print("Input shape:", hidden_state.shape)
    print("Output shape:", output.shape)
    
if __name__ == "__main__":
	test_MQA()