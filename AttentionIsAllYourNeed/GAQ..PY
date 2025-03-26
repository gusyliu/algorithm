# import torch
# from torch import nn

# class GroupQueryAttention(torch.nn.Module):
#     def __init__(self, hidden_size, num_heads, group_num):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = hidden_size // num_heads
#         self.group_num = group_num  # 组的数量
        
#         # 初始化 Q、K、V 投影矩阵，注意这里的 K V 做了折衷
#         self.q_linear = nn.Linear(hidden_size, hidden_size)  # (hidden_size, hidden_size)
#         self.k_linear = nn.Linear(hidden_size, self.group_num * self.head_dim)  # (hidden_size, group_num * head_dim)
#         self.v_linear = nn.Linear(hidden_size, self.group_num * self.head_dim)  # (hidden_size, group_num * head_dim)
        
#         self.o_linear = nn.Linear(hidden_size, hidden_size)  # (hidden_size, hidden_size)
        
#     def forward(self, hidden_state, attention_mask=None):
#         batch_size = hidden_state.size(0)
        
#         query = self.q_linear(hidden_state)  # (batch_size, seq_len, hidden_size)
#         key = self.k_linear(hidden_state)    # (batch_size, seq_len, group_num * head_dim)
#         value = self.v_linear(hidden_state)  # (batch_size, seq_len, group_num * head_dim)
        
#         # 分割头部，将每个头的维度拆分出来
#         query = self.split_head(query)  # (batch_size, num_heads, seq_len, head_dim)
#         key = self.split_head(key, self.group_num)  # (batch_size, num_heads, seq_len, head_dim)
#         value = self.split_head(value, self.group_num)  # (batch_size, num_heads, seq_len, head_dim)
        
#         # 计算注意力分数，自动广播，(batch_size, num_heads, seq_len, seq_len)
#         attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
#         if attention_mask is not None:
#             attention_scores += attention_mask * -1e9  
        
#         attention_probs = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
#         output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_dim)
        
#         # 对注意力输出进行拼接，形状: (batch_size, seq_len, hidden_size)
#         output = output.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.num_heads)
        
#         # 通过线性层将拼接后的输出变换为所需的输出维度
#         output = self.o_linear(output)  # (batch_size, seq_len, hidden_size)
        
#         return output

#     def split_head(self, x, group_num=None):
#         batch_size, seq_len = x.size()[:2]  # 获取批量大小和序列长度
        
#         if group_num is None:
#             return x.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         else:
#             # 将 hidden_size 分割为 group_num 和 head_dim
#             x = x.reshape(batch_size, -1, group_num, self.head_dim).transpose(1, 2)
#             # 再将其手动 expand 到相同大小
#             x = x[:, :, None, :, :].expand(batch_size, group_num, self.num_heads // group_num, seq_len, self.head_dim).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
#             return x 	# 形状: (batch_size, num_heads, seq_len, head_dim)