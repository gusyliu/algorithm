# 多头注意力

## 1. 多头注意力

**system**: 

```
You are a helpful assistant.
```

**user**: 

```
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 每个头的维度，二者必须整除
        
        # 初始化 Q、K、V 的投影矩阵，将输入词向量线性变换为 Q、K、V，维度保持一致
        self.q_linear = nn.Linear(hidden_size, hidden_size) 
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
        # 输出线性层，将拼接后的多头注意力输出变换为所需的输出维度，这里维度保持一致
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_state, attention_mask=None):
        # hidden_state 形状: (batch_size, seq_len, hidden_size)
        batch_size = hidden_state.size(0)  # 获取批量大小

        # 计算 Q、K、V，线性变换
        query = self.q_linear(hidden_state)  # (batch_size, seq_len, hidden_size)
        key = self.k_linear(hidden_state)    # (batch_size, seq_len, hidden_size)
        value = self.v_linear(hidden_state)  # (batch_size, seq_len, hidden_size)

        # 分割多头，将每个头的维度拆分出来
        query = self.split_head(query)  # (batch_size, num_heads, seq_len, head_dim)
        key = self.split_head(key)      # (batch_size, num_heads, seq_len, head_dim)
        value = self.split_head(value)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力分数，使用缩放点积注意力机制
        # attention_scores 形状: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 添加注意力掩码（seq_len, seq_len），掩码位置（1）的值为负无穷
        if attention_mask is not None:
            attention_scores += attention_mask * -1e9
        
        # 对注意力分数进行归一化，得到注意力概率
        attention_probs = torch.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # 计算注意力输出，通过注意力概率加权值
        output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 对多头注意力输出进行拼接
        # output.transpose(1, 2) 将 num_heads 和 seq_len 维度转置
        # 将形状调整为 (batch_size, seq_len, hidden_size)
        output = output.transpose(1, 2).reshape(batch_size, -1, self.head_dim * self.num_heads)
        
        # 通过线性层将拼接后的输出变换为所需的输出维度
        output = self.o_linear(output)  # (batch_size, seq_len, hidden_size)
        
        return output

    def split_head(self, x):
        batch_size = x.size(0)  # 获取批量大小
        # x 形状: (batch_size, seq_len, hidden_size)
        # 将 hidden_size 分割为 num_heads 和 head_dim
        return x.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # 返回形状: (batch_size, num_heads, seq_len, head_dim)

为什么在split_head中，x.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)为什么不直接变为x.reshape(batch_size, self.num_heads, -1, self.head_dim)，而是经过transpose(1, 2)变换
```

**assistant**: 

```
在`split_head`方法中，使用`reshape`后接`transpose`而非直接`reshape`到`(batch_size, num_heads, seq_len, head_dim)`，是为了保证**数据分组的正确性**。具体原因如下：

---

### 1. **数据存储顺序的兼容性**
假设原始输入`x`的形状为 `(batch_size, seq_len, hidden_size)`，其中 `hidden_size = num_heads * head_dim`。例如：
- `batch_size = 2`
- `seq_len = 3`
- `hidden_size = 8`（假设 `num_heads = 2`, `head_dim = 4`）

此时，`x`的最后一个维度 `hidden_size` 的数据是按如下方式连续存储的：  

头1的特征： [h1_1, h1_2, h1_3, h1_4]  
头2的特征： [h2_1, h2_2, h2_3, h2_4]


若直接使用 `x.reshape(batch_size, num_heads, seq_len, head_dim)`，会导致：  
- **错误的数据分组**：每个头会错误地分配到不同位置的词向量（如头1可能分到词1的头1特征，而头2分到词1的头2特征）。  
- **逻辑混乱**：每个头应处理所有词向量的同一子空间（如头1处理所有词的头1特征）。

---

### 2. **正确的分割逻辑**
通过以下两步操作：

x.reshape(batch_size, seq_len, num_heads, head_dim)  # 按词向量顺序分割多头
x.transpose(1, 2)  # 交换 num_heads 和 seq_len 维度

最终得到形状 `(batch_size, num_heads, seq_len, head_dim)`，其含义为：
- **每个头独立处理所有词向量的同一子空间**（如头1处理所有词的头1特征）。
- 确保注意力计算时，每个头的子空间数据是连续且完整的。

---

### 3. **反例：直接 reshape 的问题**
若直接使用 `x.reshape(batch_size, num_heads, seq_len, head_dim)`：
- **数据错位**：每个头会错误地分配到不同词的特征。  
  例如，头1可能分到词1的头1特征，但头2分到词2的头1特征，导致注意力计算完全错误。
- **内存布局不兼容**：PyTorch的 `reshape` 严格按内存顺序重组数据，而 `hidden_size` 的特征是按头连续存储的，必须通过 `transpose` 调整维度顺序。

---

### 总结
通过先 `reshape` 为 `(batch_size, seq_len, num_heads, head_dim)` 再 `transpose`，可以确保：
1. 每个头处理所有词向量的同一子空间。
2. 数据的内存顺序与多头注意力机制的设计一致。
3. 避免因直接 `reshape` 导致的数据错位和逻辑错误。
```

**user**: 

```
reshape 的工作原理
```

**assistant**: 

```
在 PyTorch 中，`reshape` 是一种**不改变数据内存顺序**的维度变换操作，它通过**按内存顺序重新解释张量的形状**来实现。以下是其核心工作原理：

---

### 1. **内存顺序不变**
`reshape` **不重新排列数据**，而是根据张量的**物理存储顺序**（即内存中的原始排列）重新划分维度。  
例如：  
假设一个张量 `x` 的原始形状为 `(2, 3)`，数据在内存中按行优先存储：

x = [[a, b, c],
     [d, e, f]]

内存顺序为：`[a, b, c, d, e, f]`  
若调用 `x.reshape(3, 2)`，新形状为：

[[a, b],
 [c, d],
 [e, f]]

**数据未变**，只是按内存顺序重新划分维度。

---

### 2. **维度必须兼容**
新形状的**总元素数**必须与原形状一致。例如：
- 原形状 `(2, 3)` → 6 个元素
- 新形状 `(3, 2)` → 6 个元素（合法）
- 新形状 `(4, 2)` → 8 个元素（非法，会报错）

---

### 3. **与 `transpose` 的关键区别**
- `reshape` **不改变数据顺序**，仅调整维度解释方式。  
- `transpose` **交换维度**（可能改变内存顺序），例如将 `(batch, seq, heads, dim)` 变为 `(batch, heads, seq, dim)`。  
- **联合使用场景**：  
  在多头注意力中，`reshape` + `transpose` 的组合可实现**逻辑维度的正确分割**，同时保持数据连续性（见后文示例）。

---

### 4. **示例：多头注意力的 `split_head`**
假设输入 `x` 的形状为 `(batch=2, seq=3, hidden=8)`，其中 `hidden=num_heads=2 * head_dim=4`。

#### 直接 `reshape` 的问题：

x_reshaped = x.reshape(2, 2, 3, 4)  # (batch, num_heads, seq, head_dim)

此时数据的内存顺序为：

头1的词1特征 → 头1的词2特征 → 头1的词3特征 → 头2的词1特征 → ...

这会导致**每个头无法独立处理所有词的特征**（头2的特征被分散到不同词的位置）。

#### 正确做法：`reshape` + `transpose`

x_reshaped = x.reshape(2, 3, 2, 4)  # (batch, seq, num_heads, head_dim)
x_transposed = x_reshaped.transpose(1, 2)  # (batch, num_heads, seq, head_dim)

此时数据的内存顺序为：

头1的词1特征 → 头1的词2特征 → 头1的词3特征 → 头2的词1特征 → ...

每个头可以**连续访问所有词的特征子空间**。

---

### 5. **为什么 `reshape` 不能替代 `transpose`？**
- `reshape` 只能**按内存顺序重新划分维度**，无法交换维度顺序。  
- 若要改变维度逻辑顺序（如将 `seq` 和 `num_heads` 交换），必须使用 `transpose`。

---

### 总结
| 操作       | 作用                          | 内存顺序 | 适用场景                     |
|------------|-------------------------------|----------|------------------------------|
| `reshape`  | 按内存顺序重新划分维度         | 不变     | 调整形状，保持数据连续性     |
| `transpose`| 交换维度顺序                   | 可能改变 | 调整维度逻辑顺序（如多头分割）|

在多头注意力中，`reshape` + `transpose` 的组合确保了：  
1. 数据的内存连续性（通过 `reshape`）。  
2. 维度的逻辑正确性（通过 `transpose`）。
```



--------------------


<a href="https://chatboxai.app" style="display: flex; align-items: center;">
<img src='https://chatboxai.app/icon.png' style='width: 40px; height: 40px; padding-right: 6px'>
<b style='font-size:30px'>Chatbox AI</b>
</a>
