import torch
import torch.nn as nn
import math

class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 线性变换层
        self.Wq = nn.Linear(d_model, d_model)  # Query投影
        self.Wk = nn.Linear(d_model, d_model)  # Key投影
        self.Wv = nn.Linear(d_model, d_model)  # Value投影
        self.Wo = nn.Linear(d_model, d_model)  # 输出投影
        
    def forward(self, x, kv_cache=None):
        """
        x: 当前步输入 [batch_size, seq_len=1, d_model]
        kv_cache: 元组 (prev_keys, prev_values)
                   prev_keys: [batch, num_heads, cache_len, head_dim]
                   prev_values: 同上
        """
        batch_size = x.size(0)
        
        # 生成Q/K/V（当前步）
        Q = self.Wq(x)  # [batch, 1, d_model]
        K = self.Wk(x)
        V = self.Wv(x)
        
        # 分割多头
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, 1, dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 拼接缓存（如果存在）
        if kv_cache is not None:
            prev_K, prev_V = kv_cache
            K = torch.cat([prev_K, K], dim=2)  # 沿序列维度拼接
            V = torch.cat([prev_V, V], dim=2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, heads, 1, cache_len+1]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # [batch, heads, 1, dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出投影
        output = self.Wo(attn_output)  # [batch, 1, d_model]
        
        # 更新缓存
        new_kv_cache = (K, V) if self.training else None  # 训练时不缓存
        
        return output, new_kv_cache