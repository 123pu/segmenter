# -*- coding: utf-8 -*-
"""
transformer blocks
"""
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor



class FeedForward(nn.Cell):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Dense(dim, hidden_dim, has_bias=True)
        self.act = nn.GELU()   # 高斯误差线性单元激活函数
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Dense(hidden_dim, out_dim, has_bias=True)
        self.drop = nn.Dropout(1. - dropout)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Cell):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Dense(dim, dim * 3, has_bias=True)
        self.attn_drop = nn.Dropout(1. - dropout)
        self.proj_drop = nn.Dropout(1. - dropout)
        self.proj = nn.Dense(dim, dim, has_bias=True)
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.softmax = ops.Softmax()

    def construct(self, x, mask=None):
        B, N, C = x.shape
        x = self.qkv(x)
        x = self.reshape(x, (B, N, 3, self.heads, C // self.heads))
        x = self.transpose(x, (2, 0, 3, 1, 4))
        qkv = (x)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        trans_k = self.transpose(k, (0, 1, 3, 2))
        attn = (ops.matmul(q, trans_k)) * self.scale

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        amv = ops.matmul(attn, v)
        x = self.transpose(amv, (0, 2, 1, 3))
        x = self.reshape(x, (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class DropPath(nn.Cell):
    """
    正则化方法
    """
    def __init__(self, drop_prob=0.1, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - drop_prob
        seed = min(seed, 0)

        self.shape = ops.Shape()
        self.ones = ops.Ones()
        self.dropout = nn.Dropout(self.keep_prob)

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)
            mask = self.ones((x_shape[0], 1, 1), ms.float32)
            x = self.dropout(mask) * x
        return x

class Block(nn.Cell):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm((dim,))
        self.norm2 = nn.LayerNorm((dim,))
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        if drop_path > 0.0:
            self.drop_path = DropPath(drop_prob=drop_path)
        else:
            self.drop_path = ops.Identity()

    def construct(self, x, mask=None, return_attention=False):
        x = self.norm1(x)
        y, attn = self.attn(x, mask=mask)
        if return_attention:
            return attn
        y_drop = self.drop_path(y)
        x1 = x + y_drop
        x2 = self.norm2(x1)
        x2 = self.mlp(x2)
        x_drop = self.drop_path(x2)
        x = x1 + x_drop
        return x

