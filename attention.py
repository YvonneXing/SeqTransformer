import torch
import math
from torch import nn


# ===========
# Masked softmax for attention
# ===========
# 将<pad>部分对应的attention_score替换为非常大的负值
def sequence_mask(X, valid_lens, value):
    # X(B,Q,K) -> X(B*Q,K)
    shape = X.shape
    X = X.reshape(-1, X.shape[-1])
    # 必须满足valid_lens.shape[0] == X.shape[0]
    assert valid_lens.shape[0] == X.shape[0]
    for i, valid_len in enumerate(valid_lens):
        X[i, valid_len:] = value
    return X.reshape(shape)


def masked_softmax(scores, valid_lens):
    # scores是attention分数矩阵(B, Q, K)
    # 多头注意力中，valid_lens已经完成B*num_heads
    # Q是序列中氨基酸的个数，一个batch为一条序列
    # 在autoregressive时，无需valid_lens, valid_lens is None
    if valid_lens is not None:
        if valid_lens.ndim == 1:
            # 在encoder部分，valid_lens应该为1D，同batch内所有q共享mask
            # repeat_interleave如果不指定dim，则先展平，后重复，操作后变为1维
            valid_lens = torch.repeat_interleave(valid_lens, scores.shape[1])
            scores = sequence_mask(scores, valid_lens, value=-1e6)
        elif valid_lens.ndim == 2:
            # 在causal mask时出现，对于dec_valid_lens
            valid_lens = valid_lens.reshape(-1, )
            scores = sequence_mask(scores, valid_lens, value=-1e6)
        else:
            raise ValueError("Invalid shape for valid_lens")
    else:
        raise ValueError("dec_valid_lens needed when autoregression")

    return nn.functional.softmax(scores, dim=-1)


# ===========
# DotProductAttention
# ===========
class DotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, valid_lens=None):
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        self._attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), v)

    @property
    def attention_weights(self):
        return self._attention_weights


# ===========
# MultiHeadAttention
# ===========
def transpose_qkv(x, num_heads):
    """为了多头注意力并行计算而变换形状"""
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x= x.permute(0, 2, 1, 3)
    # 方便对每个batch和每个head进行attention_weights计算
    x = x.reshape(-1, x.shape[2], x.shape[3])
    return x


def transpose_output(x, num_heads):
    """逆transpose_qkv操作"""
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    return x


class MultiHeadAttention(nn.Module):
    def __init__(self, q_size, k_size, v_size, num_hiddens, num_heads,
                 dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(q_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(k_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(v_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, q, k, v, valid_lens=None):
        # 对于每个batch的每个head，例如queries(B*num_heads,Q,per_head_feature)
        queries = transpose_qkv(self.W_q(q), self.num_heads)
        keys = transpose_qkv(self.W_k(k), self.num_heads)
        values = transpose_qkv(self.W_v(v), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)

        head_organized_output = self.attention(queries, keys, values, valid_lens)
        output = transpose_output(head_organized_output, self.num_heads)

        return self.W_o(output)