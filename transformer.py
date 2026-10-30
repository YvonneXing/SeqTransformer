import math
import torch
from torch import nn

from attention import MultiHeadAttention


# ===========
# Positional encoding
# ===========
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建position表
        self.position = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1)
        denominator = torch.pow(10000,
                                torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        x = x / denominator

        self.position[:, :, 0::2] = torch.sin(x)
        self.position[:, :, 1::2] = torch.cos(x)

    def forward(self, x, offset=0):
        x = x + self.position[:, offset : offset + x.shape[1], :].to(x.device)
        return self.dropout(x)



# ===========
# Positionwise feed-forward network
# ===========
class PositionwiseFFN(nn.Module):
    def __init__(self, ffn_num_inputs, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_inputs, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x


# ===========
# add & norm
# ===========
class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, x, y):
        return self.ln(x + self.dropout(y))


# ===========
# EncoderBlock
# ===========
class EncoderBlock(nn.Module):
    def __init__(self, q_size, k_size, v_size, num_hiddens,
                 num_heads, norm_shape, dropout, ffn_num_inputs, ffn_num_hiddens,
                 use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(q_size, k_size, v_size,
                                            num_hiddens, num_heads, dropout, use_bias)
        self.add_norm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionwiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.add_norm2 = AddNorm(norm_shape, dropout)

    def forward(self, x, valid_lens):
        x1 = self.attention(x, x, x, valid_lens)
        y = self.add_norm1(x, x1)
        y1 = self.ffn(y)
        z = self.add_norm2(y, y1)
        return z


# ===========
# TransformerEncoder
# ===========
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers,
                 q_size, k_size, v_size,
                 num_hiddens,num_heads, norm_shape,
                 dropout, ffn_num_inputs, ffn_num_hiddens,
                 use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(q_size, k_size, v_size,
                                              num_hiddens, num_heads, norm_shape,
                                              dropout, ffn_num_inputs, ffn_num_hiddens,
                                              use_bias))

    def forward(self, x, valid_lens):
        x = self.embedding(x) * math.sqrt(self.num_hiddens)
        x = self.pos_encoding(x)
        self._attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            x = blk(x, valid_lens)
            self._attention_weights[i] = blk.attention.attention.attention_weights

        return x


# ===========
# DecoderBlock
# ===========
class DecoderBlock(nn.Module):
    def __init__(self, q_size, k_size, v_size, num_hiddens,
                 num_heads, norm_shape, dropout, ffn_num_inputs, ffn_num_hiddens,
                 i, use_bias=False):
        super().__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(q_size, k_size, v_size, num_hiddens, num_heads, dropout, use_bias)
        self.add_norm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(q_size, k_size, v_size, num_hiddens, num_heads, dropout, use_bias)
        self.add_norm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionwiseFFN(ffn_num_inputs, ffn_num_hiddens, num_hiddens)
        self.add_norm3 = AddNorm(norm_shape, dropout)

    # state存了encoder端的信息和decoder每个模块的因果自注意力的keys_values
    def forward(self, x, state):
        enc_outputs = state[0]
        enc_valid_lens = state[1]

        # 初始时间步，对于training模式，一次性传入所有时间步；对于eval模式，传入<Bos>
        # keys_values(B, L, E)
        if state[2][self.i] is None:
            keys_values = x
        # 非初始时间步，对于eval模式，只传入一个时间步
        else:
            keys_values = torch.cat((state[2][self.i], x), dim=1)

        # 用于dec_valid_lens计算
        past_len = keys_values.shape[1] - x.shape[1]
        num_steps = x.shape[1]

        # 更新state
        state[2][self.i] = keys_values

        # dec_valid_lens形状(B, num_steps)
        # 需要与本次输入x(B, num_steps, H)对应
        dec_valid_lens = torch.arange(
            past_len + 1,
            past_len + 1 + num_steps,
            dtype=torch.long,
            device=x.device
        )
        batch_size = x.shape[0]
        dec_valid_lens = dec_valid_lens.repeat(batch_size, 1)

        assert dec_valid_lens.shape[1] == x.shape[1], "dec_valid_lens.shape应为(B, num_steps)"

        # x(B, num_steps, H)
        # num_steps在teacher forcing中是padded len
        # 在autoregressive是每次输入几步，例如在这个项目中，是 1
        x1 = self.attention1(x, keys_values, keys_values, dec_valid_lens)
        y = self.add_norm1(x, x1)
        y1 = self.attention2(y, enc_outputs, enc_outputs, enc_valid_lens)
        z = self.add_norm2(y, y1)
        z1 = self.ffn(z)
        output = self.add_norm3(z, z1)
        return output, state


# ===========
# TransformerDecoder
# ===========
class TransformerDecoder(nn.Module):
    def __init__(self, dec_class_size, class_size, embed_size, num_layers,
                 q_size, k_size, v_size, num_hiddens,
                 num_heads, norm_shape, dropout, ffn_num_inputs, ffn_num_hiddens,
                 use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(dec_class_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(q_size, k_size, v_size, num_hiddens,
                                              num_heads, norm_shape, dropout, ffn_num_inputs, ffn_num_hiddens,
                                              i, use_bias)
                                 )
        self.dense = nn.Linear(num_hiddens, class_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        state0 = [enc_outputs, enc_valid_lens, [None] * self.num_layers]
        return state0

    def forward(self, x, state):
        x = self.embedding(x) * math.sqrt(self.num_hiddens)
        if state[2][0] is None:
            offset = 0
        else:
            # state[2][0]和state[2][1]里边存的元素shape(B, num_steps, H)
            offset = state[2][0].shape[1]
        x = self.pos_encoding(x, offset)
        # 存放attention_weights的容器shape(2, len(blks))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            x, state = blk(x, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights

        x = self.dense(x)

        return x, state


# ===========
# Transformer 用于训练的架构，evaluation需要重新按时间步写
# ===========
class Transformer(nn.Module):
    """连接Encoder-Decoder"""
    def __init__(self, amino_size, dec_class_size, class_size,
                 embed_size, num_layers, num_hiddens,
                 num_heads, dropout, ffn_num_hiddens,
                 use_bias=False):
        super().__init__()
        self.encoder = TransformerEncoder(amino_size, embed_size, num_layers,
                                          embed_size, embed_size, embed_size, num_hiddens,
                                          num_heads, num_hiddens, dropout, num_hiddens, ffn_num_hiddens,
                                          use_bias)
        self.decoder = TransformerDecoder(dec_class_size, class_size, embed_size, num_layers,
                                          embed_size, embed_size, embed_size, num_hiddens,
                                          num_heads, num_hiddens, dropout, num_hiddens, ffn_num_hiddens,
                                          use_bias)

    def forward(self, enc_inputs, enc_valid_lens, dec_inputs):
        enc_outputs = self.encoder(enc_inputs, enc_valid_lens)
        dec_state0 = self.decoder.init_state(enc_outputs, enc_valid_lens)

        # 提供历史标签
        logits, dec_state_ultimate = self.decoder(dec_inputs, dec_state0)

        return logits, dec_state_ultimate

