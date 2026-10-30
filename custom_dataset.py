import os
import pickle

import torch
from torch.utils.data import Dataset


def load_data(data_dir):
    sequences = []
    labels = []
    for i in os.listdir(data_dir):
        if not i.endswith(".pkl"):
            continue
        raw_data_path = os.path.join(data_dir, i)
        with open(raw_data_path, 'rb') as f:
            sample = pickle.load(f)  # dict
        seq = sample['seq']
        ssp = sample['ssp']
        sequences.append(seq)
        labels.append(ssp)
    print(f'example of seq:{sequences[0]}')
    print(f'example of ssp:{labels[0]}')
    print(f'example of seq:{sequences[100]}')
    print(f'example of ssp:{labels[100]}')

    return sequences, labels


# ======AA及Class Index======
# 对氨基酸实现one-hot-encoding
# 原先的CNN中，因为没有embedding这一步，所以氨基酸被赋值
# 采用embedding则可以实现对于氨基酸的one-hot-encoding
AminoAcids = "ACDEFGHIKLMNPQRSTVWY"
AA_Idx = {}
for i, aa in enumerate(AminoAcids):
    AA_Idx[aa] = i + 1  # 预留0 for padding
AA_Idx["<Pad>"] = 0

# 对于分类来讲，数字只作为loss_fn计算的index
# 因此也实现了one-hot-encoding
# for decoder_input_encoding
Label_Idx = {
    "H": 0,
    "E": 1,
    "C": 2
}  # 虽然也执行了训练步骤，但是因为不计入loss，所以不会影响参数学习

# for decoder_output_classification
Idx_Label = {
    0: "H",
    1: "E",
    2: "C"
}


# ======将输入str转换为int list======
def encode_seq(seq):
    encoded = []
    for aa in seq:
        encoded.append(AA_Idx[aa])
    return encoded

def encode_ssp(ssp):
    encoded = []
    for x in ssp:
        encoded.append(Label_Idx[x])
    return encoded

# =======dataset=======
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        super().__init__()
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    # 单个batch内包含的东西
    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        enc_valid_len = len(sequence)
        input_seq = torch.tensor(encode_seq(sequence), dtype=torch.long)  # 一般int表示int32
        target_ssp = torch.tensor(encode_ssp(label), dtype=torch.long)  # long表示int64
        return input_seq, target_ssp, enc_valid_len

# ======建表======
# 整理批次数据，使得批次数据长度相等，可以堆叠
def collate_fn(batch):
    input_seqs, target_ssps, enc_valid_lens = zip(*batch)

    max_len = max(enc_valid_lens)
    batch_size = len(batch)

    padding_inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
    padding_targets = torch.full(
        (batch_size, max_len),
        fill_value=-100,
        dtype=torch.long
    )

    for i, (seq, ssp, valid_len) in enumerate(zip(input_seqs, target_ssps, enc_valid_lens)):
        assert len(seq) == len(ssp) == valid_len, "输入，预测和长度向量不一致"

        padding_inputs[i, :valid_len] = seq
        padding_targets[i, :valid_len] = ssp

    enc_valid_lens = torch.tensor(enc_valid_lens, dtype=torch.long)

    return {
        "input_seqs": padding_inputs,
        "target_ssps": padding_targets,
        "enc_valid_lens": enc_valid_lens
    }


# ======Autoregression inputs======
#  计划 bos_id = 3, pad_id = 4
def build_decoder_inputs(target_ssps, bos_id, pad_id):
    """
    输入：
        target_ssps: (B, L)，包含 0、1、2 和 -100
    输出：
        dec_inputs: (B, L)，包含合法的 Decoder token ID
    """

    # 1. 创建一份新的 tensor，不修改原始 target_ssps
    #    将有效类别 ID 转为 Decoder token ID
    #    将 -100 转为 pad_id
    decoder_token_ids = torch.clone(target_ssps)
    decoder_token_ids[decoder_token_ids == -100] = pad_id

    # 2. 创建 (B, 1) 的 BOS 列
    #    dtype 和 device 与输入一致
    bos_column = torch.full((target_ssps.size(0), 1), bos_id, dtype=torch.long, device=target_ssps.device)

    # 3. 拼接 BOS 列和 decoder_token_ids 的前 L-1 列
    #    保持总长度仍为 L
    dec_inputs = torch.cat((bos_column, decoder_token_ids[:, :-1]), dim=1)
    assert dec_inputs.shape[1] == target_ssps.shape[1], "左移后的dec_inputs与padded_ssps形状不一致"

    return dec_inputs