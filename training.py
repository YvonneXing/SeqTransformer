from custom_dataset import build_decoder_inputs

def train_one_epoch(model, loader, loss_fn, optimizer, device,
                    bos_id, dec_pad_id, ignore_index):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        input_seqs = batch["input_seqs"].to(device)
        target_ssps = batch["target_ssps"].to(device)
        enc_valid_lens = batch["enc_valid_lens"].to(device)

        # 1. 用真实标签构造右移输入，函数内部已经包含.to(device)
        dec_inputs = build_decoder_inputs(
            target_ssps, bos_id, dec_pad_id
        )

        # 2. 清空梯度
        optimizer.zero_grad()

        # 3. 使用关键字，避免三个输入的顺序混淆
        logits, _ = model(
            enc_inputs=input_seqs,
            enc_valid_lens=enc_valid_lens,
            dec_inputs=dec_inputs,
        )

        # Transformer 输出 (B, L, 3)
        # CrossEntropyLoss 接收 (B, 3, L)
        # CrossEntropyLoss 要求分类类别维度放在dim=1
        loss = loss_fn(logits.transpose(1, 2), target_ssps)

        # 4. 反向传播、更新参数
        loss.backward()
        optimizer.step()

        # 5. 统计有效残基数量，累计加权 loss
        # target_ssp != -100 生成一个 bool tensor | .sum() 统计True个数 | .item() -> tensor(N) -> N(int)
        valid_tokens = (target_ssps != ignore_index).sum().item()

        # loss.item()是per batch对所有有效元素的平均，每个batch残基数量不同，因此使用per token loss更合理
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    each_epoch_loss = total_loss / total_tokens

    return each_epoch_loss  # 整个 epoch 的平均有效残基 loss