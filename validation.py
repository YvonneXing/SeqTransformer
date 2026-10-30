import torch
from custom_dataset import build_decoder_inputs
from sklearn.metrics import accuracy_score, f1_score

# ===========
# evaluation prediction function
# ===========
# evaluation与training不同，需要按照时间步输入并预测
@torch.no_grad()
def predict_autoregressive(model, input_seqs, enc_valid_lens,
                           bos_id, dec_pad_id, ignore_index):
    """
    输入：
        input_seqs: (B, L)
        enc_valid_lens: (B,)

    输出：
        predicted_ssps: (B, L)
        只有真实长度范围内的位置需要用于评分。
    """
    model.eval()

    # 1. Encoder 只计算一次
    enc_outputs = model.encoder(input_seqs, enc_valid_lens)

    # 2. 初始化一份新的 Decoder state
    state = model.decoder.init_state(enc_outputs, enc_valid_lens)

    # 3.  创建 (B, 1) 的 BOS 输入，作为初始输入
    batch_size, padded_len = input_seqs.shape
    current_tokens = torch.full(
        (batch_size, 1),
        bos_id,
        dtype=torch.long,
        device=input_seqs.device
    )

    # 4. 准备保存预测logits分数的容器和预测结果的容器
    autoregressive_logits = torch.zeros(
        (batch_size, padded_len, 3),
        dtype=torch.float32,
        device=input_seqs.device
    )
    predicted_ssps = torch.full(
        (batch_size, padded_len),
        ignore_index,
        dtype=torch.long,
        device=input_seqs.device
    )

    # 5. 循环到当前 batch 的最大有效长度
    for t in range(padded_len):
    # 每次直接调用 model.decoder，而不是外层 model.forward
    # 输入 current_tokens 和 state，得到当前步的 logits(B, 1, 3)，以及更新后的 state
    # logits因为是针对training设计的，所以一次会对所有时间步执行预测，需要只取最后一个时间步作为本次预测
        step_logits, state = model.decoder(current_tokens, state)
        autoregressive_logits[:, t, :] = step_logits[:, 0, :]  # 各自取一个(B,3)的切片用于保存

    # 沿类别维 argmax，取得当前预测类别 next_token(B, 1)
        next_token = step_logits.argmax(dim=-1)
    # 在填写预测结果之前，先判断哪些位置有效
        valid_row = t < enc_valid_lens
    # 保存预测结果,valid_row筛选有效行
        predicted_ssps[valid_row, t] = next_token[valid_row, 0]
    # 将预测类别作为下一步输入，保持 current_tokens(B, 1)
        current_tokens = next_token.clone()
    # 已超过某条序列有效长度的位置，不参与评分
    # 可以给这些已结束的样本current_tokens设置为 dec_pad_id
        finished = (t + 1) >= enc_valid_lens
        current_tokens[finished, 0] = dec_pad_id

    return predicted_ssps, autoregressive_logits


# ===========
# valid_one_epoch
# ===========
@torch.no_grad()
def valid_one_epoch(model, loader,loss_fn, device,
                    bos_id, dec_pad_id, ignore_index,
                    num_examples=0):
    model.eval()

    Teacher_forced_total_loss = 0.0
    autoregressive_total_loss = 0.0
    total_tokens = 0

    # 用于显示存放target和对应的prediction结果，未展平
    predictions_visual = []
    targets_visual = []
    idx_to_label = {0: "H", 1: "E", 2: "C"}

    # 计算f1 score的已经展平的结果
    all_predictions = []
    all_targets = []

    for batch in loader:
        # 1. 取出三个字段，并移动到 device
        input_seqs = batch["input_seqs"].to(device)
        target_ssps = batch["target_ssps"].to(device)
        enc_valid_lens = batch["enc_valid_lens"].to(device)

        # 2. 评估Teacher-forced loss
        # 不能代替自回归的loss，因为teacher-forced的每一步输入都一定正确
        # 但是自回归如果有一步预测错，依然会当作下一个时间步的输入
        # 前提：修正 Decoder，使 eval 下整段输入仍有 causal mask?
        dec_inputs = build_decoder_inputs(
            target_ssps, bos_id, dec_pad_id
        )

        Teacher_forced_logits, _ = model(
            enc_inputs=input_seqs,
            enc_valid_lens=enc_valid_lens,
            dec_inputs=dec_inputs,
        )

        Teacher_forced_loss = loss_fn(Teacher_forced_logits.transpose(1, 2), target_ssps)

        # 计算valid_tokens
        valid_tokens = (target_ssps != ignore_index).sum().item()
        total_tokens += valid_tokens

        # 按有效残基数量累计Teacher_forced loss
        Teacher_forced_total_loss += Teacher_forced_loss.item() * valid_tokens

        # 3. 评估自回归预测：不传入 target_ssps
        predicted_ssps, autoregressive_logits = predict_autoregressive(
            model ,input_seqs, enc_valid_lens, bos_id, dec_pad_id, ignore_index
        )
        # autoregressive_logits(B, L, 3)，是否只要求target_ssps带padding？
        autoregressive_loss = loss_fn(autoregressive_logits.transpose(1, 2), target_ssps)
        autoregressive_total_loss += autoregressive_loss.item() * valid_tokens

        # 4. 排除 padding
        # 虽然CrossEntrpyLoss可以自动忽略pad，但是accuracy和f1不行
        valid_mask = target_ssps != ignore_index

        # 在test时展示target对应的prediction
        # 先按序列保存用于展示的结果
        # 此时 predicted_ssps 和 target_ssps 都还是 (B, L) 的张量
        remaining = max(0, num_examples - len(predictions_visual))
        take = min(remaining, target_ssps.size(0))

        if take > 0:
            # 只把需要展示的样本移到 CPU
            preview_predictions = predicted_ssps[:take].cpu()
            preview_targets = target_ssps[:take].cpu()
            preview_masks = valid_mask[:take].cpu()

            for i in range(take):
                row_mask = preview_masks[i]

                # 使用同一个 mask，确保 target 和 prediction 位置对应
                target_ids = preview_targets[i][row_mask].tolist()
                prediction_ids = preview_predictions[i][row_mask].tolist()

                target_text = "".join(idx_to_label[x] for x in target_ids)
                prediction_text = "".join(
                    idx_to_label[x] for x in prediction_ids
                )

                targets_visual.append(target_text)
                predictions_visual.append(prediction_text)

        # 布尔索引触发展平
        # 收集 predicted_ssps[valid_mask] (B, padded_len) -> (B * 每行valid_len)
        flat_predictions = predicted_ssps[valid_mask].cpu().tolist()
        # 收集 target_ssps[valid_mask] (B, padded_len) -> (B * 每行valid_len)
        flat_targets = target_ssps[valid_mask].cpu().tolist()  # 从tensor变成普通的python数据

        all_predictions.extend(flat_predictions)  # extend逐元素添加
        all_targets.extend(flat_targets)


    # 5. 计算平均 TF loss、Autoregression loss、Q3、macro-F1
    # macro-F1 建议明确 labels=[0, 1, 2]
    # 针对Teacher_forcedavg和autoregressive的loss
    Teacher_forced_avg_loss = Teacher_forced_total_loss / total_tokens
    autoregressive_avg_loss = autoregressive_total_loss / total_tokens

    # 只针对autoregressive的accuracy
    accuracy = accuracy_score(all_targets, all_predictions)

    # 只针对autoregressive的f1 score
    f1 = f1_score(all_targets, all_predictions, labels=[0,1,2], average='macro',zero_division=0)

    # print target对应的precition
    for sample_id, (target_text, prediction_text) in enumerate(
            zip(targets_visual, predictions_visual), start=1):

        print(f"\nSample {sample_id} | length={len(target_text)}")

        # 每行显示 80 个残基，避免长序列难以对齐
        width = 80
        for start in range(0, len(target_text), width):
            end = min(start + width, len(target_text))

            target_part = target_text[start:end]
            prediction_part = prediction_text[start:end]

            matches = "".join(
                "|" if target == prediction else "."
                for target, prediction in zip(target_part, prediction_part)
            )

            print(f"Position   : {start + 1}-{end}")
            print(f"Target     : {target_part}")
            print(f"Prediction : {prediction_part}")
            print(f"Match      : {matches}")

    return Teacher_forced_avg_loss, autoregressive_avg_loss, accuracy, f1