import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from custom_dataset import (
    load_data,
    ProteinDataset,
    collate_fn
)
from transformer import Transformer
from training import train_one_epoch
from validation import valid_one_epoch


def main():
    # ======1. 设置随机种子、device 和配置======
    data_dir = "./assignment1_data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    patience = 10  # 设置早停，连续n轮valid Q3没有提高就提前停止
    batch_size = 32
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-4
    seed = 42

    ignore_index = -100
    bos_id = 3
    dec_pad_id = 4

    # vocab_size
    amino_size, class_size = 21, 3
    dec_class_size = class_size + 2

    embed_size = 128
    num_layers = 2
    num_hiddens = 128
    num_heads = 4
    dropout = 0.1
    ffn_num_hiddens = 256

    # 种子应用
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 参数检查
    print(f"device: {device}")
    assert num_hiddens % num_heads == 0, "num_hiddens与num_heads不怕匹配"


    # ======2. 创建本次实验目录======
    # 目录内保存 CSV、最佳 checkpoint、模型配置
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S_%f")
    run_dir = Path("Transformer_runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "best_model.pt"
    config_path = run_dir / "config.json"
    test_path = run_dir / "test_metrics.csv"

    # 模型构造参数单独保存
    model_config = {
        "amino_size": amino_size,
        "dec_class_size": dec_class_size,
        "class_size": class_size,
        "embed_size": embed_size,
        "num_layers": num_layers,
        "num_hiddens": num_hiddens,
        "num_heads": num_heads,
        "dropout": dropout,
        "ffn_num_hiddens": ffn_num_hiddens,
        "use_bias": False
    }

    # 项目构造参数单独保存
    config = {
        "data_dir": data_dir,
        "device": str(device),
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "seed": seed,
        "optimizer": "SGD",
        "loss_fn": "CrossEntropyLoss",
        "bos_id": bos_id,
        "dec_pad_id": dec_pad_id,
        "ignore_index": ignore_index,
        "model_config": model_config
    }

    with config_path.open("w", encoding="utf-8") as f:
        # 不把中文等不符合ascii的转义，缩进为2
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"本次实验目录：{run_dir}")


    # ======3. 读取数据，按原来的 80% / 10% / 10% 划分======
    # 创建 ProteinDataset 和 DataLoader
    # train: shuffle=True
    # val/test: shuffle=False
    sequences, labels = load_data(data_dir)
    # 划分数据
    train_seq, oth_seq, train_ssp, oth_ssp = train_test_split(
        sequences,
        labels,
        test_size=0.2,
        random_state=seed
    )

    test_seq, valid_seq, test_ssp, valid_ssq = train_test_split(
        oth_seq,
        oth_ssp,
        test_size=0.5,
        random_state=seed
    )

    # 创建dataset
    train_dataset = ProteinDataset(train_seq, train_ssp)
    test_dataset = ProteinDataset(test_seq, test_ssp)
    valid_dataset = ProteinDataset(valid_seq, valid_ssq)

    # 创建数据加载迭代器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(
        f"样本数：train={len(train_dataset)}",
        f"test={len(test_dataset)}",
        f"valid={len(valid_dataset)}"
    )


    # ======4. 创建 Transformer 和 optimizer======
    model = Transformer(
        amino_size,
        dec_class_size,
        class_size,
        embed_size=embed_size,
        num_layers=num_layers,
        num_hiddens=num_hiddens,
        num_heads=num_heads,
        dropout=dropout,
        ffn_num_hiddens=ffn_num_hiddens,
        use_bias=False
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    best_q3 = -1.0
    best_state = None
    best_epoch = None
    epochs_without_improvement = 0


    # ======5. 打开CSV，逐轮记录训练和验证结果======
    filenames = [
        "epoch",
        "train_tf_loss",
        "val_tf_loss",
        "val_ar_loss",
        "val_q3",
        "val_f1"
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=filenames)
        writer.writeheader()


    # ======6. 逐epoch训练======
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, loss_fn, optimizer, device,
                bos_id, dec_pad_id, ignore_index
            )

            val_tf_loss, val_ar_loss, val_q3, val_f1 = valid_one_epoch(
                model, valid_loader, loss_fn, device,
                bos_id, dec_pad_id, ignore_index
            )

            # 几轮每一个epoch的结果
            writer.writerow({
                "epoch": epoch,
                "train_tf_loss": train_loss,
                "val_tf_loss": val_tf_loss,
                "val_ar_loss": val_ar_loss,
                "val_q3": val_q3,
                "val_f1": val_f1
            })

            f.flush()  # 将缓冲区的内容交给操作系统，方便训练过程中查看记录

            # 这里假设 Q3 和 F1 的取值范围是 0～1
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train TF loss={train_loss:.4f} | "
                f"val TF loss={val_tf_loss:.4f} | "
                f"val AR loss={val_ar_loss:.4f} | "
                f"val Q3={val_q3:.2%} | "
                f"val macro-F1={val_f1:.4f}"
            )

            # 按验证集的自回归 Q3 选择最佳模型
            if val_q3 > best_q3:
                best_q3 = val_q3
                best_epoch = epoch


                # 复制一份独立的权重，保存在cpu上
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

                epochs_without_improvement = 0

                # torch.save(模型参数字典, model_path)
                checkpoint = {
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "best_val_ar_q3": best_q3,
                    "val_tf_loss": val_tf_loss,
                    "val_ar_loss": val_ar_loss,
                    "val_ar_macro_f1": val_f1,
                    "config": config,
                }

                torch.save(checkpoint, checkpoint_path)

                print(f"已保存最佳模型：epoch={best_epoch}")

            else:
                epochs_without_improvement += 1

            # early stopping
            if (
                patience is not None
                and epochs_without_improvement >= patience
            ):
                print(f"连续{patience}轮验证Q3未提高，提前停止")
                break

    # ======7. 训练结束后，恢复最佳模型======
    if best_state is None:
        raise RuntimeError(
            "没有获得最佳权重，请检查epoch数量和验证Q3"
        )

    # 默认的 load_state_dict() 会把数值复制到现有模型参数所在的设备
    # 模型仍然留在原来的 GPU 上，不需要因此再次调用 model.to(device)
    model.load_state_dict(best_state)
    print(
        f"加载最佳模型：epoch={best_epoch}",
        f"val Q3={best_q3:.2%}"
    )


    # ======8. 在模型选择完成后，使用测试集=======
    test_tf_loss, test_ar_loss, test_q3, test_f1 = valid_one_epoch(
        model, test_loader, loss_fn, device,
        bos_id, dec_pad_id, ignore_index,
        num_examples=8
    )

    test_metrics = {
        "best_epoch": best_epoch,
        "best_val_ar_q3": best_q3,
        "test_tf_loss": test_tf_loss,
        "test_ar_loss": test_ar_loss,
        "test_q3": test_q3,
        "test_f1": test_f1,
    }


    # ======9. 记录最终测试结果======
    with test_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=test_metrics.keys())
        writer.writeheader()
        writer.writerow(test_metrics)

    print(
        f"Test | "
        f"TF loss={test_tf_loss:.4f} | "
        f"Q3={test_q3:.2%} | "
        f"macro-F1={test_f1:.4f}"
    )


if __name__ == "__main__":
    main()


