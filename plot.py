import os
import matplotlib.pyplot as plt
import pandas as pd


# TODO. 填写Transformer含时间戳的runs文件夹
csv_paths = {
    "CNN": "CNN_runs/per_epoch_results.csv",
    "RNN": "RNN_runs/per_epoch_results.csv",
    "LSTM": "LSTM_runs/per_epoch_results.csv",
    "Transformer": "Transformer_runs/待定/metrics.csv"
}

# 折线图样式配置
styles = {
    "CNN": {"color": "#1f77b4", "marker": "o"},  # 蓝色 + 圆点
    "RNN": {"color": "#ff7f0e", "marker": "s"},  # 橙色 + 方块
    "LSTM": {"color": "#2ca02c", "marker": "^"},  # 绿色 + 三角
    "Transformer": {"color": "#d62728", "marker": "*"}
}

data = {}
for model_name, path in csv_paths.items():
    data[model_name] = pd.read_csv(path)

# 创建画布 (1 行 3 列的子图)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
# axes(2, 2) -> (4, )
axes = axes.ravel()  # 展平axes，方便索引调用
fig.suptitle(
    "Performance Comparison: CNN vs RNN vs LSTM vs Transformer",
    fontsize=10,
    fontweight="bold",
)

# --- 子图 1: Training Loss ---
for model_name, df in data.items():
    if model_name == "Transformer":
        epoch_col = "epoch"
        loss_col = "train_tf_loss"
    else:
        epoch_col = "Epoch"
        loss_col = "Training_loss"

    axes[0].plot(
        df[epoch_col],
        df[loss_col],
        label=model_name,
        color=styles[model_name]["color"],
        marker=styles[model_name]["marker"],
        linewidth=0.5,
        markersize=1,
    )
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.6)

# --- 子图 2: Validation Loss ---
for model_name, df in data.items():
    if model_name == "Transformer":
        epoch_col = "epoch"
        val_loss = "val_tf_loss"
    else:
        epoch_col = "Epoch"
        val_loss = "Validation_loss"

    axes[1].plot(
        df[epoch_col],
        df[val_loss],
        label=model_name,
        color=styles[model_name]["color"],
        marker=styles[model_name]["marker"],
        linewidth=0.5,
        markersize=1,
    )
axes[1].set_title("Validation Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.6)

# --- 子图 3: Validation Accuracy (Q3) ---
for model_name, df in data.items():
    if model_name == "Transformer":
        epoch_col = "epoch"
        val_q3 = "val_q3"
    else:
        epoch_col = "Epoch"
        val_q3 = "Validation_accuracy"

    axes[2].plot(
        df[epoch_col],
        df[val_q3],
        label=model_name,
        color=styles[model_name]["color"],
        marker=styles[model_name]["marker"],
        linewidth=0.5,
        markersize=1,
    )
axes[2].set_title("Validation Accuracy (Q3)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy")
axes[2].legend()
axes[2].grid(True, linestyle="--", alpha=0.6)


# --- 子图 4: Training Loss ---
df = data["Transformer"]
axes[3].plot(
    df["epoch"],
    df["val_ar_loss"],
    label="Autoregressive (AR)",
    color="#0072B2",
    marker="o",
    linewidth=0.5,
    markersize=1,
)

axes[3].plot(
    df["epoch"],
    df["val_tf_loss"],
    label="Teacher forcing (TF)",
    color="#D55E00",
    marker="s",
    linewidth=0.5,
    markersize=1,
)
axes[3].set_title("Transformer Validation Loss: TF vs AR")
axes[3].set_xlabel("Epoch")
axes[3].set_ylabel("Loss")
axes[3].legend()
axes[3].grid(True, linestyle="--", alpha=0.6)


# 调整布局并保存/展示图片
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/model_comparison_curves.png", bbox_inches="tight")
plt.show()