import os
import matplotlib.pyplot as plt
import pandas as pd

csv_paths = {
    "CNN": "CNN_runs/per_epoch_results.csv",
    "RNN": "RNN_runs/per_epoch_results.csv",
    "LSTM": "LSTM_runs/per_epoch_results.csv",
}

# 折线图样式配置
styles = {
    "CNN": {"color": "#1f77b4", "marker": "o"},  # 蓝色 + 圆点
    "RNN": {"color": "#ff7f0e", "marker": "s"},  # 橙色 + 方块
    "LSTM": {"color": "#2ca02c", "marker": "^"},  # 绿色 + 三角
}

data = {}
for model_name, path in csv_paths.items():
    data[model_name] = pd.read_csv(path)

# 创建画布 (1 行 3 列的子图)
fig, axes = plt.subplots(1, 3, figsize=(10, 8), dpi=300)
fig.suptitle(
    "Performance Comparison: CNN vs RNN vs LSTM",
    fontsize=10,
    fontweight="bold",
)

# --- 子图 1: Training Loss ---
for model_name, df in data.items():
    axes[0].plot(
        df["Epoch"],
        df["Training_loss"],
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
    axes[1].plot(
        df["Epoch"],
        df["Validation_loss"],
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
    axes[2].plot(
        df["Epoch"],
        df["Validation_accuracy"],
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

# 调整布局并保存/展示图片
plt.tight_layout()
plt.savefig("plot/model_comparison_curves.png", bbox_inches="tight")
plt.show()