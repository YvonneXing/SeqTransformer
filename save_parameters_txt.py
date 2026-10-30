import torch
from pathlib import Path

def save_parameters_txt(state_dict, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 完整显示张量，避免中间数值被省略为 ...
    # precision=8 表示显示精度，不会改变原张量的数值
    torch.set_printoptions(profile="default", precision=8, linewidth=120)

    try:
        with output_path.open("w", encoding="utf-8") as f:
            for name, tensor in state_dict.items():
                # 用于查看，不参与梯度计算；转到 CPU 后输出
                tensor_cpu = tensor.detach().cpu()

                f.write("=" * 80 + "\n")
                f.write(f"name: {name}\n")
                f.write(f"size: {tensor_cpu.size()}\n")
                f.write(f"dtype: {tensor_cpu.dtype}\n")
                f.write(f"numel: {tensor_cpu.numel()}\n")
                f.write("tensor:\n")
                f.write(str(tensor_cpu))
                f.write("\n\n")
    finally:
        # 恢复 PyTorch 默认打印配置
        torch.set_printoptions(profile="default")

    print(f"参数已保存到：{output_path}")