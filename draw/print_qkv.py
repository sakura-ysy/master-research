import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (用于 3D 绘图)
from matplotlib import cm

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_tensor_3d(tensor, title="3D Tensor Visualization"):
    """
    绘制一个形状为 [token, channel] 的 tensor 的三维热力图。
    
    参数:
        tensor: torch.Tensor 或 numpy.ndarray，形状为 [token, channel]
        title: str, 图像标题
    """
    print("Tensor shape:", tensor.shape)
    # 如果是 torch.Tensor，转换为 numpy
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().to(torch.float32).cpu().abs().numpy()
    else:
        data = np.abs(np.array(tensor))

    if data.ndim != 2:
        raise ValueError(f"输入 tensor 的维度必须为 2，但得到的是 {data.shape}")

    token_len, channel_len = data.shape
    X = np.arange(token_len)
    Y = np.arange(channel_len)
    X, Y = np.meshgrid(X, Y, indexing="ij")

    Z = data

    # 绘图
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True)

    ax.set_xlabel("Token", labelpad=8)
    ax.set_ylabel("Channel", labelpad=8)
    ax.set_zlabel("Absolute value", labelpad=8)
    ax.set_title(title, pad=12)
    
    # 调整视角，使图像视觉效果更好
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.savefig("tensor_plot.png", dpi=300)


if __name__ == "__main__":
    # 生成一个示例 tensor，形状 [token, channel, hidden]

    model = "Qwen2.5-7B-Instruct"
    model = model.lower()
    dataset = "evol-instruct-python-1k"
    tensor_dir = f"/home/ysy/code/research/test/kvcache/{model}/{dataset}"
    png_out_dir = "qkv_3d_plots"
    os.makedirs(png_out_dir, exist_ok=True)
    png_out_dir = os.path.join(png_out_dir, model)
    os.makedirs(png_out_dir, exist_ok=True)
    png_out_dir = os.path.join(png_out_dir, dataset)
    os.makedirs(png_out_dir, exist_ok=True)

    prompt_cnt = 0
    for subdir in os.listdir(tensor_dir):
      if prompt_cnt > 0:
         break
      subdir_path = os.path.join(tensor_dir, subdir)
      if os.path.isdir(subdir_path):
        prompt_cnt += 1
        print(f"Found directory: {subdir_path}")
        os.makedirs(os.path.join(png_out_dir, subdir), exist_ok=True)
        for lay_dir in os.listdir(subdir_path):
          lay_dir_path = os.path.join(subdir_path, lay_dir)
          if os.path.isdir(lay_dir_path):
            print(f"Found layer directory: {lay_dir_path}")
            os.makedirs(os.path.join(png_out_dir, subdir, lay_dir), exist_ok=True)
            for file in os.listdir(lay_dir_path):
              if file.endswith(".pt"):
                tensor_path = os.path.join(lay_dir_path, file)
                print(f"Processing tensor file: {tensor_path}")
                tensor = torch.load(tensor_path)
                tensor = tensor.squeeze()
                tensor = tensor[0]
                title = f"{model} - {lay_dir} - {file[:-3]}"
                plot_tensor_3d(tensor, title=title)
                out_png_path = os.path.join(png_out_dir, subdir, lay_dir, f"{file[:-3]}.png")
                plt.savefig(out_png_path, dpi=300)
                plt.close()
                print(f"Saved plot to: {out_png_path}")


    # tensor = torch.load(tensor_path)
    # tensor = tensor.squeeze()
    # tensor = tensor[0]
    # plot_tensor_3d(tensor, title="Tensor Activation Heatmap")