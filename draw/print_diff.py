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

    model1 = "meta-llama-3.1-8b-instruct".lower()
    model2 = "evolcodellama-3.1-8b-instruct".lower()
    print("Comparing QKV tensors:", model1, "vs", model2)

    dataset = "lcc"
    tensor1_dir = f"/home/ysy/code/research/example/kvcache/{model1}/{dataset}"
    tensor2_dir = f"/home/ysy/code/research/example/kvcache/{model2}/{dataset}"
    
    png_out_dir = "diff_3d_plots"
    os.makedirs(png_out_dir, exist_ok=True)
    png_out_dir = os.path.join(png_out_dir, f"{model1}_vs_{model2}")
    os.makedirs(png_out_dir, exist_ok=True)
    png_out_dir = os.path.join(png_out_dir, dataset)
    os.makedirs(png_out_dir, exist_ok=True)

    prompt_cnt = 0
    for prompt1_dir in os.listdir(tensor1_dir):
      if not os.path.isdir(os.path.join(tensor2_dir, prompt1_dir)):
        print(f"Directory {prompt1_dir} not found in {tensor2_dir}, skipping.")
        continue
      if prompt_cnt > 4:
         break
      prompt_cnt += 1
      prompt1_dir_path = os.path.join(tensor1_dir, prompt1_dir)
      prompt2_dir_path = os.path.join(tensor2_dir, prompt1_dir)
      if os.path.isdir(prompt2_dir_path) and os.path.isdir(prompt1_dir_path):
        print(f"Found directory: {prompt1_dir_path}")
        os.makedirs(os.path.join(png_out_dir, prompt1_dir), exist_ok=True)
  
        layers1 = sorted([d for d in os.listdir(prompt1_dir_path) if os.path.isdir(os.path.join(prompt1_dir_path, d))])
        layers2 = sorted([d for d in os.listdir(prompt2_dir_path) if os.path.isdir(os.path.join(prompt2_dir_path, d))])
        # assert len(layers1) == len(layers2), f"Directory count mismatch: {len(layers1)} vs {len(layers2)}"
        layer_num = min(len(layers1), len(layers2))
        for idx, layer in enumerate(layers1[:layer_num]):
          if idx >= layer_num:
            break
          layer_path1 = os.path.join(prompt1_dir_path, layer)
          layer_path2 = os.path.join(prompt2_dir_path, layer)
          if os.path.isdir(layer_path1) and os.path.isdir(layer_path2):
            print(f"Found layer directories: {layer_path1}, {layer_path2}")
            os.makedirs(os.path.join(png_out_dir, prompt1_dir, layer), exist_ok=True)
            for file in os.listdir(layer_path1):
              if file.endswith(".pt"):
                tensor1_path = os.path.join(layer_path1, file)
                tensor2_path = os.path.join(layer_path2, file)
                print(f"Processing tensor files: {tensor1_path} and {tensor2_path}")
                tensor1 = torch.load(tensor1_path)
                tensor1 = tensor1.squeeze()[0]
                tensor2 = torch.load(tensor2_path)
                tensor2 = tensor2.squeeze()[0]
                assert tensor1.shape == tensor2.shape, f"Tensor shape mismatch: {tensor1.shape} vs {tensor2.shape}"
                diff_tensor = tensor1 - tensor2
                title = f"{model1} vs {model2} - {layer} - {file[:-3]}"
                plot_tensor_3d(diff_tensor, title=title)
                out_png_path = os.path.join(png_out_dir, prompt1_dir, layer, f"{file[:-3]}.png")
                plt.savefig(out_png_path, dpi=300)
                plt.close()
                print(f"Saved plot to: {out_png_path}")
