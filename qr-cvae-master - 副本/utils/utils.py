import torch
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os

clamp = lambda x: torch.clamp(x, min=-5, max=5)



def get_kdeplot(data, output_dir ,data_name):
    df = pd.DataFrame(data, columns=["Dimension 1", "Dimension 2", "Dimension 3", "Dimension 4"])

    # 检查并创建保存文件夹

    # 创建画布和子图
    fig, axes = plt.subplots(2, 1, figsize=(6, 10))  # 上下排列 2 个子图，宽度为 6，高度为 10

    # 第一幅图：第一维度和第二维度
    sns.kdeplot(
        x=df["Dimension 1"], y=df["Dimension 2"], cmap="Blues_d", ax=axes[0]
    )
    axes[0].set_title("Dimension 1 vs Dimension 2")
    axes[0].set_xlabel("Dimension 1")
    axes[0].set_ylabel("Dimension 2")
    axes[0].set_xticks([])  # 去掉 x 轴刻度
    axes[0].set_yticks([])  # 去掉 y 轴刻度

    # 第二幅图：第二维度和第三维度
    sns.kdeplot(
        x=df["Dimension 2"], y=df["Dimension 3"], cmap="Blues_d", ax=axes[1]
    )
    axes[1].set_title("Dimension 2 vs Dimension 3")
    axes[1].set_xlabel("Dimension 2")
    axes[1].set_ylabel("Dimension 3")
    axes[1].set_xticks([])  # 去掉 x 轴刻度
    axes[1].set_yticks([])  # 去掉 y 轴刻度

    # 调整布局
    plt.tight_layout()

    # 保存图片为 PDF 和 PNG 格式
    plt.savefig(os.path.join(output_dir, data_name+".pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, data_name+".png"), format="png")
