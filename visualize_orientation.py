import os
import sys
import torch
import numpy as np
import matplotlib

# 可视化点云和图像加载数据，存储在vis_results中

# 设置后端为 'Agg'，使其无需图形界面即可运行 (必须在导入 pyplot 之前设置)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
from torchvision import transforms

# 确保能导入项目模块
sys.path.append(os.getcwd())
from DataLoader.data_loader import ModelNet40NeighbourDataset, load_config

# 定义反归一化参数 (与 config.yaml 保持一致)
INV_MEAN = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]
INV_STD = [1 / 0.229, 1 / 0.224, 1 / 0.225]


def denormalize(tensor):
    """将归一化后的 Tensor 还原为可显示的图像"""
    inv_normalize = transforms.Normalize(mean=INV_MEAN, std=INV_STD)
    # 如果是 batch 维度 (C, H, W)，执行反归一化
    tensor = inv_normalize(tensor)
    # 转为 numpy 并调整维度为 (H, W, C)
    img = tensor.permute(1, 2, 0).numpy()
    # 截断到 [0, 1] 范围，防止显示异常
    img = np.clip(img, 0, 1)
    return img


def visualize_and_save():
    print("🎨 开始可视化检查 (保存图片模式)...")

    # 1. 加载配置
    config_path = './DataLoader/config.yaml'
    if not os.path.exists(config_path):
        print(f"❌ 错误: 找不到配置文件 {config_path}")
        return
    config = load_config(config_path)

    # 2. 准备 Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 加载数据集
    # 注意：请确保你的 config.yaml 中 pointcloud_root 路径是正确的！
    # 之前报错 RecursionError 就是因为路径不对。
    try:
        dataset = ModelNet40NeighbourDataset(
            root_dir=config['dataset']['root_dir'],
            transform=transform,
            expected_images_per_view=config['dataset']['expected_images_per_view'],
            pointcloud_root=config.get('pointcloud', {}).get('root_dir')
        )
    except RecursionError:
        print("\n❌ 严重错误：DataLoader 陷入无限递归！")
        print("   原因：找不到任何有效的点云文件。")
        print("   解决：请检查 config.yaml 中的 'pointcloud -> root_dir' 路径是否正确。")
        return
    except Exception as e:
        print(f"❌ 数据集初始化失败: {e}")
        return

    # 使用 shuffle=True 随机抽查
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # 创建保存目录
    save_dir = "./vis_results"
    os.makedirs(save_dir, exist_ok=True)
    print(f"✅ 数据集加载完成，图片将保存到目录: {save_dir}")

    # 4. 循环可视化并保存
    max_samples = 10  # 限制只保存前10张

    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            print(f"\n✅ 已保存 {max_samples} 张样本，程序结束。")
            break

        try:
            obj_id = batch['object_id'][0]
            category = batch['category'][0]

            # 检查点云是否存在
            if batch['pointcloud'] is None:
                print(f"⚠️ 跳过样本 {obj_id}: 点云数据丢失")
                continue

            pointcloud = batch['pointcloud'][0].numpy()  # (N, 3)
            views = batch['views']

            # 随机取一个视点名称
            view_name = list(views.keys())[0]

            # 取出该视点的【中心图】
            img_tensor = views[view_name][0, 0]
            img_display = denormalize(img_tensor)

            # --- 绘图 ---
            fig = plt.figure(figsize=(12, 6))
            fig.suptitle(f"Category: {category} | ID: {obj_id} | View: {view_name}", fontsize=14)

            # 左图：2D 图像
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(img_display)
            ax1.set_title("2D Image")
            ax1.axis('off')

            # 右图：3D 点云
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')

            # 随机采样显示
            if pointcloud.shape[0] > 1024:
                choice = np.random.choice(pointcloud.shape[0], 1024, replace=False)
                pc_show = pointcloud[choice]
            else:
                pc_show = pointcloud

            # 绘制散点
            ax2.scatter(pc_show[:, 0], pc_show[:, 1], pc_show[:, 2], s=2, c=pc_show[:, 2], cmap='viridis')

            ax2.set_title("3D Point Cloud")
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')

            # 设置坐标轴一致
            max_range = np.array([pc_show[:, 0].max() - pc_show[:, 0].min(),
                                  pc_show[:, 1].max() - pc_show[:, 1].min(),
                                  pc_show[:, 2].max() - pc_show[:, 2].min()]).max() / 2.0
            mid_x = (pc_show[:, 0].max() + pc_show[:, 0].min()) * 0.5
            mid_y = (pc_show[:, 1].max() + pc_show[:, 1].min()) * 0.5
            mid_z = (pc_show[:, 2].max() + pc_show[:, 2].min()) * 0.5
            ax2.set_xlim(mid_x - max_range, mid_x + max_range)
            ax2.set_ylim(mid_y - max_range, mid_y + max_range)
            ax2.set_zlim(mid_z - max_range, mid_z + max_range)

            # 保存文件
            filename = f"{save_dir}/{category}_{obj_id}_{view_name}.png"
            plt.savefig(filename)
            plt.close()  # 这一步很重要，释放内存

            print(f"💾 [{i + 1}/{max_samples}] 已保存: {filename}")

        except Exception as e:
            print(f"❌ 处理样本 {i} 时出错: {e}")
            continue


if __name__ == "__main__":
    visualize_and_save()