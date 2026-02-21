import os
import sys
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from DataLoader.data_loader import ModelNet40NeighbourDataset, load_config
from Models.multi_view_visual_encoder import MultiViewVisualEncoder

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def find_latest_checkpoint(checkpoint_dir):
    """
    自动寻找最新的checkpoint文件夹
    :param checkpoint_dir: checkpoint根目录
    :return: 最新checkpoint文件夹路径
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint subdirectories found in: {checkpoint_dir}")
    
    # 解析时间戳并排序
    from datetime import datetime
    def parse_timestamp(dir_name):
        try:
            return datetime.strptime(dir_name, '%Y-%m-%d_%H-%M-%S')
        except ValueError:
            return datetime.min
    
    sorted_subdirs = sorted(subdirs, key=parse_timestamp, reverse=True)
    latest_subdir = sorted_subdirs[0]
    latest_checkpoint_path = os.path.join(checkpoint_dir, latest_subdir, 'best_model.pth')
    
    if not os.path.exists(latest_checkpoint_path):
        raise FileNotFoundError(f"best_model.pth not found in: {os.path.join(checkpoint_dir, latest_subdir)}")
    
    logging.info(f"Found latest checkpoint: {latest_checkpoint_path}")
    return latest_checkpoint_path

def create_dataloader(config, batch_size, num_workers):
    """创建数据加载器"""
    from torchvision import transforms
    
    # 创建图像转换
    transform_list = [
        transforms.Resize(tuple(config['transform']['resize'])),
        transforms.ToTensor()
    ]
    
    # 添加归一化（如果配置中存在）
    if 'normalize' in config['transform']:
        norm_config = config['transform']['normalize']
        transform_list.append(transforms.Normalize(mean=norm_config['mean'], std=norm_config['std']))
    
    transform = transforms.Compose(transform_list)
    
    # 创建数据集 - 默认加载所有split数据（train+test）
    dataset = ModelNet40NeighbourDataset(
        root_dir=config['dataset']['root_dir'],
        transform=transform,
        expected_images_per_view=config['dataset']['expected_images_per_view'],
        pointcloud_root=config.get('pointcloud', {}).get('root_dir')
    )
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 提取特征时不打乱顺序
        num_workers=num_workers,
        drop_last=False  # 不丢弃最后一个不完整批次
    )
    
    return dataloader

def extract_features(model, dataloader, device):
    """提取全视点特征（Teacher模式）"""
    model.eval()
    all_feats = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特征"):
            # 将数据移动到设备
            for v in batch['views']:
                batch['views'][v] = batch['views'][v].to(device)
            
            # 提取特征（Teacher模式）
            global_feat = model(batch, return_predictor=False)
            
            # 保存特征和元数据
            all_feats.append(global_feat.cpu())
            all_ids.extend(batch['object_id'])
    
    # 合并特征
    all_feats = torch.cat(all_feats, dim=0)
    # L2归一化
    all_feats = torch.nn.functional.normalize(all_feats, dim=1)
    
    return all_feats, all_ids

def compute_feature_stats(feats):
    """计算特征统计信息"""
    # 计算每个维度的均值和标准差
    mean_feats = feats.mean(dim=0)
    std_feats = feats.std(dim=0)
    
    # 计算平均均值和平均标准差
    avg_mean = mean_feats.mean().item()
    avg_std = std_feats.mean().item()
    
    # 计算1 - 标准差
    one_minus_std = 1.0 - std_feats.numpy()
    
    return mean_feats, std_feats, avg_mean, avg_std, one_minus_std

def plot_std_histogram(one_minus_std, save_path):
    """绘制1 - 标准差直方图"""
    plt.figure(figsize=(12, 6))
    
    # 绘制直方图
    plt.hist(one_minus_std, bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    # 设置图表属性
    plt.title('Distribution of 1 - Feature Standard Deviation', fontsize=16)
    plt.xlabel('1 - Standard Deviation', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"标准差直方图保存到: {save_path}")

def plot_similarity_heatmap(feats, save_path, num_samples=50):
    """绘制实例间相似度热力图"""
    # 随机抽取num_samples个实例
    if len(feats) > num_samples:
        indices = np.random.choice(len(feats), num_samples, replace=False)
        sample_feats = feats[indices]
    else:
        sample_feats = feats
    
    # 计算余弦相似度矩阵
    similarity_matrix = torch.matmul(sample_feats, sample_feats.T).numpy()
    
    plt.figure(figsize=(12, 10))
    
    # 绘制热力图
    sns.heatmap(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1, square=True)
    
    # 设置图表属性
    plt.title(f'Instance Similarity Heatmap (Top {len(sample_feats)} Samples)', fontsize=16)
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"相似度热力图保存到: {save_path}")
    
    # 计算非对角线相似度均值
    diagonal_mask = np.eye(len(similarity_matrix), dtype=bool)
    non_diagonal_similarity = similarity_matrix[~diagonal_mask]
    avg_non_diag_sim = non_diagonal_similarity.mean()
    
    return avg_non_diag_sim

def save_metrics(avg_mean, avg_std, avg_non_diag_sim, save_path):
    """保存指标到文件"""
    with open(save_path, 'w') as f:
        f.write('Feature Collapse Metrics\n')
        f.write('========================\n')
        f.write(f'Average feature mean: {avg_mean:.6f}\n')
        f.write(f'Average feature standard deviation: {avg_std:.6f}\n')
        f.write(f'Average non-diagonal similarity: {avg_non_diag_sim:.6f}\n')
        f.write('\n')
        f.write('Interpretation:\n')
        f.write('- If average std is close to 0, features are collapsed.\n')
        f.write('- If average non-diagonal similarity is close to 1, features are collapsed.\n')
        f.write('- For healthy features, average std should be significantly greater than 0,\n')
        f.write('  and average non-diagonal similarity should be close to 0.\n')
    
    logging.info(f"指标保存到: {save_path}")

def main():
    """主函数"""
    setup_logging()
    logging.info("开始特征塌陷验证")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='特征塌陷验证脚本')
    parser.add_argument('--config', type=str, default='../DataLoader/config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型权重路径（手动指定）')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader工作线程数')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备类型')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    logging.info(f"配置文件加载完成: {args.config}")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 初始化模型
    logging.info("初始化模型...")
    model = MultiViewVisualEncoder(feature_dim=1024)
    
    # 加载模型权重
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        logging.info(f"使用手动指定的checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = find_latest_checkpoint('../Checkpoints')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重（处理DataParallel情况）
    def load_state_dict(model, state_dict):
        if 'module.' in list(state_dict.keys())[0]:
            # 如果是DataParallel保存的模型，移除module前缀
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    
    if 'student_model_state_dict' in checkpoint:
        state_dict = checkpoint['student_model_state_dict']
    elif 'image_encoder_state_dict' in checkpoint:
        state_dict = checkpoint['image_encoder_state_dict']
    else:
        state_dict = checkpoint
        
    load_state_dict(model, state_dict)
    
    # 移动模型到设备
    model = model.to(device)
    logging.info("模型加载完成")
    
    # 创建数据加载器
    dataloader = create_dataloader(config, args.batch_size, args.num_workers)
    logging.info(f"数据加载器创建完成，总样本数: {len(dataloader.dataset)}")
    
    # 提取特征
    logging.info("提取全视点特征（Teacher模式）...")
    feats, ids = extract_features(model, dataloader, device)
    logging.info(f"特征提取完成，共 {len(ids)} 个样本")
    
    # 计算特征统计信息
    logging.info("计算特征统计信息...")
    mean_feats, std_feats, avg_mean, avg_std, one_minus_std = compute_feature_stats(feats)
    logging.info(f"平均特征均值: {avg_mean:.6f}")
    logging.info(f"平均特征标准差: {avg_std:.6f}")
    
    # 创建结果保存目录
    save_dir = '../Results/Validation_Plots'
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制并保存标准差直方图
    std_hist_path = os.path.join(save_dir, 'feature_std_histogram.png')
    plot_std_histogram(one_minus_std, std_hist_path)
    
    # 绘制并保存相似度热力图
    heatmap_path = os.path.join(save_dir, 'similarity_heatmap.png')
    avg_non_diag_sim = plot_similarity_heatmap(feats, heatmap_path)
    logging.info(f"平均非对角线相似度: {avg_non_diag_sim:.6f}")
    
    # 保存指标
    metrics_path = os.path.join(save_dir, 'collapse_metrics.txt')
    save_metrics(avg_mean, avg_std, avg_non_diag_sim, metrics_path)
    
    logging.info("特征塌陷验证完成")

if __name__ == '__main__':
    main()