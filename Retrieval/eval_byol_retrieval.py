import os
import sys
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
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
    
    # 创建数据加载器 - 必须设置shuffle=False和drop_last=False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 必须设置为False，确保索引对齐
        num_workers=num_workers,
        drop_last=False  # 必须设置为False，不丢弃最后一个不完整批次
    )
    
    return dataloader

def extract_features(model, dataloader, device):
    """在同一个Batch循环中提取Teacher和所有K值的Student特征"""
    model.eval()
    
    # 确定最大视点数
    sample_batch = next(iter(dataloader))
    num_views = len(sample_batch['views'])
    K_list = list(range(1, num_views + 1))
    logging.info(f"视点数范围: K = 1 到 {num_views}")
    
    # 初始化特征存储
    teacher_feats_all = []
    student_feats_dict = {k: [] for k in K_list}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            # 将数据移动到设备
            for v in batch['views']:
                batch['views'][v] = batch['views'][v].to(device)
            
            # --- 1. 提取 Teacher (Gallery) 特征 ---
            t_feat = model(batch, return_predictor=False)
            # 强制 L2 归一化
            t_feat = torch.nn.functional.normalize(t_feat, p=2, dim=1)
            teacher_feats_all.append(t_feat.cpu())
            
            # --- 2. 在同一个 Batch 下，提取不同 K 值的 Student (Query) 特征 ---
            view_keys = list(batch['views'].keys())
            for K in K_list:
                # 截取前 K 个视点
                sub_keys = view_keys[:K]
                sub_batch = {'views': {k: batch['views'][k] for k in sub_keys}}
                
                s_pred = model(sub_batch, return_predictor=False)
                # 强制 L2 归一化
                s_pred = torch.nn.functional.normalize(s_pred, p=2, dim=1)
                student_feats_dict[K].append(s_pred.cpu())
    
    # 拼接张量
    teacher_feats_tensor = torch.cat(teacher_feats_all, dim=0).to(device)
    for K in K_list:
        student_feats_dict[K] = torch.cat(student_feats_dict[K], dim=0).to(device)
    
    return teacher_feats_tensor, student_feats_dict, K_list

def compute_retrieval_accuracy(teacher_feats, student_feats_dict, K_list, device):
    """计算检索准确率"""
    results_log = []
    num_samples = teacher_feats.size(0)
    labels = torch.arange(num_samples).view(-1, 1).to(device)
    
    for K in K_list:
        query_feats = student_feats_dict[K]
        
        # 计算余弦相似度矩阵
        sim_matrix = torch.matmul(query_feats, teacher_feats.T)
        
        # 获取 Top-5 索引
        _, top5_idx = sim_matrix.topk(5, dim=1, largest=True, sorted=True)
        
        # 计算 Top-1 和 Top-5 准确率
        top1_acc = (top5_idx[:, 0:1] == labels).float().mean().item()
        top5_acc = (top5_idx == labels).any(dim=1).float().mean().item()
        
        results_log.append((K, top1_acc, top5_acc))
        logging.info(f"K={K}: Top-1 Accuracy = {top1_acc:.4f}, Top-5 Accuracy = {top5_acc:.4f}")
    
    return results_log

def save_results(results_log, save_dir):
    """保存结果到文件"""
    # 保存为文本文件
    txt_path = os.path.join(save_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write('K,Top-1 Accuracy,Top-5 Accuracy\n')
        for K, top1_acc, top5_acc in results_log:
            f.write(f'{K},{top1_acc:.4f},{top5_acc:.4f}\n')
    
    # 保存为CSV文件
    csv_path = os.path.join(save_dir, 'results.csv')
    with open(csv_path, 'w') as f:
        f.write('K,Top-1 Accuracy,Top-5 Accuracy\n')
        for K, top1_acc, top5_acc in results_log:
            f.write(f'{K},{top1_acc:.4f},{top5_acc:.4f}\n')
    
    logging.info(f"Results saved to: {txt_path} and {csv_path}")

def plot_retrieval_curve(results_log, save_dir):
    """绘制检索准确率曲线"""
    # 提取数据
    K_values = [item[0] for item in results_log]
    top1_accs = [item[1] for item in results_log]
    top5_accs = [item[2] for item in results_log]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制Top-1准确率曲线
    plt.plot(K_values, top1_accs, 'o-', label='Top-1 Accuracy', linewidth=2, markersize=6)
    
    # 绘制Top-5准确率曲线
    plt.plot(K_values, top5_accs, 's-', label='Top-5 Accuracy', linewidth=2, markersize=6)
    
    # 设置图表属性
    plt.title('Retrieval Accuracy vs Number of Input Views (K)', fontsize=16)
    plt.xlabel('Number of Input Views (K)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, 'accuracy_vs_k.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Retrieval curve saved to: {save_path}")

def main():
    """主函数"""
    setup_logging()
    logging.info("开始BYOL检索评估")
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='BYOL检索评估脚本')
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
    
    # 创建结果保存目录
    save_dir = '../Results/Retrieval_Curves'
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"结果保存目录: {save_dir}")
    
    # 提取特征（在同一个Batch循环中）
    logging.info("提取特征（在同一个Batch循环中）...")
    teacher_feats, student_feats_dict, K_list = extract_features(model, dataloader, device)
    logging.info(f"特征提取完成，Teacher特征形状: {teacher_feats.shape}")
    for K in K_list:
        logging.info(f"K={K}时Student特征形状: {student_feats_dict[K].shape}")
    
    # 计算检索准确率
    logging.info("计算检索准确率...")
    results_log = compute_retrieval_accuracy(teacher_feats, student_feats_dict, K_list, device)
    
    # 保存结果
    logging.info("保存结果...")
    save_results(results_log, save_dir)
    
    # 绘制并保存检索曲线
    logging.info("绘制检索曲线...")
    plot_retrieval_curve(results_log, save_dir)
    
    logging.info("BYOL检索评估完成")

if __name__ == '__main__':
    main()