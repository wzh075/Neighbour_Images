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
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 提取特征时不打乱顺序
        num_workers=num_workers,
        drop_last=False  # 不丢弃最后一个不完整批次
    )
    
    return dataloader

def extract_gallery_features(model, dataloader, device):
    """提取Gallery特征（全视点）"""
    model.eval()
    gallery_feats = []
    gallery_ids = []
    gallery_categories = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取Gallery特征"):
            # 将数据移动到设备
            for v in batch['views']:
                batch['views'][v] = batch['views'][v].to(device)
            
            # 提取特征（Teacher模式）
            global_feat = model(batch, return_predictor=False)
            
            # 保存特征和元数据
            gallery_feats.append(global_feat.cpu())
            gallery_ids.extend(batch['object_id'])
            gallery_categories.extend(batch['category'])
    
    # 合并特征
    gallery_feats = torch.cat(gallery_feats, dim=0)
    # L2归一化
    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1)
    
    return gallery_feats, gallery_ids, gallery_categories

def extract_query_features(model, dataloader, device, K):
    """提取Query特征（随机K个视点）"""
    model.eval()
    query_feats = []
    query_ids = []
    query_categories = []
    
    import random
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"提取Query特征 (K={K})"):
            # 将数据移动到设备
            for v in batch['views']:
                batch['views'][v] = batch['views'][v].to(device)
            
            # 随机选择K个视点
            num_views = len(batch['views'])
            if num_views > K:
                view_keys = list(batch['views'].keys())
                selected_keys = random.sample(view_keys, K)
                # 创建子集视点批次
                subset_batch = {'views': {k: batch['views'][k] for k in selected_keys}}
            else:
                subset_batch = batch
            
            # 提取特征（Student模式）
            _, pred_feat = model(subset_batch, return_predictor=True)
            
            # 保存特征和元数据
            query_feats.append(pred_feat.cpu())
            query_ids.extend(batch['object_id'])
            query_categories.extend(batch['category'])
    
    # 合并特征
    query_feats = torch.cat(query_feats, dim=0)
    # L2归一化
    query_feats = torch.nn.functional.normalize(query_feats, dim=1)
    
    return query_feats, query_ids, query_categories

def compute_retrieval_accuracy(query_feats, query_ids, gallery_feats, gallery_ids):
    """计算检索准确率"""
    # 计算余弦相似度
    similarity_matrix = torch.matmul(query_feats, gallery_feats.T)
    
    # 获取Top-K索引
    top1_indices = similarity_matrix.topk(1, dim=1)[1].squeeze(1)
    top5_indices = similarity_matrix.topk(5, dim=1)[1]
    
    # 计算准确率
    correct_top1 = 0
    correct_top5 = 0
    
    for i, query_id in enumerate(query_ids):
        # Top-1 准确率
        if gallery_ids[top1_indices[i]] == query_id:
            correct_top1 += 1
        
        # Top-5 准确率
        for idx in top5_indices[i]:
            if gallery_ids[idx] == query_id:
                correct_top5 += 1
                break
    
    top1_acc = correct_top1 / len(query_ids)
    top5_acc = correct_top5 / len(query_ids)
    
    return top1_acc, top5_acc

def plot_retrieval_curve(K_values, top1_accs, top5_accs, save_dir):
    """绘制检索准确率曲线"""
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

def save_results(K_values, top1_accs, top5_accs, save_dir):
    """保存结果到文件"""
    # 保存为CSV文件
    csv_path = os.path.join(save_dir, 'results.csv')
    with open(csv_path, 'w') as f:
        f.write('K,Top-1 Accuracy,Top-5 Accuracy\n')
        for K, top1, top5 in zip(K_values, top1_accs, top5_accs):
            f.write(f'{K},{top1:.4f},{top5:.4f}\n')
    
    # 保存为文本文件
    txt_path = os.path.join(save_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write('Retrieval Accuracy Results\n')
        f.write('============================\n')
        f.write('K\tTop-1 Accuracy\tTop-5 Accuracy\n')
        for K, top1, top5 in zip(K_values, top1_accs, top5_accs):
            f.write(f'{K}\t{top1:.4f}\t\t{top5:.4f}\n')
    
    logging.info(f"Results saved to: {csv_path} and {txt_path}")

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
    
    # 提取Gallery特征
    logging.info("提取Gallery特征（全视点）...")
    gallery_feats, gallery_ids, gallery_categories = extract_gallery_features(model, dataloader, device)
    logging.info(f"Gallery特征提取完成，共 {len(gallery_ids)} 个样本")
    
    # 确定最大视点数
    sample_batch = next(iter(dataloader))
    max_K = len(sample_batch['views'])
    logging.info(f"最大视点数 N = {max_K}")
    
    # 初始化结果列表
    K_values = list(range(1, max_K + 1))
    top1_accs = []
    top5_accs = []
    
    # 循环测试不同K值
    for K in K_values:
        logging.info(f"\n测试 K = {K} 视点...")
        
        # 提取Query特征
        query_feats, query_ids, query_categories = extract_query_features(model, dataloader, device, K)
        
        # 计算检索准确率
        top1_acc, top5_acc = compute_retrieval_accuracy(query_feats, query_ids, gallery_feats, gallery_ids)
        
        # 记录结果
        top1_accs.append(top1_acc)
        top5_accs.append(top5_acc)
        
        logging.info(f"K={K}: Top-1 Accuracy = {top1_acc:.4f}, Top-5 Accuracy = {top5_acc:.4f}")
    
    # 创建结果保存目录
    save_dir = '../Results/Retrieval_Curves'
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制并保存检索曲线
    plot_retrieval_curve(K_values, top1_accs, top5_accs, save_dir)
    
    # 保存结果
    save_results(K_values, top1_accs, top5_accs, save_dir)
    
    logging.info("BYOL检索评估完成")

if __name__ == '__main__':
    main()