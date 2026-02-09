import os
import sys
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataLoader.data_loader import load_config
from Models.multi_view_visual_encoder import MultiViewVisualEncoder
from Main.extract import find_latest_checkpoint, create_dataloader


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_models(device, checkpoint_path):
    image_encoder = MultiViewVisualEncoder(feature_dim=1024)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    def load_state_dict(model, state_dict):
        first_key = list(state_dict.keys())[0]
        if first_key.startswith('module.'):
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

    if 'image_encoder_state_dict' in checkpoint:
        load_state_dict(image_encoder, checkpoint['image_encoder_state_dict'])
    else:
        # 兼容只保存了模型 state_dict 而没有保存完整 checkpoint 的情况
        load_state_dict(image_encoder, checkpoint)

    image_encoder = image_encoder.to(device).eval()
    return image_encoder


def compute_metrics(fake_feats, real_feats):
    """
    计算生成特征与真实特征之间的指标
    """
    # 计算余弦相似度
    fake_feats_norm = fake_feats / fake_feats.norm(dim=1, keepdim=True)
    real_feats_norm = real_feats / real_feats.norm(dim=1, keepdim=True)
    cosine_sim = (fake_feats_norm * real_feats_norm).sum(dim=1).mean().item()
    
    # 计算 L2 距离
    l2_dist = torch.norm(fake_feats - real_feats, dim=1).mean().item()
    
    return cosine_sim, l2_dist


def compute_cosine_similarity(feats1, feats2, device):
    """
    计算两个特征集合之间的余弦相似度矩阵
    """
    feats1 = feats1.to(device)
    feats2 = feats2.to(device)
    feats1 = feats1 / feats1.norm(dim=1, keepdim=True)
    feats2 = feats2 / feats2.norm(dim=1, keepdim=True)
    similarity_matrix = torch.matmul(feats1, feats2.t())
    return similarity_matrix


def analyze_instance_distinctness(args):
    setup_logging()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint('../Checkpoints')
    
    logging.info(f"Loading model from: {checkpoint_path}")
    model = load_models(device, checkpoint_path)
    
    # 加载测试集
    dataloader = create_dataloader(config, args.batch_size, args.num_workers)
    logging.info(f"Test dataset loaded with {len(dataloader.dataset)} samples")
    
    # 初始化特征和标签存储
    view_feats_all = []
    global_feats_all = []
    object_ids_all = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="提取特征")):
            # 将所有视点图像移动到设备
            for view_name in batch['views']:
                batch['views'][view_name] = batch['views'][view_name].to(device)
            
            # 获取排序后的视点名称
            view_keys = list(batch['views'].keys())
            try:
                view_keys = sorted(view_keys, key=lambda x: int(x.split('_')[-1]))
            except Exception:
                view_keys = sorted(view_keys)
            num_views = len(view_keys)
            
            # 提取所有真实特征
            view_tensors = [batch['views'][view_name] for view_name in view_keys]
            x = torch.stack(view_tensors, dim=0).permute(1, 0, 2, 3, 4, 5)
            B, N, NUM_NEIGHBOURS, C, H, W = x.size()
            x = x.reshape(B * N * NUM_NEIGHBOURS, C, H, W)
            
            # 提取骨干网络特征
            backbone_feats = model.backbone(x)
            backbone_feats = backbone_feats.view(backbone_feats.size(0), -1)
            backbone_feats = backbone_feats.view(B, N, NUM_NEIGHBOURS, -1)
            backbone_feats = model.projection(backbone_feats)
            
            # 邻域图融合，得到单个视点的特征
            view_feats = backbone_feats.max(dim=2)[0]  # (B, N, D)
            
            # 提取全局特征
            _, global_feat = model.set_transformer(view_feats)
            global_feat = global_feat.squeeze(1)  # (B, D)
            
            # 存储特征和标签
            view_feats_all.append(view_feats.cpu())
            global_feats_all.append(global_feat.cpu())
            object_ids_all.extend(batch['object_id'])
    
    # 拼接所有特征
    view_feats_all = torch.cat(view_feats_all, dim=0)
    global_feats_all = torch.cat(global_feats_all, dim=0)
    
    # 分析实例间区分度
    logging.info("分析实例间区分度...")
    
    # 1. 分析全局特征的实例间区分度
    logging.info("\n=== 全局特征实例间区分度分析 ===")
    analyze_feats_distinctness(global_feats_all, object_ids_all, "Global Features")
    
    # 2. 分析视点级特征的实例间区分度
    logging.info("\n=== 视点级特征实例间区分度分析 ===")
    # 对每个视点分别分析
    num_views = view_feats_all.shape[1]
    for view_idx in range(num_views):
        view_feats = view_feats_all[:, view_idx, :]
        logging.info(f"\n--- 视点 {view_idx} 特征分析 ---")
        analyze_feats_distinctness(view_feats, object_ids_all, f"View {view_idx} Features")


def analyze_feats_distinctness(feats, object_ids, feature_name):
    """
    分析特征的实例间区分度
    """
    num_samples = feats.shape[0]
    
    # 计算余弦相似度矩阵
    similarity_matrix = compute_cosine_similarity(feats, feats, device='cpu')
    
    # 提取同一实例和不同实例的相似度
    same_instance_sims = []
    different_instance_sims = []
    
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            if object_ids[i] == object_ids[j]:
                same_instance_sims.append(similarity_matrix[i, j].item())
            else:
                different_instance_sims.append(similarity_matrix[i, j].item())
    
    # 计算统计指标
    same_mean = np.mean(same_instance_sims) if same_instance_sims else 0
    same_std = np.std(same_instance_sims) if same_instance_sims else 0
    different_mean = np.mean(different_instance_sims) if different_instance_sims else 0
    different_std = np.std(different_instance_sims) if different_instance_sims else 0
    
    # 计算区分度指标
    # 1. 相似度差距
    similarity_gap = same_mean - different_mean
    # 2. 标准化互信息 (Normalized Mutual Information) 类似的指标
    # 3. 计算对比损失
    contrastive_loss = compute_contrastive_loss(similarity_matrix, object_ids)
    
    # 打印结果
    print(f"特征类型: {feature_name}")
    print(f"同一实例相似度: 均值 = {same_mean:.4f}, 标准差 = {same_std:.4f}")
    print(f"不同实例相似度: 均值 = {different_mean:.4f}, 标准差 = {different_std:.4f}")
    print(f"相似度差距: {similarity_gap:.4f}")
    print(f"对比损失: {contrastive_loss:.4f}")
    
    # 可视化相似度分布
    if args.visualize:
        visualize_similarity_distribution(
            same_instance_sims, 
            different_instance_sims, 
            feature_name
        )
    
    # 计算检索性能
    recall_at_1, recall_at_5, recall_at_10 = compute_retrieval_performance(
        similarity_matrix, object_ids
    )
    print(f"检索性能: R@1 = {recall_at_1:.4f}, R@5 = {recall_at_5:.4f}, R@10 = {recall_at_10:.4f}")


def compute_contrastive_loss(similarity_matrix, object_ids):
    """
    计算对比损失
    """
    num_samples = similarity_matrix.shape[0]
    loss = 0.0
    margin = 0.2
    
    for i in range(num_samples):
        same_instance = []
        different_instance = []
        
        for j in range(num_samples):
            if i == j:
                continue
            if object_ids[i] == object_ids[j]:
                same_instance.append(similarity_matrix[i, j])
            else:
                different_instance.append(similarity_matrix[i, j])
        
        if same_instance and different_instance:
            # 同一实例的相似度应尽可能高，不同实例的相似度应尽可能低
            same_loss = torch.mean(1.0 - torch.tensor(same_instance))
            different_loss = torch.mean(torch.maximum(
                torch.tensor(0.0),
                torch.tensor(margin) + torch.tensor(different_instance)
            ))
            loss += same_loss + different_loss
    
    return loss.item() / num_samples if num_samples > 0 else 0.0


def compute_retrieval_performance(similarity_matrix, object_ids):
    """
    计算检索性能 (Recall@k)
    """
    num_samples = similarity_matrix.shape[0]
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    
    for i in range(num_samples):
        # 获取除了自身之外的所有相似度
        sims = similarity_matrix[i].clone()
        sims[i] = -float('inf')  # 排除自身
        
        # 获取top-k索引
        top10_indices = torch.argsort(sims, descending=True)[:10]
        
        # 检查是否包含同一实例
        query_id = object_ids[i]
        retrieved_ids = [object_ids[idx] for idx in top10_indices]
        
        if query_id in retrieved_ids[:1]:
            recall_at_1 += 1
        if query_id in retrieved_ids[:5]:
            recall_at_5 += 1
        if query_id in retrieved_ids[:10]:
            recall_at_10 += 1
    
    return (
        recall_at_1 / num_samples,
        recall_at_5 / num_samples,
        recall_at_10 / num_samples
    )

def visualize_similarity_distribution(same_sims, different_sims, feature_name):
    """
    可视化同一实例和不同实例的相似度分布，并保存到本地
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(same_sims, bins=50, alpha=0.6, label='Same Instance', color='green')
    sns.histplot(different_sims, bins=50, alpha=0.6, label='Different Instance', color='red')
    plt.title(f'Similarity Distribution - {feature_name}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 保存图像到本地
    output_dir = '../Validation/results'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{feature_name.replace(" ", "_")}_similarity_distribution.png')
    plt.savefig(save_path)
    plt.close()
    print(f"可视化结果已保存到: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='验证实例间区分度')
    parser.add_argument('--config', type=str, default='../DataLoader/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--visualize', action='store_true', default=True, help='是否可视化结果并保存到本地')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    analyze_instance_distinctness(args)
