import os
import sys
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm

import random

# 设置随机种子以确保结果可重现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

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


def compute_cosine_similarity(query_feats, gallery_feats, device, batch_size):
    """
    计算查询特征与图库特征之间的余弦相似度
    """
    query_feats = query_feats.to(device)
    gallery_feats = gallery_feats.to(device)
    query_feats = query_feats / query_feats.norm(dim=1, keepdim=True)
    gallery_feats = gallery_feats / gallery_feats.norm(dim=1, keepdim=True)
    num_queries = query_feats.shape[0]
    num_gallery = gallery_feats.shape[0]
    similarity_matrix = torch.zeros((num_queries, num_gallery), device=device)
    adjusted_batch_size = batch_size
    if device.type == 'cuda':
        try:
            estimated_memory_per_batch = (query_feats.element_size() * query_feats.shape[1] * num_gallery * 2) / (1024 ** 3)
            available_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device)
            available_memory_gb = available_memory / (1024 ** 3)
            if estimated_memory_per_batch > available_memory_gb * 0.8:
                adjusted_batch_size = max(1, int(batch_size * (available_memory_gb * 0.8 / estimated_memory_per_batch)))
        except Exception:
            adjusted_batch_size = batch_size
    for i in range(0, num_queries, adjusted_batch_size):
        query_batch = query_feats[i:i+adjusted_batch_size]
        if query_batch.dim() != 2:
            query_batch = query_batch.view(-1, query_feats.shape[1])
        sim_batch = torch.matmul(query_batch, gallery_feats.t())
        similarity_matrix[i:i+adjusted_batch_size] = sim_batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return similarity_matrix


def evaluate_recall(similarity_matrix, query_object_ids, query_categories, gallery_object_ids, gallery_categories, exclude_self=False):
    """
    评估检索性能，计算 Recall@1, Recall@5, Recall@10
    并返回前10检索结果的最高和最低余弦相似度
    新增：计算类别检索精度
    """
    num_queries = similarity_matrix.shape[0]
    num_gallery = similarity_matrix.shape[1]
    list_gallery_length = len(gallery_object_ids)
    if num_gallery != list_gallery_length:
        num_gallery = min(num_gallery, list_gallery_length)
    top_k = 10
    topk_values, indices = similarity_matrix.topk(top_k, dim=1, largest=True, sorted=True)
    indices = indices.cpu().numpy()
    topk_values = topk_values.cpu().numpy()
    max_valid_idx = list_gallery_length - 1
    indices = np.where(indices > max_valid_idx, -1, indices)
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    total_instance_hits = 0
    total_class_hits = 0
    max_similarities = []
    min_similarities = []
    for i in range(num_queries):
        query_obj_id = query_object_ids[i]
        query_category = query_categories[i] if i < len(query_categories) else None
        retrieved_obj_ids = []
        retrieved_categories = []
        valid_similarities = []
        for j, idx in enumerate(indices[i]):
            if idx < 0 or idx >= list_gallery_length:
                continue
            retrieved_obj_ids.append(gallery_object_ids[idx])
            if idx < len(gallery_categories):
                retrieved_categories.append(gallery_categories[idx])
            valid_similarities.append(topk_values[i][j])
        if exclude_self:
            if i in indices[i]:
                new_retrieved_obj_ids = []
                new_retrieved_categories = []
                new_valid_similarities = []
                for idx, obj_id, cat, sim in zip(indices[i], retrieved_obj_ids, retrieved_categories, valid_similarities):
                    if idx != i:
                        new_retrieved_obj_ids.append(obj_id)
                        new_retrieved_categories.append(cat)
                        new_valid_similarities.append(sim)
                retrieved_obj_ids = new_retrieved_obj_ids
                retrieved_categories = new_retrieved_categories
                valid_similarities = new_valid_similarities
        if query_obj_id in retrieved_obj_ids[:1]:
            recall_1 += 1
            total_instance_hits += 1
        if query_obj_id in retrieved_obj_ids[:5]:
            recall_5 += 1
        if query_obj_id in retrieved_obj_ids[:10]:
            recall_10 += 1
        if query_category and retrieved_categories:
            if query_category in retrieved_categories[:1]:
                total_class_hits += 1
        if valid_similarities:
            max_similarities.append(max(valid_similarities))
            min_similarities.append(min(valid_similarities))
    recall_1 /= num_queries
    recall_5 /= num_queries
    recall_10 /= num_queries
    mean_max_sim = np.mean(max_similarities) if max_similarities else 0.0
    mean_min_sim = np.mean(min_similarities) if min_similarities else 0.0
    instance_acc = total_instance_hits / num_queries
    class_acc = total_class_hits / num_queries
    return recall_1, recall_5, recall_10, mean_max_sim, mean_min_sim, instance_acc, class_acc


def run_evaluation(args):
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
    
    # 初始化指标累积器
    input_count_stats = {}
    for k in args.input_counts:
        input_count_stats[k] = {
            'gen_cosine_sim': 0.0,
            'gen_l2_dist': 0.0,
            'gen_pairs': 0,
            'query_feats': [],
            'query_ids': [],
            'query_categories': []
        }
    
    # 收集 Gallery 特征和 ID
    gallery_feats = []
    gallery_ids = []
    gallery_categories = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="评估多视点质量")):
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
            
            # Step A: 提取所有真实特征
            # 模拟模型前向传播中的view_feats计算过程
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
            all_real_feats = backbone_feats.max(dim=2)[0]  # (B, N, D)
            
            # 提取全视点真实全局特征
            _, real_global_feat = model.set_transformer(all_real_feats)
            real_global_feat = real_global_feat.squeeze(1)  # (B, D)
            
            # 收集 Gallery 特征、ID 和类别
            gallery_feats.append(real_global_feat.cpu())
            gallery_ids.extend(batch['object_id'])
            if 'category' in batch:
                gallery_categories.extend(batch['category'])
            elif 'label' in batch:
                gallery_categories.extend(batch['label'])
            
            # Step B: 循环测试不同的输入数量
            for k in args.input_counts:
                if k >= num_views:
                    continue
                
                # 随机选择输入视点
                input_indices = random.sample(range(num_views), k)
                target_indices = [i for i in range(num_views) if i not in input_indices]
                
                # Step C: 多源幻觉生成与聚合
                final_view_feats = []
                gen_cosine_sum = 0.0
                gen_l2_sum = 0.0
                gen_count = 0
                
                # 填充真实特征
                for i in range(num_views):
                    if i in input_indices:
                        final_view_feats.append(all_real_feats[:, i, :].unsqueeze(1))
                
                # 填充/生成缺失特征
                for t_idx in target_indices:
                    preds = []
                    # 多源生成
                    for s_idx in input_indices:
                        # 取源特征
                        source_feat = all_real_feats[:, s_idx, :]
                        # 取目标位姿嵌入
                        target_pose_emb = model.view_embedding(torch.tensor([t_idx], device=device).repeat(B))
                        # 生成伪特征
                        fake_feat = model.view_generator(source_feat, target_pose_emb)
                        preds.append(fake_feat)
                    # 聚合策略
                    agg_feat = torch.mean(torch.stack(preds), dim=0)
                    final_view_feats.append(agg_feat.unsqueeze(1))
                    
                    # 生成质量统计
                    real_target_feat = all_real_feats[:, t_idx, :]
                    cosine_sim, l2_dist = compute_metrics(agg_feat, real_target_feat)
                    gen_cosine_sum += cosine_sim * B
                    gen_l2_sum += l2_dist * B
                    gen_count += B
                
                # 拼接最终的视点特征集合
                final_view_feats = torch.cat(final_view_feats, dim=1)  # (B, N, D)
                
                # Step D: 全局特征提取与检索
                _, pseudo_global_feat = model.set_transformer(final_view_feats)
                pseudo_global_feat = pseudo_global_feat.squeeze(1)  # (B, D)
                
                # 累积生成质量指标
                input_count_stats[k]['gen_cosine_sim'] += gen_cosine_sum
                input_count_stats[k]['gen_l2_dist'] += gen_l2_sum
                input_count_stats[k]['gen_pairs'] += gen_count
                
                # 收集 Query 特征、ID 和类别
                input_count_stats[k]['query_feats'].append(pseudo_global_feat.cpu())
                input_count_stats[k]['query_ids'].extend(batch['object_id'])
                if 'category' in batch:
                    input_count_stats[k]['query_categories'].extend(batch['category'])
                elif 'label' in batch:
                    input_count_stats[k]['query_categories'].extend(batch['label'])
    
    # 拼接 Gallery 特征
    gallery_feats = torch.cat(gallery_feats, dim=0)
    
    # 执行检索测试并打印结果
    for k in args.input_counts:
        stats = input_count_stats[k]
        if not stats['query_feats']:
            continue
        
        # 计算生成质量平均指标
        mean_gen_cosine = stats['gen_cosine_sim'] / stats['gen_pairs'] if stats['gen_pairs'] > 0 else 0.0
        mean_gen_l2 = stats['gen_l2_dist'] / stats['gen_pairs'] if stats['gen_pairs'] > 0 else 0.0
        
        # 拼接 Query 特征
        query_feats = torch.cat(stats['query_feats'], dim=0)
        query_ids = stats['query_ids']
        query_categories = stats['query_categories']
        
        # 计算相似度矩阵
        similarity_matrix = compute_cosine_similarity(query_feats, gallery_feats, device, args.batch_size)
        
        # 计算 Recall 和相似度统计
        r1, r5, r10, mean_max_sim, mean_min_sim, instance_acc, class_acc = evaluate_recall(
            similarity_matrix, query_ids, query_categories, gallery_ids, gallery_categories
        )
        
        # 打印结果
        print(f"\n--- Results for K={k} Input Views ---")
        print(f"Avg Gen Cosine Sim: {mean_gen_cosine:.4f}")
        print(f"Avg Gen L2 Dist: {mean_gen_l2:.4f}")
        print(f"Retrieval Instance Accuracy: {instance_acc:.2%}")
        print(f"Retrieval Class Accuracy: {class_acc:.2%}")
        print(f"R@1: {r1:.2%}, R@5: {r5:.2%}, R@10: {r10:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(description='多视点质量验证')
    parser.add_argument('--config', type=str, default='../DataLoader/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input_counts', type=int, nargs='+', default=[1, 2, 3], help='要测试的输入视点数量')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_evaluation(args)
