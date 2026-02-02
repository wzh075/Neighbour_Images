import os
import sys
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm

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


def evaluate_recall(similarity_matrix, query_object_ids, gallery_object_ids, exclude_self=False):
    """
    评估检索性能，计算 Recall@1, Recall@5, Recall@10
    并返回前10检索结果的最高和最低余弦相似度
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
    max_similarities = []
    min_similarities = []
    for i in range(num_queries):
        query_obj_id = query_object_ids[i]
        retrieved_obj_ids = []
        valid_similarities = []
        for j, idx in enumerate(indices[i]):
            if idx < 0 or idx >= list_gallery_length:
                continue
            retrieved_obj_ids.append(gallery_object_ids[idx])
            valid_similarities.append(topk_values[i][j])
        if exclude_self:
            if i in indices[i]:
                new_retrieved_obj_ids = []
                new_valid_similarities = []
                for idx, obj_id, sim in zip(indices[i], retrieved_obj_ids, valid_similarities):
                    if idx != i:
                        new_retrieved_obj_ids.append(obj_id)
                        new_valid_similarities.append(sim)
                retrieved_obj_ids = new_retrieved_obj_ids
                valid_similarities = new_valid_similarities
        if query_obj_id in retrieved_obj_ids[:1]:
            recall_1 += 1
        if query_obj_id in retrieved_obj_ids[:5]:
            recall_5 += 1
        if query_obj_id in retrieved_obj_ids[:10]:
            recall_10 += 1
        if valid_similarities:
            max_similarities.append(max(valid_similarities))
            min_similarities.append(min(valid_similarities))
    recall_1 /= num_queries
    recall_5 /= num_queries
    recall_10 /= num_queries
    mean_max_sim = np.mean(max_similarities) if max_similarities else 0.0
    mean_min_sim = np.mean(min_similarities) if min_similarities else 0.0
    return recall_1, recall_5, recall_10, mean_max_sim, mean_min_sim


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
    total_cosine_sim = 0.0
    total_l2_dist = 0.0
    total_pairs = 0
    
    # 新增：单视点全局特征与全视点全局特征比较指标
    total_global_cosine_sim = 0.0
    total_global_l2_dist = 0.0
    total_global_pairs = 0
    
    # 可选：按视点跨度统计
    span_stats = {}
    # 新增：按源视点统计全局特征相似度
    source_view_stats = {}
    
    # 新增：收集 Gallery 和 Query 特征用于检索测试
    gallery_feats = []
    gallery_ids = []
    query_feats = {}
    query_ids = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="评估生成器质量")):
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
            real_feats = backbone_feats.max(dim=2)[0]  # (B, N, D)
            
            # 新增：提取全视点真实全局特征
            # 使用 Set Transformer 聚合所有真实视点特征
            _, real_global_feat = model.set_transformer(real_feats)
            real_global_feat = real_global_feat.squeeze(1)  # (B, D)
            
            # 新增：收集 Gallery 特征和 ID
            gallery_feats.append(real_global_feat.cpu())
            gallery_ids.extend(batch['object_id'])
            
            # Step B: 遍历生成对 (Source -> Target Pairs)
            for source_idx in range(num_views):
                # 新增：生成单视点全局特征
                # 构建幻觉集合 (Hallucinated Set)
                hallucinated_view_feats = []
                for target_idx in range(num_views):
                    if target_idx == source_idx:
                        # 如果是源视点，使用真实特征
                        hallucinated_view_feats.append(real_feats[:, target_idx, :].unsqueeze(1))
                    else:
                        # 如果是目标视点，使用生成器生成特征
                        # 取源特征
                        source_feat = real_feats[:, source_idx, :]
                        # 取目标位姿嵌入
                        target_pose_emb = model.view_embedding(torch.tensor([target_idx], device=device).repeat(B))
                        # 生成伪特征
                        fake_feat = model.view_generator(source_feat, target_pose_emb)
                        hallucinated_view_feats.append(fake_feat.unsqueeze(1))
                
                # 拼接幻觉集合
                hallucinated_view_feats = torch.cat(hallucinated_view_feats, dim=1)  # (B, N, D)
                
                # 使用 Set Transformer 聚合特征
                _, hallucinated_global_feat = model.set_transformer(hallucinated_view_feats)
                hallucinated_global_feat = hallucinated_global_feat.squeeze(1)  # (B, D)
                
                # 新增：收集 Query 特征和 ID
                if source_idx not in query_feats:
                    query_feats[source_idx] = []
                    query_ids[source_idx] = []
                query_feats[source_idx].append(hallucinated_global_feat.cpu())
                query_ids[source_idx].extend(batch['object_id'])
                
                # 新增：计算单视点全局特征与全视点真实全局特征的相似度
                global_cosine_sim, global_l2_dist = compute_metrics(hallucinated_global_feat, real_global_feat)
                
                # 累积全局特征相似度指标
                total_global_cosine_sim += global_cosine_sim * B
                total_global_l2_dist += global_l2_dist * B
                total_global_pairs += B
                
                # 按源视点统计全局特征相似度
                if source_idx not in source_view_stats:
                    source_view_stats[source_idx] = {'cosine': 0.0, 'l2': 0.0, 'count': 0}
                source_view_stats[source_idx]['cosine'] += global_cosine_sim * B
                source_view_stats[source_idx]['l2'] += global_l2_dist * B
                source_view_stats[source_idx]['count'] += B
                
                # 原有：遍历目标视点，生成伪特征并与真实目标特征比较
                for target_idx in range(num_views):
                    if source_idx == target_idx:
                        continue
                    
                    # 取源特征
                    source_feat = real_feats[:, source_idx, :]
                    
                    # 取目标位姿嵌入
                    target_pose_emb = model.view_embedding(torch.tensor([target_idx], device=device).repeat(B))
                    
                    # 生成伪特征
                    fake_feat = model.view_generator(source_feat, target_pose_emb)
                    
                    # 取真实目标特征
                    real_target_feat = real_feats[:, target_idx, :]
                    
                    # Step C: 计算指标
                    cosine_sim, l2_dist = compute_metrics(fake_feat, real_target_feat)
                    
                    # 累积指标
                    total_cosine_sim += cosine_sim * B
                    total_l2_dist += l2_dist * B
                    total_pairs += B
                    
                    # 可选：按视点跨度统计
                    span = abs(source_idx - target_idx)
                    if span not in span_stats:
                        span_stats[span] = {'cosine': 0.0, 'l2': 0.0, 'count': 0}
                    span_stats[span]['cosine'] += cosine_sim * B
                    span_stats[span]['l2'] += l2_dist * B
                    span_stats[span]['count'] += B
    
    # 计算平均指标
    mean_cosine_sim = total_cosine_sim / total_pairs
    mean_l2_dist = total_l2_dist / total_pairs
    
    # 计算全局特征相似度平均指标
    mean_global_cosine_sim = total_global_cosine_sim / total_global_pairs
    mean_global_l2_dist = total_global_l2_dist / total_global_pairs
    
    # 新增：执行检索测试
    retrieval_results = []
    if gallery_feats and query_feats:
        # 拼接 Gallery 特征
        gallery_feats = torch.cat(gallery_feats, dim=0)
        
        # 对每个源视点执行检索测试
        for source_idx in sorted(query_feats.keys()):
            # 拼接 Query 特征
            source_query_feats = torch.cat(query_feats[source_idx], dim=0)
            source_query_ids = query_ids[source_idx]
            
            # 计算相似度矩阵
            similarity_matrix = compute_cosine_similarity(source_query_feats, gallery_feats, device, args.batch_size)
            
            # 计算 Recall 和相似度统计
            r1, r5, r10, mean_max_sim, mean_min_sim = evaluate_recall(similarity_matrix, source_query_ids, gallery_ids)
            retrieval_results.append((source_idx, r1, r5, r10, mean_max_sim, mean_min_sim))
    
    # 打印结果
    print("\n=== 视点生成器质量评估结果 ===")
    print(f"平均余弦相似度: {mean_cosine_sim:.4f}")
    print(f"平均 L2 距离: {mean_l2_dist:.4f}")
    print(f"总测试样本对: {total_pairs}")
    
    # 打印按视点跨度统计的结果
    if span_stats:
        print("\n=== 按视点跨度统计 ===")
        print(f"{'跨度':<6} {'余弦相似度':<12} {'L2 距离':<10}")
        print("-" * 30)
        for span in sorted(span_stats.keys()):
            stats = span_stats[span]
            span_cosine = stats['cosine'] / stats['count']
            span_l2 = stats['l2'] / stats['count']
            print(f"{span:<6} {span_cosine:.4f}        {span_l2:.4f}")
    
    # 新增：打印单视点全局特征与全视点真实全局特征的相似度结果
    print("\n=== 单视点全局特征验证结果 ===")
    print(f"平均余弦相似度: {mean_global_cosine_sim:.4f}")
    print(f"平均 L2 距离: {mean_global_l2_dist:.4f}")
    print(f"总测试样本对: {total_global_pairs}")
    
    # 打印按源视点统计的全局特征相似度结果
    if source_view_stats:
        print("\n=== 按源视点统计全局特征相似度 ===")
        print(f"{'源视点':<6} {'余弦相似度':<12} {'L2 距离':<10}")
        print("-" * 30)
        for source_idx in sorted(source_view_stats.keys()):
            stats = source_view_stats[source_idx]
            source_cosine = stats['cosine'] / stats['count']
            source_l2 = stats['l2'] / stats['count']
            print(f"{source_idx:<6} {source_cosine:.4f}        {source_l2:.4f}")
    
    # 新增：打印检索测试结果
    if retrieval_results:
        print("\n=== 单视点检索全局特征测试结果 ===")
        print(f"{'源视点':<6} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'最高相似度':<10} {'最低相似度':<10}")
        print("-" * 60)
        for source_idx, r1, r5, r10, max_sim, min_sim in retrieval_results:
            print(f"{source_idx:<6} {r1*100:>7.2f}% {r5*100:>7.2f}% {r10*100:>7.2f}% {max_sim:>9.4f}     {min_sim:>9.4f}")
        
        # 计算平均 Recall 和相似度统计
        if retrieval_results:
            mean_r1 = np.mean([r[1] for r in retrieval_results])
            mean_r5 = np.mean([r[2] for r in retrieval_results])
            mean_r10 = np.mean([r[3] for r in retrieval_results])
            mean_max_sim = np.mean([r[4] for r in retrieval_results])
            mean_min_sim = np.mean([r[5] for r in retrieval_results])
            print(f"{'平均':<6} {mean_r1*100:>7.2f}% {mean_r5*100:>7.2f}% {mean_r10*100:>7.2f}% {mean_max_sim:>9.4f}     {mean_min_sim:>9.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='视点生成器质量验证')
    parser.add_argument('--config', type=str, default='../DataLoader/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_evaluation(args)
