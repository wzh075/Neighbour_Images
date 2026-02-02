import os
import sys
import argparse
import logging
import torch
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


def compute_cosine_similarity(query_feats, gallery_feats, device, batch_size):
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
    for i in tqdm(range(0, num_queries, adjusted_batch_size), desc="计算相似度"):
        query_batch = query_feats[i:i+adjusted_batch_size]
        if query_batch.dim() != 2:
            query_batch = query_batch.view(-1, query_feats.shape[1])
        sim_batch = torch.matmul(query_batch, gallery_feats.t())
        similarity_matrix[i:i+adjusted_batch_size] = sim_batch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return similarity_matrix


def evaluate_recall(similarity_matrix, query_object_ids, gallery_object_ids, exclude_self=False):
    num_queries = similarity_matrix.shape[0]
    num_gallery = similarity_matrix.shape[1]
    list_gallery_length = len(gallery_object_ids)
    if num_gallery != list_gallery_length:
        num_gallery = min(num_gallery, list_gallery_length)
    top_k = 10
    _, indices = similarity_matrix.topk(top_k, dim=1, largest=True, sorted=True)
    indices = indices.cpu().numpy()
    max_valid_idx = list_gallery_length - 1
    indices = np.where(indices > max_valid_idx, -1, indices)
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    for i in tqdm(range(num_queries), desc="计算Recall"):
        query_obj_id = query_object_ids[i]
        retrieved_obj_ids = []
        for idx in indices[i]:
            if idx < 0 or idx >= list_gallery_length:
                continue
            retrieved_obj_ids.append(gallery_object_ids[idx])
        if exclude_self:
            if i in indices[i]:
                new_retrieved_obj_ids = []
                for idx, obj_id in zip(indices[i], retrieved_obj_ids):
                    if idx != i:
                        new_retrieved_obj_ids.append(obj_id)
                retrieved_obj_ids = new_retrieved_obj_ids
        if query_obj_id in retrieved_obj_ids[:1]:
            recall_1 += 1
        if query_obj_id in retrieved_obj_ids[:5]:
            recall_5 += 1
        if query_obj_id in retrieved_obj_ids[:10]:
            recall_10 += 1
    recall_1 /= num_queries
    recall_5 /= num_queries
    recall_10 /= num_queries
    return recall_1, recall_5, recall_10


def run_eval(args):
    setup_logging()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint('../Checkpoints')
    image_encoder = load_models(device, checkpoint_path)
    dataloader = create_dataloader(config, args.batch_size, args.num_workers)
    
    # 生成 Gallery 特征
    gallery_feats = []
    gallery_ids = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="生成 Gallery 特征"):
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
            
            # 提取所有视点的真实特征
            view_tensors = [batch['views'][view_name] for view_name in view_keys]
            x = torch.stack(view_tensors, dim=0).permute(1, 0, 2, 3, 4, 5)
            B, N, NUM_NEIGHBOURS, C, H, W = x.size()
            x = x.reshape(B * N * NUM_NEIGHBOURS, C, H, W)
            backbone_feats = image_encoder.backbone(x)
            backbone_feats = backbone_feats.view(backbone_feats.size(0), -1)
            backbone_feats = backbone_feats.view(B, N, NUM_NEIGHBOURS, -1)
            backbone_feats = image_encoder.projection(backbone_feats)
            real_view_feats = backbone_feats.max(dim=2)[0]  # (B, N, D)
            
            # 使用所有真实视点特征生成全局特征
            _, global_feat = image_encoder.set_transformer(real_view_feats)
            global_feat = global_feat.squeeze(1)  # (B, D)
            gallery_feats.append(global_feat.cpu())
            gallery_ids.extend(batch['object_id'])
    
    # 拼接 Gallery 特征
    gallery_feats = torch.cat(gallery_feats, dim=0)
    gallery_feats = gallery_feats / gallery_feats.norm(dim=1, keepdim=True)
    results = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="推理"):
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
            
            # Step A: 提取所有视点的真实特征
            # 首先通过完整前向传播获取真实的view_feats
            # 注意：我们需要修改模型调用方式，获取中间特征
            # 这里我们先获取原始的view_feats
            # 模拟模型前向传播中的view_feats计算过程
            view_tensors = [batch['views'][view_name] for view_name in view_keys]
            x = torch.stack(view_tensors, dim=0).permute(1, 0, 2, 3, 4, 5)
            B, N, NUM_NEIGHBOURS, C, H, W = x.size()
            x = x.reshape(B * N * NUM_NEIGHBOURS, C, H, W)
            backbone_feats = image_encoder.backbone(x)
            backbone_feats = backbone_feats.view(backbone_feats.size(0), -1)
            backbone_feats = backbone_feats.view(B, N, NUM_NEIGHBOURS, -1)
            backbone_feats = image_encoder.projection(backbone_feats)
            real_view_feats = backbone_feats.max(dim=2)[0]  # (B, N, D)
            
            # Step B: 遍历每一个视点作为"源 (Source)"
            for source_idx, source_view in enumerate(view_keys):
                # 构建幻觉集合 (Hallucinated Set)
                hallucinated_view_feats = []
                
                # 遍历每一个目标位置
                for target_idx in range(num_views):
                    if target_idx == source_idx:
                        # 如果是源视点，使用真实特征
                        hallucinated_view_feats.append(real_view_feats[:, target_idx, :].unsqueeze(1))
                    else:
                        # 如果是目标视点，使用生成器生成特征
                        # 获取源视点特征
                        source_feat = real_view_feats[:, source_idx, :]
                        # 获取目标视点的嵌入
                        target_pose_emb = image_encoder.view_embedding(torch.tensor([target_idx], device=device).repeat(B))
                        # 生成目标视点特征
                        generated_feat = image_encoder.view_generator(source_feat, target_pose_emb)
                        hallucinated_view_feats.append(generated_feat.unsqueeze(1))
                
                # 拼接幻觉集合
                hallucinated_view_feats = torch.cat(hallucinated_view_feats, dim=1)  # (B, N, D)
                
                # Step C: 使用Set Transformer聚合特征
                # 调用set_transformer获取全局特征
                _, aggregated_feat = image_encoder.set_transformer(hallucinated_view_feats)
                global_image_feat = aggregated_feat.squeeze(1)  # (B, D)
                
                # 保存结果
                if source_idx not in results:
                    results[source_idx] = {'img_feats': [], 'ids': []}
                results[source_idx]['img_feats'].append(global_image_feat.cpu())
                results[source_idx]['ids'].extend(batch['object_id'])
    
    # 计算评估结果
    report = []
    for idx in sorted(results.keys()):
        img_feats = torch.cat(results[idx]['img_feats'], dim=0)
        ids = results[idx]['ids']
        sim = compute_cosine_similarity(img_feats, gallery_feats, device, args.eval_batch_size)
        r1, r5, r10 = evaluate_recall(sim, ids, gallery_ids)
        report.append((idx, r1, r5, r10))
    
    # 输出结果
    print("\n=== 单视点输入鲁棒性评估 (Single-to-Global Hallucination) ===")
    print(f"{'输入源视点':<12} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    for idx, r1, r5, r10 in report:
        print(f"View {idx:<6} {r1*100:>7.2f}% {r5*100:>7.2f}% {r10*100:>7.2f}%")
    if report:
        mean_r1 = np.mean([r[1] for r in report])
        mean_r5 = np.mean([r[2] for r in report])
        mean_r10 = np.mean([r[3] for r in report])
        print(f"{'Mean':<12} {mean_r1*100:>7.2f}% {mean_r5*100:>7.2f}% {mean_r10*100:>7.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description='单视点输入鲁棒性评估')
    parser.add_argument('--config', type=str, default='../DataLoader/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_batch_size', type=int, default=128)
    return parser.parse_args()


if __name__ == '__main__':
    import numpy as np
    args = parse_args()
    run_eval(args)

