import os
import sys
import argparse
import logging
import copy
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataLoader.data_loader import load_config
from Models.multi_view_visual_encoder import MultiViewVisualEncoder
from Models.pointcloud_encoder import PointCloudEncoder
from Main.extract import find_latest_checkpoint, create_dataloader


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_models(device, checkpoint_path):
    image_encoder = MultiViewVisualEncoder(feature_dim=1024)
    pointcloud_encoder = PointCloudEncoder(feature_dim=1024)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    def load_state_dict(model, state_dict):
        first_key = list(state_dict.keys())[0]
        if first_key.startswith('module.'):
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

    load_state_dict(image_encoder, checkpoint['image_encoder_state_dict'])
    load_state_dict(pointcloud_encoder, checkpoint['pointcloud_encoder_state_dict'])

    image_encoder = image_encoder.to(device).eval()
    pointcloud_encoder = pointcloud_encoder.to(device).eval()
    return image_encoder, pointcloud_encoder


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
    image_encoder, pointcloud_encoder = load_models(device, checkpoint_path)
    dataloader = create_dataloader(config, args.batch_size, args.num_workers)
    results = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="推理"):
            if batch['pointcloud'] is None:
                continue
            for view_name in batch['views']:
                batch['views'][view_name] = batch['views'][view_name].to(device)
            batch['pointcloud'] = batch['pointcloud'].to(device)
            _, global_point_feat = pointcloud_encoder(batch)
            view_keys = list(batch['views'].keys())
            try:
                view_keys = sorted(view_keys, key=lambda x: int(x.split('_')[-1]))
            except Exception:
                view_keys = sorted(view_keys)
            for idx, source_view in enumerate(view_keys):
                modified_batch = copy.deepcopy(batch)
                source_tensor = batch['views'][source_view]
                modified_views = {k: source_tensor for k in view_keys}
                modified_batch['views'] = modified_views
                _, global_image_feat = image_encoder(modified_batch)
                if idx not in results:
                    results[idx] = {'img_feats': [], 'pc_feats': [], 'ids': []}
                results[idx]['img_feats'].append(global_image_feat.cpu())
                results[idx]['pc_feats'].append(global_point_feat.cpu())
                results[idx]['ids'].extend(batch['object_id'])
    report = []
    for idx in sorted(results.keys()):
        img_feats = torch.cat(results[idx]['img_feats'], dim=0)
        pc_feats = torch.cat(results[idx]['pc_feats'], dim=0)
        ids = results[idx]['ids']
        sim = compute_cosine_similarity(img_feats, pc_feats, device, args.eval_batch_size)
        r1, r5, r10 = evaluate_recall(sim, ids, ids)
        report.append((idx, r1, r5, r10))
    print("\n=== 单视点输入鲁棒性评估 ===")
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
    args = parse_args()
    run_eval(args)

