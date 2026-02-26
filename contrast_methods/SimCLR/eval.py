import os
import sys
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from DataLoader.data_loader import ModelNet40NeighbourDataset, load_config
from contrast_methods.SimCLR.simclr_model import SimCLRModel


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint subdirectories found in: {checkpoint_dir}")
    from datetime import datetime

    def parse_timestamp(dir_name):
        try:
            return datetime.strptime(dir_name, '%Y-%m-%d_%H-%M-%S')
        except ValueError:
            return datetime.min

    sorted_subdirs = sorted(subdirs, key=parse_timestamp, reverse=True)
    latest_dir = os.path.join(checkpoint_dir, sorted_subdirs[0])
    last_model = os.path.join(latest_dir, 'last_model.pth')
    if os.path.exists(last_model):
        return last_model
    raise FileNotFoundError(f"last_model.pth not found in: {latest_dir}")


def create_dataloader(config, batch_size, num_workers):
    transform_list = [
        transforms.Resize(tuple(config['transform']['resize'])),
        transforms.ToTensor()
    ]
    if 'normalize' in config['transform']:
        norm = config['transform']['normalize']
        transform_list.append(transforms.Normalize(mean=norm['mean'], std=norm['std']))

    dataset = ModelNet40NeighbourDataset(
        root_dir=config['dataset']['root_dir'],
        transform=transforms.Compose(transform_list),
        expected_images_per_view=config['dataset']['expected_images_per_view'],
        pointcloud_root=config.get('pointcloud', {}).get('root_dir')
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


def _sort_view_keys(view_keys):
    def parse_view_index(k):
        if isinstance(k, str) and '_' in k:
            suffix = k.rsplit('_', 1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return k

    return sorted(view_keys, key=parse_view_index)


def views_dict_to_tensor(views):
    view_keys = _sort_view_keys(list(views.keys()))
    view_tensors = [views[k] for k in view_keys]
    x = torch.stack(view_tensors, dim=1)
    return x


def load_state_dict(model, checkpoint):
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.info(f"Missing keys: {missing}")
    if unexpected:
        logging.info(f"Unexpected keys: {unexpected}")


def extract_features(model, dataloader, device):
    model.eval()

    sample_batch = next(iter(dataloader))
    num_views = len(sample_batch['views'])
    K_list = list(range(1, num_views + 1))
    logging.info(f"视点数范围: K = 1 到 {num_views}")

    gallery_feats_all = []
    query_feats_dict = {k: [] for k in K_list}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            for v in batch['views']:
                batch['views'][v] = batch['views'][v].to(device)

            x_full = views_dict_to_tensor(batch['views'])
            gallery_feat = model(x_full, use_projection=False)
            gallery_feat = torch.nn.functional.normalize(gallery_feat, p=2, dim=1)
            gallery_feats_all.append(gallery_feat.cpu())

            view_keys = _sort_view_keys(list(batch['views'].keys()))
            for K in K_list:
                sub_keys = view_keys[:K]
                sub_views = {k: batch['views'][k] for k in sub_keys}
                x_sub = views_dict_to_tensor(sub_views)
                query_feat = model(x_sub, use_projection=False)
                query_feat = torch.nn.functional.normalize(query_feat, p=2, dim=1)
                query_feats_dict[K].append(query_feat.cpu())

    gallery_feats_tensor = torch.cat(gallery_feats_all, dim=0).to(device)
    for K in K_list:
        query_feats_dict[K] = torch.cat(query_feats_dict[K], dim=0).to(device)

    return gallery_feats_tensor, query_feats_dict, K_list


def compute_retrieval_accuracy(gallery_feats, query_feats_dict, K_list, device):
    results_log = []
    num_samples = gallery_feats.size(0)
    labels = torch.arange(num_samples).view(-1, 1).to(device)

    for K in K_list:
        query_feats = query_feats_dict[K]
        sim_matrix = torch.matmul(query_feats, gallery_feats.T)
        _, top5_idx = sim_matrix.topk(5, dim=1, largest=True, sorted=True)
        top1_acc = (top5_idx[:, 0:1] == labels).float().mean().item()
        top5_acc = (top5_idx == labels).any(dim=1).float().mean().item()
        results_log.append((K, top1_acc, top5_acc))
        logging.info(f"K={K}: Top-1 Accuracy = {top1_acc:.4f}, Top-5 Accuracy = {top5_acc:.4f}")

    return results_log


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='SimCLR检索评估脚本')
    parser.add_argument('--config', type=str, default=os.path.join(ROOT_DIR, 'DataLoader/config.yaml'))
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = SimCLRModel()

    checkpoint_path = args.checkpoint or find_latest_checkpoint(os.path.join(ROOT_DIR, 'Checkpoints/SimCLR'))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_state_dict(model, checkpoint)

    model = model.to(device)
    dataloader = create_dataloader(config, args.batch_size, args.num_workers)

    gallery_feats, query_feats_dict, K_list = extract_features(model, dataloader, device)
    compute_retrieval_accuracy(gallery_feats, query_feats_dict, K_list, device)


if __name__ == '__main__':
    main()
