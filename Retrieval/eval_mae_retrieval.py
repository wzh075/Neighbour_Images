import os
import sys
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataLoader.data_loader import ModelNet40NeighbourDataset, load_config
from Models.multi_view_visual_encoder import MultiViewVisualEncoder


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def _sort_view_keys(view_keys):
    def parse_view_index(k):
        if isinstance(k, str) and '_' in k:
            suffix = k.rsplit('_', 1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return k

    return sorted(view_keys, key=parse_view_index)


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

    best_encoder = os.path.join(latest_dir, 'best_encoder.pth')
    last_model = os.path.join(latest_dir, 'last_model.pth')
    if os.path.exists(best_encoder):
        logging.info(f"Found latest checkpoint: {best_encoder}")
        return best_encoder
    if os.path.exists(last_model):
        logging.info(f"Found latest checkpoint: {last_model}")
        return last_model

    raise FileNotFoundError(f"No checkpoint file found in: {latest_dir}")


def create_dataloader(config, batch_size, num_workers):
    from torchvision import transforms

    transform_list = [
        transforms.Resize(tuple(config['transform']['resize'])),
        transforms.ToTensor()
    ]
    if 'normalize' in config['transform']:
        norm_config = config['transform']['normalize']
        transform_list.append(transforms.Normalize(mean=norm_config['mean'], std=norm_config['std']))
    transform = transforms.Compose(transform_list)

    dataset = ModelNet40NeighbourDataset(
        root_dir=config['dataset']['root_dir'],
        transform=transform,
        expected_images_per_view=config['dataset']['expected_images_per_view'],
        pointcloud_root=config.get('pointcloud', {}).get('root_dir')
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    return dataloader


def load_state_dict(model, state_dict):
    if isinstance(state_dict, dict) and 'encoder_state_dict' in state_dict:
        state_dict = state_dict['encoder_state_dict']

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

            latent_full, _, _, _ = model(batch, mask_ratio=0.0)
            gallery_feat = latent_full.mean(dim=1)
            gallery_feat = torch.nn.functional.normalize(gallery_feat, p=2, dim=1)
            gallery_feats_all.append(gallery_feat.cpu())

            view_keys = _sort_view_keys(list(batch['views'].keys()))
            for K in K_list:
                sub_keys = view_keys[:K]
                sub_batch = {'views': {k: batch['views'][k] for k in sub_keys}}
                latent_k, _, _, _ = model(sub_batch, mask_ratio=0.0)
                query_feat = latent_k.mean(dim=1)
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


def save_results(results_log, save_dir):
    txt_path = os.path.join(save_dir, 'results.txt')
    with open(txt_path, 'w') as f:
        f.write('K,Top-1 Accuracy,Top-5 Accuracy\n')
        for K, top1_acc, top5_acc in results_log:
            f.write(f'{K},{top1_acc:.4f},{top5_acc:.4f}\n')

    csv_path = os.path.join(save_dir, 'results.csv')
    with open(csv_path, 'w') as f:
        f.write('K,Top-1 Accuracy,Top-5 Accuracy\n')
        for K, top1_acc, top5_acc in results_log:
            f.write(f'{K},{top1_acc:.4f},{top5_acc:.4f}\n')

    logging.info(f"Results saved to: {txt_path} and {csv_path}")


def plot_retrieval_curve(results_log, save_dir):
    K_values = [item[0] for item in results_log]
    top1_accs = [item[1] for item in results_log]
    top5_accs = [item[2] for item in results_log]

    plt.figure(figsize=(10, 6))
    plt.plot(K_values, top1_accs, 'o-', label='Top-1 Accuracy', linewidth=2, markersize=6)
    plt.plot(K_values, top5_accs, 's-', label='Top-5 Accuracy', linewidth=2, markersize=6)
    plt.title('Retrieval Accuracy vs Number of Input Views (K)', fontsize=16)
    plt.xlabel('Number of Input Views (K)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'accuracy_vs_k.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Retrieval curve saved to: {save_path}")


def main():
    setup_logging()
    logging.info("开始MAE检索评估")

    parser = argparse.ArgumentParser(description='MAE检索评估脚本')
    parser.add_argument('--config', type=str, default='../DataLoader/config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型权重路径（手动指定）')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader工作线程数')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备类型')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    mae_cfg = config.get('mae', {})
    feature_dim = int(mae_cfg.get('feature_dim', 1024))
    max_num_views = int(mae_cfg.get('max_num_views', 64))
    encoder_depth = int(mae_cfg.get('encoder_depth', 2))

    model = MultiViewVisualEncoder(
        feature_dim=feature_dim,
        max_num_views=max_num_views,
        encoder_depth=encoder_depth,
        freeze_backbone=True,
    )

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint('../Checkpoints')

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint
    load_state_dict(model, state_dict)

    model = model.to(device)
    dataloader = create_dataloader(config, args.batch_size, args.num_workers)

    save_dir = '../Results/Retrieval_Curves'
    os.makedirs(save_dir, exist_ok=True)

    gallery_feats, query_feats_dict, K_list = extract_features(model, dataloader, device)
    results_log = compute_retrieval_accuracy(gallery_feats, query_feats_dict, K_list, device)
    save_results(results_log, save_dir)
    plot_retrieval_curve(results_log, save_dir)

    logging.info("MAE检索评估完成")


if __name__ == '__main__':
    main()
