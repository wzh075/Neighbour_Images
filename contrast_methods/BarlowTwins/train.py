import os
import sys
import argparse
import random
import logging
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from DataLoader.data_loader import ModelNet40NeighbourDataset, load_config
from contrast_methods.BarlowTwins.barlow_model import BarlowTwinsModel


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )


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
        pointcloud_root=None
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def _sort_view_keys(view_keys):
    def parse_view_index(k):
        if isinstance(k, str) and '_' in k:
            suffix = k.rsplit('_', 1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return k

    return sorted(view_keys, key=parse_view_index)


def sample_view_subset(views):
    view_keys = list(views.keys())
    num_views = len(view_keys)
    if num_views <= 1:
        return views
    k = random.randint(1, num_views)
    selected = random.sample(view_keys, k)
    return {k: views[k] for k in selected}


def views_dict_to_tensor(views):
    view_keys = _sort_view_keys(list(views.keys()))
    view_tensors = [views[k] for k in view_keys]
    x = torch.stack(view_tensors, dim=1)
    return x


def off_diagonal(x):
    n, m = x.size()
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(z1, z2, lambd=5e-3):
    if z1.size(0) == 1:
        return z1.new_tensor(0.0)
    z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-9)
    z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-9)
    b = z1.size(0)
    c = torch.matmul(z1.T, z2) / b
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(ROOT_DIR, 'DataLoader/config.yaml'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default=os.path.join(ROOT_DIR, 'Log/BarlowTwins'))
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(ROOT_DIR, 'Checkpoints/BarlowTwins'))
    parser.add_argument('--lambd', type=float, default=5e-3)
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(args.log_dir)

    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    dataloader = create_dataloader(config, args.batch_size, args.num_workers)

    model = BarlowTwinsModel()
    if device.type == 'cuda':
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            device_ids = list(range(gpu_count))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    checkpoint_dir = os.path.join(args.checkpoint_dir, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            for v in batch['views']:
                batch['views'][v] = batch['views'][v].to(device)

            views_1 = sample_view_subset(batch['views'])
            views_2 = sample_view_subset(batch['views'])

            x1 = views_dict_to_tensor(views_1)
            x2 = views_dict_to_tensor(views_2)

            z1 = model(x1, use_projection=True)
            z2 = model(x2, use_projection=True)

            loss = barlow_twins_loss(z1, z2, lambd=args.lambd)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(dataloader))
        logging.info(f"Epoch {epoch}: Loss={avg_loss:.4f}")

        def get_state_dict(m):
            return m.module.state_dict() if isinstance(m, torch.nn.DataParallel) else m.state_dict()

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': get_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            },
            os.path.join(checkpoint_dir, 'last_model.pth')
        )

    logging.info("Barlow Twins training finished")


if __name__ == '__main__':
    main()
