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
from contrast_methods.VICReg.vicreg_model import VICRegModel


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


def sample_view_subset(views):
    view_keys = list(views.keys())
    num_views = len(view_keys)
    if num_views <= 1:
        return views
    k = random.randint(1, num_views)
    selected = random.sample(view_keys, k)
    return {k: views[k] for k in selected}


def off_diagonal(x):
    n, m = x.size()
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1, z2, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, gamma=1.0):
    if z1.size(0) == 1:
        return z1.new_tensor(0.0)

    inv_loss = torch.mean((z1 - z2) ** 2)

    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)

    std_z1 = torch.sqrt(z1_centered.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2_centered.var(dim=0) + 1e-4)
    var_loss = torch.mean(torch.relu(gamma - std_z1)) + torch.mean(torch.relu(gamma - std_z2))

    b = z1.size(0)
    cov_z1 = (z1_centered.T @ z1_centered) / (b - 1)
    cov_z2 = (z2_centered.T @ z2_centered) / (b - 1)
    cov_loss = off_diagonal(cov_z1).pow(2).sum() / z1.size(1) + off_diagonal(cov_z2).pow(2).sum() / z2.size(1)

    return sim_coeff * inv_loss + var_coeff * var_loss + cov_coeff * cov_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(ROOT_DIR, 'DataLoader/config.yaml'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default=os.path.join(ROOT_DIR, 'Log/VICReg'))
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(ROOT_DIR, 'Checkpoints/VICReg'))
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(args.log_dir)

    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    dataloader = create_dataloader(config, args.batch_size, args.num_workers)

    model = VICRegModel()
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

            z1 = model({'views': views_1}, use_projection=True)
            z2 = model({'views': views_2}, use_projection=True)

            loss = vicreg_loss(z1, z2)

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

    logging.info("VICReg training finished")


if __name__ == '__main__':
    main()
