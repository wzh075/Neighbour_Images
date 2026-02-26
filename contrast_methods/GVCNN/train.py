import os
import sys
import argparse
import random
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(ROOT_DIR)

from DataLoader.data_loader import ModelNet40NeighbourDataset, load_config
from contrast_methods.GVCNN.gvcnn_model import GVCNNModel


@torch.no_grad()
def update_ema_variables(student_model, teacher_model, momentum=0.996):
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir, config):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(log_dir, 'tensorboard'))
    except ImportError:
        logging.warning("TensorBoard not available")
    return writer


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(ROOT_DIR, 'DataLoader/config.yaml'))
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default=os.path.join(ROOT_DIR, 'Log/GVCNN'))
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(ROOT_DIR, 'Checkpoints/GVCNN'))
    parser.add_argument('--freeze_backbone', action='store_true', default=True)
    parser.add_argument('--ema_momentum', type=float, default=0.996)
    parser.add_argument('--lambda_distill', type=float, default=1.0)
    args = parser.parse_args()

    setup_seed(42)
    config = load_config(args.config)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dir, timestamp)
    writer = setup_logging(log_dir, config)

    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    dataloader = create_dataloader(config, args.batch_size, args.num_workers)

    student_model = GVCNNModel()
    teacher_model = GVCNNModel()
    predictor = nn.Sequential(
        nn.Linear(1024, 512),
        nn.LayerNorm(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 1024),
    )

    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.copy_(student_param.data)
        teacher_param.requires_grad = False

    if device.type == 'cuda':
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            device_ids = list(range(gpu_count))
            student_model = torch.nn.DataParallel(student_model, device_ids=device_ids)
            teacher_model = torch.nn.DataParallel(teacher_model, device_ids=device_ids)
            predictor = torch.nn.DataParallel(predictor, device_ids=device_ids)

    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    predictor = predictor.to(device)

    checkpoint_dir = os.path.join(args.checkpoint_dir, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    raw_student = student_model.module if isinstance(student_model, torch.nn.DataParallel) else student_model
    raw_predictor = predictor.module if isinstance(predictor, torch.nn.DataParallel) else predictor

    if args.freeze_backbone:
        for p in raw_student.backbone.parameters():
            p.requires_grad = False
    for p in raw_student.projector.parameters():
        p.requires_grad = True
    for p in raw_student.group_weight.parameters():
        p.requires_grad = True
    for p in raw_predictor.parameters():
        p.requires_grad = True

    params = [p for p in list(raw_student.parameters()) + list(raw_predictor.parameters()) if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    global_epoch = 0
    best_total_loss = float('inf')
    batch_counter = 0

    for epoch in range(args.epochs):
        global_epoch += 1
        epoch_total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {global_epoch}"):
            for v in batch['views']:
                batch['views'][v] = batch['views'][v].to(device)

            teacher_batch = batch
            num_views = len(batch['views'])
            if num_views > 1:
                k = random.randint(1, num_views)
                view_keys = list(batch['views'].keys())
                selected_keys = random.sample(view_keys, k)
                student_batch = {'views': {k: batch['views'][k] for k in selected_keys}}
            else:
                student_batch = batch

            x_teacher = views_dict_to_tensor(teacher_batch['views'])
            x_student = views_dict_to_tensor(student_batch['views'])

            with torch.no_grad():
                _, teacher_feat = teacher_model(x_teacher, return_projector=True)

            _, student_proj = student_model(x_student, return_projector=True)
            student_pred = predictor(student_proj)

            student_pred_norm = torch.nn.functional.normalize(student_pred, dim=1)
            teacher_feat_norm = torch.nn.functional.normalize(teacher_feat.detach(), dim=1)
            distill_loss = -torch.mean(torch.sum(student_pred_norm * teacher_feat_norm, dim=1))
            total_loss = args.lambda_distill * distill_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            update_ema_variables(student_model, teacher_model, momentum=args.ema_momentum)

            epoch_total_loss += total_loss.item()

            if writer:
                writer.add_scalar('Loss/total', total_loss.item(), batch_counter)
                writer.add_scalar('Loss/distill', distill_loss.item(), batch_counter)
                for i, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'LearningRate/group_{i}', param_group['lr'], batch_counter)
            batch_counter += 1

        avg_total = epoch_total_loss / len(dataloader)
        logging.info(f"Epoch {global_epoch}: Total Loss={avg_total:.4f}")
        scheduler.step(avg_total)

    def get_state_dict(m):
        return m.module.state_dict() if isinstance(m, torch.nn.DataParallel) else m.state_dict()

    torch.save({
        'epoch': global_epoch,
        'student_model_state_dict': get_state_dict(student_model),
        'teacher_model_state_dict': get_state_dict(teacher_model),
        'predictor_state_dict': get_state_dict(predictor),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_total
    }, os.path.join(checkpoint_dir, 'last_model.pth'))

    if avg_total < best_total_loss:
        best_total_loss = avg_total
        torch.save(get_state_dict(student_model), os.path.join(checkpoint_dir, 'best_model.pth'))

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
