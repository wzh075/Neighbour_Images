# =================== GPU 强制控制（必须最先执行） ===================
import os
import sys

# 检查 torch 是否已经被意外导入
if 'torch' in sys.modules:
    print("Warning: 'torch' module was imported before GPU setup! CUDA_VISIBLE_DEVICES might not work.")

# 手动解析参数，确保在 import torch 之前正确设置环境变量
gpu_ids = '0,1,2,3'
device_type = 'cuda'

print(f"Debug: sys.argv = {sys.argv}")

for i, arg in enumerate(sys.argv):
    if arg == '--gpus':
        if i + 1 < len(sys.argv):
            gpu_ids = sys.argv[i + 1]
    elif arg.startswith('--gpus='):
        gpu_ids = arg.split('=', 1)[1]
    elif arg == '--device':
        if i + 1 < len(sys.argv):
            device_type = sys.argv[i + 1]
    elif arg.startswith('--device='):
        device_type = arg.split('=', 1)[1]

if device_type == 'cuda':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    print(f"Info: Set CUDA_VISIBLE_DEVICES={gpu_ids}")
# ====================================================================

import argparse
import time
import yaml
import logging
import random
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataLoader.data_loader import ModelNet40NeighbourDataset, load_config
from Models.multi_view_visual_encoder import MultiViewVisualEncoder
from Loss_Function.instance_contrastive_loss import ViewDropoutContrastiveLoss


@torch.no_grad()
def update_ema_variables(student_model, teacher_model, momentum=0.996):
    """
    Update teacher model parameters using EMA from student model
    
    Args:
        student_model: Student model with learnable parameters
        teacher_model: Teacher model with EMA parameters
        momentum: EMA momentum factor
    """
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
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

    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

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

    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../DataLoader/config.yaml')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小（减小以避免显存不足）')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='../Log')
    parser.add_argument('--checkpoint_dir', type=str, default='../Checkpoints')
    parser.add_argument('--freeze_backbone', action='store_true', default=True, help='是否冻结 ResNet 骨干网络（默认冻结）')
    parser.add_argument('--ema_momentum', type=float, default=0.996, help='EMA momentum factor for teacher model')
    parser.add_argument('--lambda_distill', type=float, default=1.0, help='蒸馏损失的权重')
    args, _ = parser.parse_known_args()

    setup_seed(42)
    config = load_config(args.config)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dir, timestamp)
    writer = setup_logging(log_dir, config)

    logging.info(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")

    if device.type == 'cuda':
        logging.info(f"Actual torch.cuda.device_count() = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            try:
                p = torch.cuda.get_device_properties(i)
                logging.info(f"逻辑 cuda:{i} -> {torch.cuda.get_device_name(i)} | Memory: {p.total_memory / 1024**3:.2f} GB")
            except Exception as e:
                logging.error(f"Could not get properties for device {i}: {e}")

    dataloader = create_dataloader(config, args.batch_size, args.num_workers)

    # Create Student model
    student_model = MultiViewVisualEncoder(feature_dim=1024, freeze_backbone=args.freeze_backbone)

    # Create Teacher model
    teacher_model = MultiViewVisualEncoder(feature_dim=1024, freeze_backbone=args.freeze_backbone)
    
    # Initialize Teacher model with Student model's parameters
    for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_param.data.copy_(student_param.data)
        teacher_param.requires_grad = False  # Freeze Teacher model

    if device.type == 'cuda':
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            device_ids = list(range(gpu_count))
            # 显示物理 GPU IDs（通过 CUDA_VISIBLE_DEVICES 获取）
            physical_gpu_ids = gpu_ids.split(',') if isinstance(gpu_ids, str) else gpu_ids
            logging.info(f"DataParallel 使用逻辑 GPU IDs: {device_ids} (对应物理 GPU IDs: {physical_gpu_ids})")
            student_model = torch.nn.DataParallel(student_model, device_ids=device_ids)
            teacher_model = torch.nn.DataParallel(teacher_model, device_ids=device_ids)

    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)



    checkpoint_dir = os.path.join(args.checkpoint_dir, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 初始化优化器，只更新 Student 模型的参数
    raw_student = student_model.module if isinstance(student_model, torch.nn.DataParallel) else student_model
    
    # 设置参数梯度
    if args.freeze_backbone:
        for p in raw_student.backbone.parameters(): p.requires_grad = False
    for p in raw_student.projection.parameters(): p.requires_grad = True
    for p in raw_student.set_transformer.parameters(): p.requires_grad = True
    for p in raw_student.predictor.parameters(): p.requires_grad = True
    
    params = [p for p in raw_student.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    global_epoch = 0
    best_total_loss = float('inf')
    batch_counter = 0

    for epoch in range(args.epochs):
        global_epoch += 1
        epoch_total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {global_epoch}"):
            # 处理批次数据
            for v in batch['views']:
                batch['views'][v] = batch['views'][v].to(device)

            # ----------------------
            # 1. 准备 Teacher 输入（完整视点）
            # ----------------------
            teacher_batch = batch

            # ----------------------
            # 2. 准备 Student 输入（随机子集视点）
            # ----------------------
            # 获取视点总数
            num_views = len(batch['views'])
            if num_views > 1:
                # 随机选择 1 到 num_views-1 个视点
                K = torch.randint(1, num_views, (1,)).item()
                # 随机选择 K 个视点索引
                view_keys = list(batch['views'].keys())
                selected_keys = random.sample(view_keys, K)
                # 创建子集视点批次
                student_batch = {'views': {k: batch['views'][k] for k in selected_keys}}
            else:
                # 如果只有一个视点，使用完整视点
                student_batch = batch

            # ----------------------
            # 3. 前向传播
            # ----------------------
            # Teacher 模型（完整视点，不使用预测器）
            with torch.no_grad():
                teacher_feat = teacher_model(teacher_batch, return_predictor=False)

            # Student 模型（子集视点，使用预测器）
            student_feat, student_pred = student_model(student_batch, return_predictor=True)

            # ----------------------
            # 4. 计算蒸馏损失
            # ----------------------
            # L2 归一化
            student_pred_norm = torch.nn.functional.normalize(student_pred, dim=1)
            teacher_feat_norm = torch.nn.functional.normalize(teacher_feat.detach(), dim=1)
            
            # 基于余弦相似度的 BYOL 损失
            # 计算负的余弦相似度（因为我们要最小化损失）
            distill_loss = -torch.mean(torch.sum(student_pred_norm * teacher_feat_norm, dim=1))
            total_loss = args.lambda_distill * distill_loss

            # ----------------------
            # 5. 反向传播
            # ----------------------
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # ----------------------
            # 6. EMA 更新 Teacher 模型
            # ----------------------
            update_ema_variables(student_model, teacher_model, momentum=args.ema_momentum)

            # 累加损失
            epoch_total_loss += total_loss.item()

            if writer:
                # 记录总损失
                writer.add_scalar('Loss/total', total_loss.item(), batch_counter)
                # 记录蒸馏损失
                writer.add_scalar('Loss/distill', distill_loss.item(), batch_counter)
                # 记录学习率
                for i, param_group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'LearningRate/group_{i}', param_group['lr'], batch_counter)
            batch_counter += 1

        # 计算平均损失
        avg_total = epoch_total_loss / len(dataloader)
        
        # 打印损失
        logging.info(f"Epoch {global_epoch}: Total Loss={avg_total:.4f}")
        
        # 更新学习率
        scheduler.step(avg_total)

    def get_state_dict(m):
        return m.module.state_dict() if isinstance(m, torch.nn.DataParallel) else m.state_dict()

    torch.save({
        'epoch': global_epoch,
        'student_model_state_dict': get_state_dict(student_model),
        'teacher_model_state_dict': get_state_dict(teacher_model),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_total
    }, os.path.join(checkpoint_dir, 'last_model.pth'))

    if avg_total < best_total_loss:
        best_total_loss = avg_total
        torch.save(get_state_dict(student_model),
                   os.path.join(checkpoint_dir, 'best_model.pth'))

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
