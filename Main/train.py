import os
import sys
import time
import yaml
import logging
import argparse
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

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from DataLoader.data_loader import ModelNet40NeighbourDataset, load_config
from Models.multi_view_visual_encoder import MultiViewVisualEncoder
from Models.pointcloud_encoder import PointCloudEncoder
from Loss_Function.instance_contrastive_loss import InstanceDualContrastiveLoss

def setup_seed(seed):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir, config):
    """建立完善的日志系统"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建文本日志
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 尝试创建TensorBoard日志，如果失败则返回None
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_dir = os.path.join(log_dir, 'tensorboard')
        writer = SummaryWriter(tensorboard_dir)
    except ImportError:
        logging.warning("TensorBoard not available, skipping TensorBoard logging")
    
    # 保存配置信息
    config_file = os.path.join(log_dir, 'config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return writer

def create_dataloader(config, batch_size, num_workers):
    """创建数据加载器"""
    # 创建图像转换
    transform_list = [
        transforms.Resize(tuple(config['transform']['resize'])),
        transforms.ToTensor()
    ]
    
    # 添加归一化（如果配置中存在）
    if 'normalize' in config['transform']:
        norm_config = config['transform']['normalize']
        transform_list.append(transforms.Normalize(mean=norm_config['mean'], std=norm_config['std']))
    
    transform = transforms.Compose(transform_list)
    
    # 创建数据集
    dataset = ModelNet40NeighbourDataset(
        root_dir=config['dataset']['root_dir'],
        transform=transform,
        expected_images_per_view=config['dataset']['expected_images_per_view'],
        pointcloud_root=config.get('pointcloud', {}).get('root_dir')
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    return dataloader

def main():
    """主训练函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Unsupervised Cross-Modal Contrastive Learning Training Script')
    parser.add_argument('--config', type=str, default='../DataLoader/config.yaml', help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备类型')
    parser.add_argument('--gpus', type=str, default='1,2,3', help='要使用的显卡ID列表，用逗号分隔，如"0,1"')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader工作线程数')
    parser.add_argument('--log_dir', type=str, default='../Log', help='日志保存根目录')
    parser.add_argument('--checkpoint_dir', type=str, default='../Checkpoints', help='模型保存根目录')
    parser.add_argument('--freeze_backbone', action='store_true', default=True, help='是否冻结ResNet Backbone参数')
    parser.add_argument('--lambda_gen', type=float, default=1.0, help='生成损失权重')
    args = parser.parse_args()
    
    # 设置随机种子
    setup_seed(42)
    
    # 加载配置文件
    config = load_config(args.config)
    
    # 创建时间戳
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # 建立日志系统
    log_dir = os.path.join(args.log_dir, timestamp)
    writer = setup_logging(log_dir, config)
    logging.info("开始无监督跨模态对比学习训练")
    logging.info(f"命令行参数: {vars(args)}")
    
    # 设置CUDA可见设备
    if args.device == 'cuda' and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        gpu_ids = [int(id.strip()) for id in args.gpus.split(',')]
        logging.info(f"使用的显卡ID: {gpu_ids}")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logging.info(f"使用设备: {device}")
    
    # 创建数据加载器
    logging.info("创建数据加载器...")
    dataloader = create_dataloader(config, args.batch_size, args.num_workers)
    logging.info(f"数据加载器创建完成，训练样本数: {len(dataloader.dataset)}, 批次大小: {args.batch_size}, 批次数量: {len(dataloader)}")
    
    # 计算并建议合适的batch size
    num_gpus = torch.cuda.device_count() if device.type == 'cuda' else 1
    logging.info(f"\n=== Batch Size 建议 ===")
    logging.info(f"当前设置: batch_size={args.batch_size}, 使用显卡数={num_gpus}")
    logging.info(f"注意：在multi_view_visual_encoder.py中，实际传入Backbone的图片数量会被放大为 B * N * NUM_NEIGHBOURS")
    logging.info(f"例如：如果N=4, NUM_NEIGHBOURS=5，那么每张显卡需要处理的图片数为: (16 * 4 * 5) / {num_gpus} = {16 * 4 * 5 / num_gpus}张")
    
    # 建议合理的batch size范围
    if device.type == 'cuda':
        suggested_batch_size = max(1, int(args.batch_size / num_gpus))
        logging.info(f"\n建议：在多卡环境下，您可以将batch size设置为 {suggested_batch_size} ~ {args.batch_size}")
        logging.info(f"这样每张显卡处理的实际图片数量会更合理，减少显存压力")
    
    # 初始化模型
    logging.info("初始化模型...")
    image_encoder = MultiViewVisualEncoder(feature_dim=1024, freeze_backbone=args.freeze_backbone)
    pointcloud_encoder = PointCloudEncoder(feature_dim=1024)
    
    # 记录是否冻结backbone
    logging.info(f"是否冻结ResNet Backbone: {args.freeze_backbone}")
    
    # 使用DataParallel包装模型，利用多张显卡
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f"检测到 {torch.cuda.device_count()} 张可用显卡，使用DataParallel...")
        image_encoder = torch.nn.DataParallel(image_encoder)
        pointcloud_encoder = torch.nn.DataParallel(pointcloud_encoder)
        logging.info("模型已使用DataParallel包装")
    
    # 将模型移动到设备
    image_encoder = image_encoder.to(device)
    pointcloud_encoder = pointcloud_encoder.to(device)
    
    # 初始化损失函数
    loss_module = InstanceDualContrastiveLoss(
        feature_dim=1024,
        projection_dim=512,
        temperature=0.07,
        weights={'lambda_intra': 0.5, 'lambda_inter': 0.5}
    )
    loss_module = loss_module.to(device)
    
    # 初始化优化器，同时优化两个编码器的参数
    optimizer = optim.Adam(
        list(image_encoder.parameters()) + list(pointcloud_encoder.parameters()),
        lr=args.lr
    )
    # === 新增：LR Scheduler 初始化 ===
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # 创建模型保存目录
    checkpoint_dir = os.path.join(args.checkpoint_dir, timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练循环
    logging.info("开始训练...")
    best_total_loss = float('inf')
    batch_counter = 0
    
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # 初始化损失统计
        epoch_total_loss = 0.0
        epoch_intra_loss = 0.0
        epoch_inter_loss = 0.0
        epoch_gen_loss = 0.0
        
        # 使用tqdm包装批次循环
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, batch in enumerate(batch_iter):
            try:
                # 过滤无效数据（点云为None的情况）
                if batch['pointcloud'] is None:
                    if batch_idx == 0:  # 仅在每轮第一次出现时打印警告
                        logging.warning("检测到无效点云数据，已跳过该批次")
                    continue
                
                # 将数据移动到设备
                for view_name in batch['views']:
                    batch['views'][view_name] = batch['views'][view_name].to(device)
                batch['pointcloud'] = batch['pointcloud'].to(device)
                
                # 前向传播
                # 1. 图像编码器
                outputs = image_encoder(batch, mode='train_with_gen')
                refined_view_feats, global_image_feat, loss_gen = outputs
                if isinstance(loss_gen, torch.Tensor) and loss_gen.dim() > 0:
                    loss_gen = loss_gen.mean()
                
                # 2. 点云编码器
                _, global_point_feat = pointcloud_encoder(batch)
                
                # 3. 计算损失
                loss_dict = loss_module(refined_view_feats, global_image_feat, global_point_feat)
                
                # 反向传播和优化
                optimizer.zero_grad()
                total_loss_tensor = loss_dict['total_loss'] + args.lambda_gen * loss_gen
                total_loss_tensor.backward()
                optimizer.step()
                
                # 记录损失
                total_loss = total_loss_tensor.item()
                intra_loss = loss_dict['intra_view_loss'].item()
                inter_loss = loss_dict['inter_modal_loss'].item()
                gen_loss = loss_gen.item()
                
                epoch_total_loss += total_loss
                epoch_intra_loss += intra_loss
                epoch_inter_loss += inter_loss
                if 'epoch_gen_loss' not in locals():
                    epoch_gen_loss = 0.0
                epoch_gen_loss += gen_loss
                
                # 记录到TensorBoard
                if writer is not None:
                    writer.add_scalar('Loss/total', total_loss, batch_counter)
                    writer.add_scalar('Loss/intra', intra_loss, batch_counter)
                    writer.add_scalar('Loss/inter', inter_loss, batch_counter)
                    writer.add_scalar('Loss/gen', gen_loss, batch_counter)
                
                # 在tqdm后缀中显示当前损失
                batch_iter.set_postfix({
                    'Total Loss': f'{total_loss:.4f}',
                    'Intra Loss': f'{intra_loss:.4f}',
                    'Inter Loss': f'{inter_loss:.4f}',
                    'Gen Loss': f'{gen_loss:.4f}'
                })
                
                batch_counter += 1
                
            except Exception as e:
                logging.error(f"批次处理错误 (Epoch {epoch+1}, Batch {batch_idx+1}): {str(e)}")
                logging.error(traceback.format_exc())
                continue
        
        # 计算平均损失
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_intra_loss = epoch_intra_loss / len(dataloader)
        avg_inter_loss = epoch_inter_loss / len(dataloader)
        avg_gen_loss = epoch_gen_loss / len(dataloader) if 'epoch_gen_loss' in locals() else 0.0
        
        logging.info(f"Epoch {epoch+1} 平均损失: Total={avg_total_loss:.4f}, Intra={avg_intra_loss:.4f}, Inter={avg_inter_loss:.4f}, Gen={avg_gen_loss:.4f}")
        # === 新增：根据总损失调整学习率，并记录当前学习率 ===
        scheduler.step(avg_total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"当前学习率: {current_lr:.6f}")
        
        # 记录平均损失到TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/avg_total', avg_total_loss, epoch)
            writer.add_scalar('Loss/avg_intra', avg_intra_loss, epoch)
            writer.add_scalar('Loss/avg_inter', avg_inter_loss, epoch)
            writer.add_scalar('Loss/avg_gen', avg_gen_loss, epoch)
            # === 新增：TensorBoard记录当前学习率 ===
            writer.add_scalar('Train/Learning_Rate', current_lr, epoch)
        
        # 保存检查点
        # 处理DataParallel的state_dict，移除module前缀
        def get_state_dict(model):
            if isinstance(model, torch.nn.DataParallel):
                return model.module.state_dict()
            return model.state_dict()
        
        # 1. 保存最后一个模型
        last_checkpoint = {
            'epoch': epoch+1,
            'image_encoder_state_dict': get_state_dict(image_encoder),
            'pointcloud_encoder_state_dict': get_state_dict(pointcloud_encoder),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_total_loss
        }
        torch.save(last_checkpoint, os.path.join(checkpoint_dir, 'last_model.pth'))
        
        # 2. 保存最优模型
        if avg_total_loss < best_total_loss:
            best_total_loss = avg_total_loss
            best_checkpoint = {
                'epoch': epoch+1,
                'image_encoder_state_dict': get_state_dict(image_encoder),
                'pointcloud_encoder_state_dict': get_state_dict(pointcloud_encoder),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_total_loss
            }
            torch.save(best_checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            logging.info(f"新的最优模型已保存，总损失: {best_total_loss:.4f}")
    
    # 训练完成
    logging.info("训练完成")
    logging.info(f"最优模型总损失: {best_total_loss:.4f}")
    logging.info(f"日志保存位置: {log_dir}")
    logging.info(f"模型检查点保存位置: {checkpoint_dir}")
    
    # 输出点云加载的最终统计信息
    if hasattr(dataloader.dataset, 'total_pointcloud_attempts') and hasattr(dataloader.dataset, 'pointcloud_load_failures'):
        total_attempts = dataloader.dataset.total_pointcloud_attempts
        failed_loads = dataloader.dataset.pointcloud_load_failures
        logging.info("\n=== 点云加载最终统计信息 ===")
        logging.info(f"总尝试加载次数: {total_attempts}")
        logging.info(f"加载失败次数: {failed_loads}")
        if total_attempts > 0:
            success_rate = ((total_attempts - failed_loads) / total_attempts) * 100
            logging.info(f"加载成功率: {success_rate:.2f}%")
    
    # 关闭SummaryWriter
    if writer is not None:
        writer.close()

if __name__ == '__main__':
    main()
