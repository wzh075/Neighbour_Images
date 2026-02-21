import os
import sys
import argparse
import logging
import h5py
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from DataLoader.data_loader import ModelNet40NeighbourDataset, load_config
from Models.multi_view_visual_encoder import MultiViewVisualEncoder

def setup_logging():
    """设置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def find_latest_checkpoint(checkpoint_dir):
    """
    自动寻找最新的checkpoint文件夹
    :param checkpoint_dir: checkpoint根目录
    :return: 最新checkpoint文件夹路径
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"No checkpoint subdirectories found in: {checkpoint_dir}")
    
    # 解析时间戳并排序
    def parse_timestamp(dir_name):
        try:
            return datetime.strptime(dir_name, '%Y-%m-%d_%H-%M-%S')
        except ValueError:
            return datetime.min
    
    sorted_subdirs = sorted(subdirs, key=parse_timestamp, reverse=True)
    latest_subdir = sorted_subdirs[0]
    latest_checkpoint_path = os.path.join(checkpoint_dir, latest_subdir, 'best_model.pth')
    
    if not os.path.exists(latest_checkpoint_path):
        raise FileNotFoundError(f"best_model.pth not found in: {os.path.join(checkpoint_dir, latest_subdir)}")
    
    logging.info(f"Found latest checkpoint: {latest_checkpoint_path}")
    return latest_checkpoint_path

def create_dataloader(config, batch_size, num_workers):
    """创建数据加载器"""
    from torchvision import transforms
    
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
    
    # 创建数据集 - 默认加载所有split数据（train+test）
    dataset = ModelNet40NeighbourDataset(
        root_dir=config['dataset']['root_dir'],
        transform=transform,
        expected_images_per_view=config['dataset']['expected_images_per_view'],
        pointcloud_root=config.get('pointcloud', {}).get('root_dir')
    )
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 提取特征时不打乱顺序
        num_workers=num_workers,
        drop_last=False  # 不丢弃最后一个不完整批次
    )
    
    return dataloader

def extract_features(args):
    """提取特征并保存"""
    setup_logging()
    logging.info("开始特征提取流程")
    
    # 加载配置
    config = load_config(args.config)
    logging.info(f"配置文件加载完成: {args.config}")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    # 初始化模型
    logging.info("初始化模型...")
    image_encoder = MultiViewVisualEncoder(feature_dim=1024)
    
    # 加载模型权重
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        logging.info(f"使用手动指定的checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = find_latest_checkpoint('../Checkpoints')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重（处理DataParallel情况）
    def load_state_dict(model, state_dict):
        if 'module.' in list(state_dict.keys())[0]:
            # 如果是DataParallel保存的模型，移除module前缀
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    
    if 'student_model_state_dict' in checkpoint:
        state_dict = checkpoint['student_model_state_dict']
    elif 'image_encoder_state_dict' in checkpoint:
        state_dict = checkpoint['image_encoder_state_dict']
    else:
        state_dict = checkpoint
        
    load_state_dict(image_encoder, state_dict)
    
    # 移动模型到设备并设置为评估模式
    image_encoder = image_encoder.to(device).eval()
    
    logging.info("模型加载完成")
    
    # 创建数据加载器
    dataloader = create_dataloader(config, args.batch_size, args.num_workers)
    logging.info(f"数据加载器创建完成，总样本数: {len(dataloader.dataset)}")
    
    # 准备特征存储
    all_global_image_feats = []
    all_object_ids = []
    all_categories = []
    
    # 开始提取特征
    logging.info("开始提取特征...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="提取特征")):
            # 将数据移动到设备
            for view_name in batch['views']:
                batch['views'][view_name] = batch['views'][view_name].to(device)
            
            # 提取特征 (Teacher 模式)
            global_image_feat = image_encoder(batch, return_predictor=False)
            
            # 保存特征（移回CPU并转为numpy数组）
            all_global_image_feats.append(global_image_feat.cpu().numpy())
            
            # 保存元数据
            all_object_ids.extend(batch['object_id'])
            all_categories.extend(batch['category'])
    
    # 合并所有特征
    logging.info("合并特征数组...")
    global_image_feats = np.concatenate(all_global_image_feats, axis=0)
    
    logging.info(f"特征提取完成：")
    logging.info(f"  - Global image features shape: {global_image_feats.shape}")
    logging.info(f"  - Total valid samples: {len(all_object_ids)}")
    
    # 创建Embedding文件夹
    embedding_dir = '../Embedding'
    os.makedirs(embedding_dir, exist_ok=True)
    
    # 保存到HDF5文件
    feature_file = os.path.join(embedding_dir, 'features_all.h5')
    logging.info(f"保存特征到：{feature_file}")
    
    with h5py.File(feature_file, 'w') as f:
        f.create_dataset('global_image_feats', data=global_image_feats, dtype='float32')
        
        # 保存字符串数据
        f.create_dataset('object_ids', data=np.array(all_object_ids, dtype='S'))
        f.create_dataset('categories', data=np.array(all_categories, dtype='S'))
    
    logging.info("特征保存完成")
    logging.info(f"特征文件路径：{feature_file}")
    logging.info(f"特征文件大小：{os.path.getsize(feature_file) / (1024*1024):.2f} MB")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='特征提取脚本')
    parser.add_argument('--config', type=str, default='../DataLoader/config.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型权重路径（手动指定）')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader工作线程数')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备类型')
    
    args = parser.parse_args()
    extract_features(args)

if __name__ == '__main__':
    main()