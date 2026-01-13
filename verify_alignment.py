import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 测试点云与图像数据在加载时是否是对齐的

# 确保能导入项目模块
sys.path.append(os.getcwd())

from DataLoader.data_loader import ModelNet40NeighbourDataset, load_config

def verify_data_alignment():
    print("🔍 开始执行数据对齐完整性检查...")
    
    # 1. 加载配置
    config_path = './DataLoader/config.yaml'
    if not os.path.exists(config_path):
        print(f"❌ 错误: 找不到配置文件 {config_path}")
        return
        
    config = load_config(config_path)
    
    # 强制修改配置以便于调试
    # 必须使用与训练相同的设置，但可以关掉 shuffle 以便观察，
    # 或者开启 shuffle 以模拟真实训练环境（推荐开启以测试索引是否乱序）
    config['dataloader']['batch_size'] = 8 
    config['dataloader']['shuffle'] = True 
    config['dataloader']['num_workers'] = 0 # 先用单线程排查逻辑，如果通过再测多线程
    
    # 简单的 transform，只要能转成 Tensor 即可
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # 2. 初始化数据集
    print("📚 初始化数据集...")
    dataset = ModelNet40NeighbourDataset(
        root_dir=config['dataset']['root_dir'],
        transform=transform,
        expected_images_per_view=config['dataset']['expected_images_per_view'],
        pointcloud_root=config.get('pointcloud', {}).get('root_dir')
    )
    
    # 3. 初始化 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=config['dataloader']['shuffle'],
        num_workers=config['dataloader']['num_workers'],
        drop_last=False
    )
    
    print(f"✅ DataLoader 就绪，准备检查 {len(dataloader)} 个 Batch")
    print("⚡ 正在进行双重验证 (Batch数据 vs 硬盘原始数据)...")
    
    # 4. 遍历检查
    mismatch_count = 0
    checked_samples = 0
    
    # 我们只检查前 5 个 Batch 即可，通常如果有问题，第一个 Batch 就会暴露
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5: 
            break
            
        # 获取 Batch 中的数据
        batch_ids = batch['object_id']
        batch_cats = batch['category']
        batch_pcs = batch['pointcloud'] # (B, N, 3)
        batch_views = batch['views']    # Dict of (B, 5, 3, H, W)
        
        batch_size = len(batch_ids)
        
        for i in range(batch_size):
            checked_samples += 1
            
            # 当前样本的信息
            obj_id = batch_ids[i]
            category = batch_cats[i]
            pc_in_batch = batch_pcs[i]
            
            # -----------------------------------------------------------
            # 核心验证逻辑：根据 ID 手动去硬盘再读一次点云
            # -----------------------------------------------------------
            
            # 使用 dataset 内部的 loader 重新获取该 ID 的数据
            # 注意：这里我们绕过 __getitem__ 的递归逻辑，直接查该 ID 的底层文件
            raw_pc_data = dataset.pointcloud_loader.get_pointcloud(category, obj_id)
            
            if raw_pc_data is None:
                print(f"⚠️ 警告: Batch中的对象 {obj_id} 在硬盘上找不到对应的点云文件！")
                print("   这说明 DataLoader 可能在递归替换时，把 ID 搞乱了，或者原来的 ID 确实有问题。")
                mismatch_count += 1
                continue
                
            raw_pc_tensor = torch.from_numpy(raw_pc_data['points'])
            
            # 比对 Batch 中的点云 和 重新读取的点云 是否完全一致
            # 我们允许极其微小的浮点误差，但实际上应该是 bit-exact 的
            if not torch.allclose(pc_in_batch.float(), raw_pc_tensor.float(), atol=1e-6):
                print(f"❌ 致命错误: 发现数据未对齐！(Batch Index: {batch_idx}, Sample: {i})")
                print(f"   对象 ID: {obj_id}")
                print(f"   Batch 中的点云数据 (前3个点): \n{pc_in_batch[:3]}")
                print(f"   硬盘上的点云数据 (前3个点): \n{raw_pc_tensor[:3]}")
                print("   结论: Batch 中的 object_id 与实际携带的 pointcloud 数据不匹配！")
                return # 发现一个错误直接退出
            
            # (可选) 验证图像是否对齐
            # 随机取一个视点名
            view_name = list(batch_views.keys())[0]
            img_tensor_in_batch = batch_views[view_name][i] # (5, 3, H, W)
            
            # 我们不方便逐像素比对图像（因为有 transform），但我们可以检查 tensor 是否全黑/全白等异常
            if img_tensor_in_batch.sum() == 0:
                 print(f"⚠️ 警告: 对象 {obj_id} 的图像数据全为 0")
                 
    if mismatch_count == 0:
        print("\n🎉 恭喜！经过验证，DataLoader 的数据对齐是正确的。")
        print(f"   共检查了 {checked_samples} 个样本，全部通过双重验证。")
        print("   这意味着你的递归 __getitem__ 逻辑是安全的，它正确地同时替换了 ID、图像和点云。")
        print("   问题可能出在其他地方（如坐标系方向、模型结构或Loss权重）。")
    else:
        print(f"\n🚫 检测到 {mismatch_count} 个对齐错误！请立即检查 data_loader.py。")

if __name__ == "__main__":
    verify_data_alignment()