import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models, transforms
from tqdm import tqdm

# === 1. 环境设置: 添加项目根目录到 sys.path ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from DataLoader.data_loader import ModelNet40NeighbourDataset

# === 2. MVCNN Baseline 模型定义 ===
class MVCNN_Baseline(nn.Module):
    def __init__(self, backbone_name='resnet50'):
        super(MVCNN_Baseline, self).__init__()
        print(f">> 加载 ImageNet 预训练 {backbone_name}...")
        if backbone_name == 'resnet50':
            base_model = models.resnet50(pretrained=True)
        else:
            raise NotImplementedError
        
        # 去掉分类头 (fc)，保留特征提取部分
        modules = list(base_model.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        
        # 冻结参数，仅作为特征提取器
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # x shape: (B, V, Neigh, C, H, W)
        # 支持 V=1 (单视点查询) 或 V=12 (全视点建库)
        B, V, Neigh, C, H, W = x.size()
        
        # 1. 展平所有维度以输入 ResNet
        x = x.view(B * V * Neigh, C, H, W)
        
        # 2. 提取特征 -> (B*V*Neigh, 2048, 1, 1)
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1) # (B*V*Neigh, 2048)
        
        # 3. 还原维度 -> (B, V*Neigh, 2048)
        # MVCNN 的核心是将所有视点视为无序集合
        feats = feats.view(B, V * Neigh, -1)
        
        # 4. Global Pooling (Max) -> (B, 2048)
        global_feat, _ = torch.max(feats, dim=1)
        
        # 5. L2 归一化
        global_feat = torch.nn.functional.normalize(global_feat, p=2, dim=1)
        
        return global_feat

# === 3. 评估函数: Instance Recall ===
def evaluate_recall(query_feats, gallery_feats, query_ids, gallery_ids):
    """
    计算实例级召回率。
    成功标准: Retrieved_ID == Query_ID
    """
    num_queries = query_feats.shape[0]
    
    # 转移到 GPU 加速计算
    query_feats = query_feats.cuda()
    gallery_feats = gallery_feats.cuda()
    
    # 计算余弦相似度矩阵
    sim_matrix = torch.matmul(query_feats, gallery_feats.t())
    
    # 获取 Top-5 索引
    _, indices = sim_matrix.topk(k=5, dim=1)
    indices = indices.cpu().numpy()
    
    recall_1 = 0
    recall_5 = 0
    
    for i in range(num_queries):
        q_id = query_ids[i]
        
        # 获取检索到的 ID 列表
        retrieved_ids = [gallery_ids[idx] for idx in indices[i]]
        
        if q_id in retrieved_ids[:1]:
            recall_1 += 1
        if q_id in retrieved_ids[:5]:
            recall_5 += 1
            
    return recall_1 / num_queries, recall_5 / num_queries

# === 4. 主执行流程 ===
def main():
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 请根据实际环境调整数据集路径
    # DATASET_ROOT = os.path.join(project_root, 'data', 'ModelNet40')
    DATASET_ROOT = "/data1/Wuzhihe/Neighbour_Images/Dataset/ModelNet40_Neighbour_view4_1.0"
    
    print("=== MVCNN 单视点实例检索基准测试 (Single View Instance Retrieval) ===")
    
    # 4.1 数据加载 (Train + Test 合并)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # ModelNet40NeighbourDataset loads all splits by default
        full_dataset = ModelNet40NeighbourDataset(DATASET_ROOT, transform=data_transform)
        dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        print(f"Dataset loaded: {len(full_dataset)} total samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 4.2 模型初始化
    model = MVCNN_Baseline().to(DEVICE)
    model.eval()
    
    # 4.3 提取全库所有视点的特征
    # 我们需要暂存所有视点的特征，以便后续灵活构建 Query 和 Gallery
    all_view_feats = [] # List of (B, 12, 2048)
    all_obj_ids = []
    
    print(">> Extracting features for all 12 views...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # 收集 Object IDs
            ids = batch['object_id']
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            all_obj_ids.extend(ids)
            
            # 准备输入: 获取实际的视图名称并排序
            view_names = sorted(batch['views'].keys())
            num_views = len(view_names)
            print(f"Found {num_views} views: {view_names}")
            
            # 准备输入: (B, V, 5, 3, H, W)
            batch_views = []
            for view_name in view_names:
                v_imgs = batch['views'][view_name].to(DEVICE)
                batch_views.append(v_imgs.unsqueeze(1))
            full_input = torch.cat(batch_views, dim=1)
            
            # 分视点提取 (为了节省显存，循环提取)
            batch_feats_per_view = []
            for view_idx in range(num_views):
                # 输入单视点: (B, 1, 5, 3, H, W) -> 输出 (B, 2048)
                single_view_feat = model(full_input[:, view_idx:view_idx+1, ...])
                batch_feats_per_view.append(single_view_feat.cpu().unsqueeze(1))
            
            # 拼接: (B, V, 2048)
            all_view_feats.append(torch.cat(batch_feats_per_view, dim=1))
            
    # 合并所有 Batch -> (Total_Samples, V, 2048)
    all_view_feats = torch.cat(all_view_feats, dim=0)
    num_views = all_view_feats.shape[1]
    print(f"Total views per object: {num_views}")
    
    # 4.4 构建 Gallery (上帝视角)
    # 聚合所有视点的特征 -> (Total_Samples, 2048)
    print(f">> Building Gallery (Max Pooling {num_views} views)...")
    gallery_feats, _ = torch.max(all_view_feats, dim=1)
    gallery_feats = torch.nn.functional.normalize(gallery_feats, p=2, dim=1)
    gallery_ids = all_obj_ids
    
    # 4.5 评估循环: 轮流将每个视点作为 Query
    print("\n>> Evaluating Single View Retrieval Performance...")
    print(f"{'Query View':<12} {'R@1':<8} {'R@5':<8}")
    print("-" * 30)
    
    r1_scores = []
    r5_scores = []
    
    for v in range(num_views):
        # Query: 仅使用第 v 个视点
        query_feats = all_view_feats[:, v, :] # (Total_Samples, 2048)
        query_feats = torch.nn.functional.normalize(query_feats, p=2, dim=1)
        
        # 这里的场景是: 拿着第 v 个视点，去库里找这个物体
        r1, r5 = evaluate_recall(query_feats, gallery_feats, all_obj_ids, gallery_ids)
        
        r1_scores.append(r1)
        r5_scores.append(r5)
        print(f"View {v:<7} {r1*100:.2f}%   {r5*100:.2f}%")
        
    print("-" * 30)
    print(f"Average     {np.mean(r1_scores)*100:.2f}%   {np.mean(r5_scores)*100:.2f}%")

if __name__ == '__main__':
    main()