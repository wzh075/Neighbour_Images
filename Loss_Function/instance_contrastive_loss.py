import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceDualContrastiveLoss(nn.Module):
    """
    无监督实例级双重对比损失模块（Unsupervised Instance Dual Contrastive Loss Module）
    
    核心思想：
    1. 模态内损失：将同一实例的不同视点特征与全局图像特征对齐
    2. 跨模态损失：将同一实例的图像全局特征与点云全局特征对齐
    3. 使用Batch索引作为自监督信号，无需类别标签
    """
    
    def __init__(self, feature_dim=1024, projection_dim=512, temperature=0.07, 
                 weights={'lambda_intra': 0.5, 'lambda_inter': 0.5}):
        """
        初始化实例级双重对比损失模块
        
        参数：
            feature_dim: 输入特征维度（默认为1024）
            projection_dim: 投影空间维度（默认为512）
            temperature: InfoNCE损失的温度系数（默认为0.07）
            weights: 包含两个权重的字典：
                    - lambda_intra: 模态内损失权重（视点-图像全局）
                    - lambda_inter: 跨模态损失权重（图像全局-点云全局）
        """
        super(InstanceDualContrastiveLoss, self).__init__()
        
        # 保存超参数
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.weights = weights
        
        # 定义投影头
        # 1. 图像模态投影头（用于全局图像特征和精细化视点特征，共享权重）
        self.image_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # 2. 点云模态投影头（用于全局点云特征）
        self.pointcloud_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # 定义交叉熵损失（InfoNCE的实现基础）
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, refined_view_feats, global_image_feat, global_point_feat):
        """
        前向传播计算双重对比损失
        
        参数：
            refined_view_feats: (B, N, D) - 精细化视点特征
            global_image_feat: (B, D) - 全局图像特征
            global_point_feat: (B, D) - 全局点云特征
        
        返回：
            loss_dict: 包含总损失、模态内损失和跨模态损失的字典
        """
        B, N, D = refined_view_feats.size()
        
        # ----------------------
        # 1. 特征投影与归一化
        # ----------------------
        
        # 投影全局图像特征
        projected_global_image = F.normalize(self.image_projection(global_image_feat), dim=-1)
        
        # 投影精细化视点特征
        # 将(B, N, D)变形为(B*N, D)以便批量处理
        refined_view_feats_reshaped = refined_view_feats.reshape(-1, D)
        projected_view_feats = F.normalize(self.image_projection(refined_view_feats_reshaped), dim=-1)
        projected_view_feats = projected_view_feats.reshape(B, N, self.projection_dim)
        
        # 投影全局点云特征
        projected_global_point = F.normalize(self.pointcloud_projection(global_point_feat), dim=-1)
        
        # ----------------------
        # 2. Loss 1: 模态内视点-全局实例对齐
        # ----------------------
        
        # 计算所有视点与全局图像特征的相似度
        # projected_view_feats: (B, N, P)
        # projected_global_image: (B, P)
        # 扩展全局图像特征以便计算相似度 (B, N, P) -> (B*N, P)
        projected_global_image_expanded = projected_global_image.unsqueeze(1).expand(-1, N, -1).reshape(-1, self.projection_dim)
        
        # 计算相似度矩阵 (B*N, P) @ (P, B) = (B*N, B)
        similarity_matrix_intra = torch.matmul(projected_view_feats.reshape(-1, self.projection_dim), 
                                             projected_global_image.T) / self.temperature
        
        # 生成自监督标签：每个视点特征应与同一样本的全局图像特征对齐
        # 对于每个样本i的第n个视点，目标索引是i
        target_intra = torch.arange(B).repeat_interleave(N).to(similarity_matrix_intra.device)
        
        # 计算模态内损失
        loss_intra = self.cross_entropy_loss(similarity_matrix_intra, target_intra)
        
        # ----------------------
        # 3. Loss 2: 跨模态全局实例对齐
        # ----------------------
        
        # 计算图像全局特征与点云全局特征的相似度矩阵 (B, P) @ (P, B) = (B, B)
        similarity_matrix_cross_image_to_point = torch.matmul(projected_global_image, 
                                                             projected_global_point.T) / self.temperature
        
        similarity_matrix_cross_point_to_image = torch.matmul(projected_global_point, 
                                                             projected_global_image.T) / self.temperature
        
        # 生成自监督标签：每个样本i的图像特征应与同一样本i的点云特征对齐
        target_cross = torch.arange(B).to(similarity_matrix_cross_image_to_point.device)
        
        # 计算跨模态损失（双向）
        loss_cross_image_to_point = self.cross_entropy_loss(similarity_matrix_cross_image_to_point, target_cross)
        loss_cross_point_to_image = self.cross_entropy_loss(similarity_matrix_cross_point_to_image, target_cross)
        loss_inter = (loss_cross_image_to_point + loss_cross_point_to_image) / 2
        
        # ----------------------
        # 4. 总损失聚合
        # ----------------------
        total_loss = self.weights['lambda_intra'] * loss_intra + self.weights['lambda_inter'] * loss_inter
        
        # 返回损失字典
        loss_dict = {
            'total_loss': total_loss,
            'intra_view_loss': loss_intra,
            'inter_modal_loss': loss_inter
        }
        
        return loss_dict


if __name__ == "__main__":
    # 测试实例级双重对比损失模块
    import torch
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建模拟数据
    B = 4  # Batch大小
    N = 8  # 视点数
    D = 1024  # 特征维度
    
    # 模拟输入特征
    refined_view_feats = torch.randn(B, N, D)
    global_image_feat = torch.randn(B, D)
    global_point_feat = torch.randn(B, D)
    
    # 初始化损失模块
    loss_module = InstanceDualContrastiveLoss(feature_dim=D, projection_dim=512, temperature=0.07,
                                            weights={'lambda_intra': 0.5, 'lambda_inter': 0.5})
    
    # 计算损失
    loss_dict = loss_module(refined_view_feats, global_image_feat, global_point_feat)
    
    # 打印结果
    print("=== 实例级双重对比损失测试结果 ===")
    print(f"总损失: {loss_dict['total_loss'].item():.4f}")
    print(f"模态内损失: {loss_dict['intra_view_loss'].item():.4f}")
    print(f"跨模态损失: {loss_dict['inter_modal_loss'].item():.4f}")
    
    # 检查损失是否为标量
    assert isinstance(loss_dict['total_loss'], torch.Tensor)
    assert loss_dict['total_loss'].dim() == 0
    print("测试通过！所有损失均为标量张量。")