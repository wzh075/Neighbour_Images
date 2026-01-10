import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAlignmentLoss(nn.Module):
    """
    双重对齐联合损失模块（Dual Alignment Joint Loss Module）
    
    核心功能：
    1. 模态内对齐（Intra-modal Alignment）：将多个视点特征与全局图像特征对齐
    2. 跨模态对齐（Cross-modal Alignment）：将图像全局特征与点云全局特征对齐
    3. 语义分类（Semantic Classification）：利用图像和点云特征进行分类监督
    
    适用于：多视图图像与点云的跨模态对比学习任务
    """
    
    def __init__(self, feature_dim=1024, projection_dim=512, num_classes=40, 
                 temperature=0.07, weights=None):
        """
        参数初始化
        
        参数:
        - feature_dim: 编码器输出的特征维度 (默认: 1024)
        - projection_dim: 对比学习投影空间的维度 (默认: 512)
        - num_classes: 分类类别数 (默认: 40)
        - temperature: InfoNCE Loss的温度系数 (默认: 0.07)
        - weights: 损失权重列表 [lambda_intra, lambda_inter, lambda_cls] (默认: [0.5, 0.5, 0.1])
        """
        super(DualAlignmentLoss, self).__init__()
        
        # 损失权重配置
        if weights is None:
            self.weights = {
                'lambda_intra': 0.5,  # 视点-全局对齐权重
                'lambda_inter': 0.5,  # 跨模态对齐权重
                'lambda_cls': 0.1     # 分类损失权重
            }
        elif isinstance(weights, list) and len(weights) == 3:
            self.weights = {
                'lambda_intra': weights[0],
                'lambda_inter': weights[1],
                'lambda_cls': weights[2]
            }
        else:
            self.weights = weights
        
        self.temperature = temperature
        
        # ----------------------
        # 投影头设计 (Projection Heads)
        # ----------------------
        # 策略：图像全局特征和视点特征共享一个投影头，点云使用独立投影头
        # 理由：图像的全局特征和视点特征来自同一编码器，具有相似的特征分布
        #       点云特征来自不同编码器，需要独立的投影空间映射
        
        # 图像/视点共享投影头 (MLP: Linear -> ReLU -> Linear)
        self.img_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # 点云独立投影头 (MLP: Linear -> ReLU -> Linear)
        self.pc_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # ----------------------
        # 分类器设计 (Classifiers)
        # ----------------------
        # 用于语义监督的全连接分类层
        self.img_classifier = nn.Linear(feature_dim, num_classes)
        self.pc_classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, refined_view_feats, global_image_feat, global_point_feat, labels):
        """
        前向传播计算损失
        
        参数:
        - refined_view_feats: 视点特征, 形状 (B, N, D) 其中 B=batch_size, N=视点数量, D=feature_dim
        - global_image_feat: 图像全局特征, 形状 (B, D)
        - global_point_feat: 点云全局特征, 形状 (B, D)
        - labels: 真实标签, 形状 (B,)
        
        返回:
        - 损失字典，包含总损失和各分支损失
        """
        B, N, D = refined_view_feats.shape
        
        # ----------------------
        # 1. 特征投影 (Feature Projection)
        # ----------------------
        # 图像全局特征投影并L2归一化
        proj_global_image = F.normalize(self.img_projection(global_image_feat), dim=-1)  # (B, projection_dim)
        
        # 视点特征投影并L2归一化
        # 先将形状从 (B, N, D) 转换为 (B*N, D)
        proj_refined_view = refined_view_feats.view(B*N, D)
        proj_refined_view = F.normalize(self.img_projection(proj_refined_view), dim=-1)
        proj_refined_view = proj_refined_view.view(B, N, -1)  # (B, N, projection_dim)
        
        # 点云全局特征投影并L2归一化
        proj_global_point = F.normalize(self.pc_projection(global_point_feat), dim=-1)  # (B, projection_dim)
        
        # ----------------------
        # 2. 损失计算 (Loss Calculation)
        # ----------------------
        
        # ----------------------
        # 2.1 模态内视点对齐损失 (Intra-modal View Alignment Loss)
        # ----------------------
        # 目标：将每个样本的N个视点特征拉向其对应的全局图像特征
        
        # 扩展全局图像特征以匹配视点特征形状 (B, projection_dim) -> (B, N, projection_dim)
        expanded_global_image = proj_global_image.unsqueeze(1).expand(-1, N, -1)
        
        # 计算每个视点与全局特征的相似度 (B, N)
        view_global_similarity = torch.sum(proj_refined_view * expanded_global_image, dim=-1)
        
        # 创建批内负样本
        # 将全局图像特征作为锚点，所有视点特征作为候选
        all_proj_views = proj_refined_view.view(B*N, -1)  # (B*N, projection_dim)
        all_proj_images = proj_global_image.unsqueeze(1).expand(-1, N, -1).reshape(B*N, -1)  # (B*N, projection_dim)
        
        # 计算锚点与所有候选的相似度矩阵 (B*N, B*N)
        similarity_matrix = torch.matmul(all_proj_images, all_proj_views.T) / self.temperature
        
        # 创建标签：对角线元素为正样本，其余为负样本
        labels_intra = torch.arange(B*N, device=similarity_matrix.device)
        
        # 计算InfoNCE Loss
        intra_loss = F.cross_entropy(similarity_matrix, labels_intra)
        
        # ----------------------
        # 2.2 跨模态全局对齐损失 (Cross-modal Global Alignment Loss)
        # ----------------------
        # 目标：对齐同一样本的图像全局特征和点云全局特征
        
        # 计算图像和点云之间的相似度矩阵 (B, B)
        cross_similarity = torch.matmul(proj_global_image, proj_global_point.T) / self.temperature
        
        # 创建标签：对角线元素为正样本，其余为负样本
        labels_cross = torch.arange(B, device=cross_similarity.device)
        
        # 计算对称InfoNCE Loss
        loss_cross_image = F.cross_entropy(cross_similarity, labels_cross)
        loss_cross_point = F.cross_entropy(cross_similarity.T, labels_cross)
        inter_loss = (loss_cross_image + loss_cross_point) / 2
        
        # ----------------------
        # 2.3 语义分类损失 (Semantic Classification Loss)
        # ----------------------
        # 使用原始（未投影的）全局特征进行分类
        img_logits = self.img_classifier(global_image_feat)
        pc_logits = self.pc_classifier(global_point_feat)
        
        # 计算交叉熵损失
        img_cls_loss = F.cross_entropy(img_logits, labels)
        pc_cls_loss = F.cross_entropy(pc_logits, labels)
        cls_loss = (img_cls_loss + pc_cls_loss) / 2
        
        # ----------------------
        # 3. 总损失聚合 (Total Loss Aggregation)
        # ----------------------
        total_loss = (
            self.weights['lambda_intra'] * intra_loss +
            self.weights['lambda_inter'] * inter_loss +
            self.weights['lambda_cls'] * cls_loss
        )
        
        # 计算分类准确率
        img_preds = torch.argmax(img_logits, dim=1)
        pc_preds = torch.argmax(pc_logits, dim=1)
        img_acc = (img_preds == labels).float().mean()
        pc_acc = (pc_preds == labels).float().mean()
        
        return {
            'total_loss': total_loss,
            'intra_view_loss': intra_loss,
            'inter_modal_loss': inter_loss,
            'cls_loss': cls_loss,
            'img_acc': img_acc,
            'pc_acc': pc_acc
        }


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    B = 2  # Batch size
    N = 4  # Number of views
    D = 1024  # Feature dimension
    projection_dim = 512
    num_classes = 40
    
    # 创建特征张量
    refined_view_feats = torch.randn(B, N, D)
    global_image_feat = torch.randn(B, D)
    global_point_feat = torch.randn(B, D)
    labels = torch.randint(0, num_classes, (B,))
    
    # 初始化损失函数
    loss_fn = DualAlignmentLoss(
        feature_dim=D,
        projection_dim=projection_dim,
        num_classes=num_classes,
        temperature=0.07,
        weights=[0.5, 0.5, 0.1]
    )
    
    # 计算损失
    losses = loss_fn(refined_view_feats, global_image_feat, global_point_feat, labels)
    
    # 打印结果
    print("=== Dual Alignment Loss Test Results ===")
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
    print(f"Intra-view Loss: {losses['intra_view_loss'].item():.4f}")
    print(f"Inter-modal Loss: {losses['inter_modal_loss'].item():.4f}")
    print(f"Classification Loss: {losses['cls_loss'].item():.4f}")
    print(f"Image Classification Accuracy: {losses['img_acc'].item():.4f}")
    print(f"Point Cloud Classification Accuracy: {losses['pc_acc'].item():.4f}")
    
    print("\n=== Loss Weights ===")
    print(f"Lambda Intra: {loss_fn.weights['lambda_intra']}")
    print(f"Lambda Inter: {loss_fn.weights['lambda_inter']}")
    print(f"Lambda Cls: {loss_fn.weights['lambda_cls']}")
    
    print("\n✓ DualAlignmentLoss test completed successfully!")