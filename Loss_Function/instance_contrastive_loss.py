import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewDropoutContrastiveLoss(nn.Module):
    """
    视点随机丢弃对比损失模块（View-Dropout Contrastive Loss Module）
    
    核心思想：
    1. 比较全视点全局特征与随机子集视点全局特征的一致性
    2. 使用对称 InfoNCE 损失，确保双向一致性
    3. 保留方差正则化，防止特征塌陷
    """
    
    def __init__(self, feature_dim=1024, projection_dim=512, temperature=0.07, 
                 weights={'lambda_contrastive': 1.0, 'lambda_var': 1.0}):
        """
        初始化视点随机丢弃对比损失模块
        
        参数：
            feature_dim: 输入特征维度（默认为1024）
            projection_dim: 投影空间维度（默认为512）
            temperature: InfoNCE损失的温度系数（默认为0.07）
            weights: 包含权重的字典：
                    - lambda_contrastive: 对比损失权重
                    - lambda_var: 方差正则化损失权重
        """
        super(ViewDropoutContrastiveLoss, self).__init__()
        
        # 保存超参数
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.weights = weights
        
        # 添加默认权重
        if 'lambda_contrastive' not in self.weights:
            self.weights['lambda_contrastive'] = 1.0
        if 'lambda_var' not in self.weights:
            self.weights['lambda_var'] = 1.0
        
        # 定义投影头
        self.image_projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # 定义交叉熵损失（InfoNCE的实现基础）
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, global_feat_sub, global_feat_all):
        """
        前向传播计算对比损失
        
        参数：
            global_feat_sub: (B, D) - 子集视点全局特征
            global_feat_all: (B, D) - 全视点全局特征
        
        返回：
            loss_dict: 包含总损失、对比损失和方差损失的字典
        """
        B, D = global_feat_all.size()
        
        # ----------------------
        # 1. 特征投影与归一化
        # ----------------------
        
        # 投影特征
        z_sub = self.image_projection(global_feat_sub)
        z_all = self.image_projection(global_feat_all)
        
        # 归一化投影特征
        Z_sub = F.normalize(z_sub, p=2, dim=-1)
        Z_all = F.normalize(z_all, p=2, dim=-1)
        
        # ----------------------
        # 2. 对称 InfoNCE 损失
        # ----------------------
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(Z_sub, Z_all.T) / self.temperature
        
        # 创建标签
        labels = torch.arange(B, device=Z_sub.device)
        
        # 计算对称对比损失
        loss_contrastive = (self.cross_entropy_loss(sim_matrix, labels) + 
                           self.cross_entropy_loss(sim_matrix.T, labels)) / 2.0
        
        # ----------------------
        # 3. 方差正则化损失
        # ----------------------
        
        # 计算Batch维度上的标准差
        eps = 1e-4
        std_all = torch.sqrt(Z_all.var(dim=0) + eps)
        loss_var = torch.mean(F.relu(1.0 - std_all))
        
        # ----------------------
        # 4. 总损失聚合
        # ----------------------
        
        total_loss = (self.weights['lambda_contrastive'] * loss_contrastive + 
                     self.weights['lambda_var'] * loss_var)
        
        # 返回损失字典
        loss_dict = {
            'total_loss': total_loss,
            'contrastive_loss': loss_contrastive,
            'var_loss': loss_var
        }
        
        return loss_dict


if __name__ == "__main__":
    # 测试视点随机丢弃对比损失模块
    import torch
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 创建模拟数据
    B = 4  # Batch大小
    D = 1024  # 特征维度
    
    # 模拟输入特征
    global_feat_sub = torch.randn(B, D)
    global_feat_all = torch.randn(B, D)
    
    # 初始化损失模块
    loss_module = ViewDropoutContrastiveLoss(feature_dim=D, projection_dim=512, temperature=0.07,
                                           weights={'lambda_contrastive': 1.0, 'lambda_var': 1.0})
    
    # 计算损失
    loss_dict = loss_module(global_feat_sub, global_feat_all)
    
    # 打印结果
    print("=== 视点随机丢弃对比损失测试结果 ===")
    print(f"总损失: {loss_dict['total_loss'].item():.4f}")
    print(f"对比损失: {loss_dict['contrastive_loss'].item():.4f}")
    print(f"方差损失: {loss_dict['var_loss'].item():.4f}")
    
    # 检查损失是否为标量
    assert isinstance(loss_dict['total_loss'], torch.Tensor)
    assert loss_dict['total_loss'].dim() == 0
    print("测试通过！所有损失均为标量张量。")