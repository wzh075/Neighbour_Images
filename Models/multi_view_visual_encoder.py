import torch
import torch.nn as nn
from torchvision.models import resnet50


class SelfAttentionBlock(nn.Module):
    """Self-Attention Block (SAB) for inter-view feature interaction
    
    用于视点间（Inter-view）的特征交互。
    在邻域特征融合形成稳健的视点特征后，该模块处理不同视点之间的上下文关系。
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        # Multihead attention layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (B, N, D)
        
        # Self-attention with residual connection
        x2 = self.self_attn(x, x, x)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        # Feedforward with residual connection
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        
        return x


class PoolingByMultiheadAttention(nn.Module):
    """Pooling by Multihead Attention (PMA) for global feature aggregation"""
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(PoolingByMultiheadAttention, self).__init__()
        # Learnable seed token
        self.seed_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Multihead attention layer
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (B, N, D)
        B = x.size(0)
        
        # Expand seed token to batch size
        seed = self.seed_token.expand(B, -1, -1)  # (B, 1, D)
        
        # Multihead attention between seed and input features
        x2 = self.multihead_attn(seed, x, x)[0]
        x = seed + self.dropout1(x2)
        x = self.norm1(x)
        
        # Feedforward network
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        
        return x


class SetTransformerAggregation(nn.Module):
    """Set Transformer for aggregating multi-view features"""
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(SetTransformerAggregation, self).__init__()
        # Self-Attention Block for inter-view interaction
        self.sab = SelfAttentionBlock(d_model, nhead, dim_feedforward, dropout)
        
        # Pooling by Multihead Attention for global feature aggregation
        self.pma = PoolingByMultiheadAttention(d_model, nhead, dim_feedforward, dropout)

    def forward(self, x):
        # x: (B, N, D) - input view features
        
        # Step 1: Inter-view feature interaction
        refined_view_feats = self.sab(x)  # (B, N, D)
        
        # Step 2: Global feature aggregation
        aggregated_feat = self.pma(refined_view_feats)  # (B, 1, D)
        
        return refined_view_feats, aggregated_feat


class MultiViewVisualEncoder(nn.Module):
    """
    Multi-View Visual Encoder with Neighbour Image Fusion Strategy.
    
    核心思想 (Core Concept):
    该编码器不仅处理多个视点（Views），还引入了“邻域图（Neighbour Images）”的概念。
    
    结构定义:
    1. 输入层级：Batch -> N个视点 -> 每个视点包含 M张邻域图 (默认5张)。
       - 邻域图定义：包含一张中心视点图，以及相机向上下左右微偏渲染的4张互补图。
    
    2. 处理流程：
       - Stage 1 (Backbone): 对所有邻域图独立提取特征。
       - Stage 2 (Intra-View Fusion): 通过最大池化（Max Pooling）融合同一视点下的5张邻域图特征。
         这一步不是为了数据增强，而是为了利用邻域的互补信息，形成一个抗遮挡、信息更丰富的“全局视点表示”。
       - Stage 3 (Inter-View Aggregation): 使用 Set Transformer 融合 N 个增强后的视点特征，生成 3D 对象的全局描述。
    """
    def __init__(self, backbone_type='resnet50', feature_dim=1024, nhead=8, dim_feedforward=2048, dropout=0.1, freeze_backbone=False):
        super(MultiViewVisualEncoder, self).__init__()
        
        # Step 1: Initialize backbone
        if backbone_type == 'resnet50':
            # 使用 ImageNet 预训练权重
            self.backbone = resnet50(pretrained=True)
            print("Using default ImageNet pretrained weights for ResNet50")
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Remove classification head (fc layer)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Step 2: Projection layer to reduce feature dim from 2048 to 1024
        self.projection = nn.Linear(2048, feature_dim)
        
        # Step 3: Set Transformer for aggregation
        self.set_transformer = SetTransformerAggregation(feature_dim, nhead, dim_feedforward, dropout)
        
        # Step 4: Predictor for Student network
        # Predictor: 1024 -> 512 -> 1024
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim)
        )
        
        # Step 5: Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"ResNet Backbone parameters have been frozen. Only projection layers, set transformer and predictor will be trained.")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"All parameters (including ResNet Backbone) will be trained.")

    def forward(self, batch, return_predictor=False):
        # Input: batch containing 'views' dictionary
        views_dict = batch['views']
        
        # ----------------------
        # Stage 1: Backbone Feature Extraction
        # ----------------------
        
        # Step 1.1: Convert dictionary to tensor
        # Get all view tensors
        view_tensors = list(views_dict.values())  # List of (B, 5, 3, H, W) tensors
        
        # Stack views: (N, B, 5, 3, H, W)
        # Note: The dimension '5' here represents the Neighbour Images (Center + 4 deviations)
        x = torch.stack(view_tensors, dim=0)
        
        # Permute to (B, N, 5, 3, H, W)
        x = x.permute(1, 0, 2, 3, 4, 5)
        
        # Get dimensions
        # NUM_NEIGHBOURS (formerly CROP) represents the 5 neighbour images per view
        B, N, NUM_NEIGHBOURS, C, H, W = x.size()
        
        # Step 1.2: Reshape for parallel processing
        # Shape: (B * N * 5, 3, H, W)
        x = x.reshape(B * N * NUM_NEIGHBOURS, C, H, W)
        
        # Step 1.3: Extract features using backbone
        # Backbone output shape: (B * N * 5, D, 1, 1)
        backbone_feats = self.backbone(x)
        
        # Remove spatial dimensions
        # Shape: (B * N * 5, D)
        backbone_feats = backbone_feats.view(backbone_feats.size(0), -1)
        
        # Step 1.4: Reshape back to original structure
        # Shape: (B, N, 5, D)
        backbone_feats = backbone_feats.view(B, N, NUM_NEIGHBOURS, -1)
        
        # Step 1.5: Apply projection to reduce feature dimension
        # Shape: (B, N, 5, feature_dim)
        backbone_feats = self.projection(backbone_feats)
        
        # ----------------------
        # Stage 2: Intra-View Neighbour Fusion (Key Step)
        # ----------------------
        
        # Step 1.6: Intra-View Max Pooling
        # Aggregate Neighbour Features: Using Max Pooling to fuse the 5 neighbour images into a single view representation.
        # This allows the model to capture the most salient features from the center and its surroundings.
        # Shape: (B, N, feature_dim)
        view_feats = backbone_feats.max(dim=2)[0]
        
        # ----------------------
        # Stage 3: Simple View Aggregation (Ablation)
        # ----------------------
        
        # 放弃复杂的 Set Transformer，直接对 N 个视点进行全局最大池化或平均池化
        # 这里使用 Max Pooling，能最好地保留 3D 物体在各个视角下最显著的激活特征
        # view_feats shape: (B, N, feature_dim)
        global_feat = view_feats.max(dim=1)[0]
        
        # (可选) 如果你希望保留一些平均信息，也可以改为: global_feat = view_feats.mean(dim=1)
        
        # Step 3.2: Apply predictor if needed (Student mode)
        if return_predictor:
            pred_feat = self.predictor(global_feat)
            return global_feat, pred_feat
        else:
            # Teacher mode or test mode
            return global_feat


if __name__ == "__main__":
    # Test the MultiViewVisualEncoder
    encoder = MultiViewVisualEncoder(backbone_type='resnet50', feature_dim=1024)
    
    # Create dummy batch data
    B = 2  # Batch size
    N = 4  # Number of views
    NUM_NEIGHBOURS = 5  # Number of neighbour images per view
    C = 3  # Channels
    H = W = 224  # Height and width
    
    # Create dummy views dictionary
    dummy_views = {}
    for i in range(N):
        view_name = f'view_{i}'
        dummy_views[view_name] = torch.randn(B, NUM_NEIGHBOURS, C, H, W)
    
    dummy_batch = {'views': dummy_views}
    
    # Test Teacher mode (return_predictor=False)
    print("=== Testing Teacher mode ===")
    global_feat = encoder(dummy_batch, return_predictor=False)
    print(f"Global feature shape: {global_feat.shape}")
    print(f"Expected shape: (B, D) = ({B}, 1024)")
    assert global_feat.shape == (B, 1024), "Global feature shape mismatch"
    print("Teacher mode test passed!")
    
    # Test Student mode (return_predictor=True)
    print("\n=== Testing Student mode ===")
    global_feat, pred_feat = encoder(dummy_batch, return_predictor=True)
    print(f"Global feature shape: {global_feat.shape}")
    print(f"Predicted feature shape: {pred_feat.shape}")
    print(f"Expected shapes: (B, D) = ({B}, 1024), (B, D) = ({B}, 1024)")
    assert global_feat.shape == (B, 1024), "Global feature shape mismatch"
    assert pred_feat.shape == (B, 1024), "Predicted feature shape mismatch"
    print("Student mode test passed!")
    print("\nAll tests passed!")
