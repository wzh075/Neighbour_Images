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
       - Stage 3 (MAE Encoder): 将视点特征视为序列 Token，经位置编码后送入 Transformer Encoder（SAB 堆叠）。
    """
    def __init__(
        self,
        backbone_type='resnet50',
        feature_dim=1024,
        max_num_views=64,
        encoder_depth=2,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        freeze_backbone=False,
    ):
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
        
        # Step 2: Projection layer (Upgrade to BYOL standard MLP Projector)
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, feature_dim)
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.view_pos_embed = nn.Parameter(torch.zeros(1, max_num_views, feature_dim))

        self.encoder_blocks = nn.ModuleList(
            [SelfAttentionBlock(feature_dim, nhead, dim_feedforward, dropout) for _ in range(encoder_depth)]
        )

        self._init_parameters()
        
        # Step 5: Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"ResNet Backbone parameters have been frozen. Only projection layers and MAE encoder will be trained.")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
            print(f"All parameters (including ResNet Backbone) will be trained.")

    def _init_parameters(self):
        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(self.view_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            nn.init.normal_(self.view_pos_embed, std=0.02)
            nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _sort_view_keys(view_keys):
        def parse_view_index(k):
            if isinstance(k, str) and '_' in k:
                suffix = k.rsplit('_', 1)[-1]
                if suffix.isdigit():
                    return int(suffix)
            return k

        return sorted(view_keys, key=parse_view_index)

    @staticmethod
    def _random_masking(x, mask_ratio):
        B, N, D = x.shape
        device = x.device

        if mask_ratio <= 0:
            ids_restore = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
            mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            return x, mask, ids_restore

        len_keep = int(N * (1 - mask_ratio))
        len_keep = max(1, min(N, len_keep))

        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        return x_visible, mask, ids_restore

    def forward(self, batch, mask_ratio=0.75):
        views_dict = batch['views']
        
        # ----------------------
        # Stage 1: Backbone Feature Extraction
        # ----------------------
        
        view_keys = self._sort_view_keys(list(views_dict.keys()))
        view_tensors = [views_dict[k] for k in view_keys]
        
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

        target = view_feats

        if N > self.view_pos_embed.size(1):
            raise ValueError(f"num_views={N} exceeds max_num_views={self.view_pos_embed.size(1)}")

        x = view_feats + self.view_pos_embed[:, :N, :]
        x_visible, mask, ids_restore = self._random_masking(x, mask_ratio=mask_ratio)

        for blk in self.encoder_blocks:
            x_visible = blk(x_visible)

        return x_visible, mask, ids_restore, target


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
    
    print("=== Testing MAE encoder (no mask) ===")
    latent, mask, ids_restore, target = encoder(dummy_batch, mask_ratio=0.0)
    print(f"Latent shape: {latent.shape}")
    print(f"Target shape: {target.shape}")
    assert latent.shape == (B, N, 1024), "Latent shape mismatch when mask_ratio=0"
    assert mask.shape == (B, N), "Mask shape mismatch"
    assert ids_restore.shape == (B, N), "ids_restore shape mismatch"
    assert target.shape == (B, N, 1024), "Target shape mismatch"

    print("\n=== Testing MAE encoder (with mask) ===")
    latent, mask, ids_restore, target = encoder(dummy_batch, mask_ratio=0.5)
    print(f"Latent shape: {latent.shape}")
    num_visible = latent.size(1)
    assert 1 <= num_visible <= N, "Invalid number of visible tokens"
    assert mask.shape == (B, N), "Mask shape mismatch"
    assert ids_restore.shape == (B, N), "ids_restore shape mismatch"
    assert target.shape == (B, N, 1024), "Target shape mismatch"

    print("\nAll tests passed!")
