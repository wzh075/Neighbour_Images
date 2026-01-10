import torch
import torch.nn as nn
from torchvision.models import resnet50


class SelfAttentionBlock(nn.Module):
    """Self-Attention Block (SAB) for inter-view feature interaction"""
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
    """Multi-View Visual Encoder for feature extraction"""
    def __init__(self, backbone_type='resnet50', feature_dim=1024, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(MultiViewVisualEncoder, self).__init__()
        
        # Step 1: Initialize backbone
        if backbone_type == 'resnet50':
            self.backbone = resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Remove classification head (fc layer)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Step 2: Projection layer to reduce feature dim from 2048 to 1024
        self.projection = nn.Linear(2048, feature_dim)
        
        # Step 3: Set Transformer for aggregation
        self.set_transformer = SetTransformerAggregation(feature_dim, nhead, dim_feedforward, dropout)

    def forward(self, batch):
        # Input: batch containing 'views' dictionary
        views_dict = batch['views']
        
        # ----------------------
        # Stage 1: Backbone Feature Extraction
        # ----------------------
        
        # Step 1.1: Convert dictionary to tensor
        # Get all view tensors
        view_tensors = list(views_dict.values())  # List of (B, 5, 3, H, W) tensors
        
        # Stack views: (N, B, 5, 3, H, W)
        x = torch.stack(view_tensors, dim=0)
        
        # Permute to (B, N, 5, 3, H, W)
        x = x.permute(1, 0, 2, 3, 4, 5)
        
        # Get dimensions
        B, N, CROP, C, H, W = x.size()
        
        # Step 1.2: Reshape for parallel processing
        # Shape: (B * N * 5, 3, H, W)
        x = x.reshape(B * N * CROP, C, H, W)
        
        # Step 1.3: Extract features using backbone
        # Backbone output shape: (B * N * 5, D, 1, 1)
        backbone_feats = self.backbone(x)
        
        # Remove spatial dimensions
        # Shape: (B * N * 5, D)
        backbone_feats = backbone_feats.view(backbone_feats.size(0), -1)
        
        # Step 1.4: Reshape back to original structure
        # Shape: (B, N, 5, D)
        backbone_feats = backbone_feats.view(B, N, CROP, -1)
        
        # Step 1.5: Apply projection to reduce feature dimension
        # Shape: (B, N, 5, feature_dim)
        backbone_feats = self.projection(backbone_feats)
        
        # Step 1.6: Intra-View Max Pooling
        # Max pool over 5 crops
        # Shape: (B, N, feature_dim)
        view_feats = backbone_feats.max(dim=2)[0]
        
        # ----------------------
        # Stage 2: Set Transformer Aggregation
        # ----------------------
        
        # Step 2.1: Inter-view interaction and aggregation
        refined_view_feats, aggregated_feat = self.set_transformer(view_feats)
        
        # ----------------------
        # Stage 3: Output
        # ----------------------
        
        # Step 3.1: Squeeze aggregated feature
        # Shape: (B, D)
        global_image_feat = aggregated_feat.squeeze(1)
        
        return refined_view_feats, global_image_feat


if __name__ == "__main__":
    # Test the MultiViewVisualEncoder
    encoder = MultiViewVisualEncoder(backbone_type='resnet50', feature_dim=2048)
    
    # Create dummy batch data
    B = 2  # Batch size
    N = 4  # Number of views
    CROP = 5  # Number of crops per view
    C = 3  # Channels
    H = W = 224  # Height and width
    
    # Create dummy views dictionary
    dummy_views = {}
    for i in range(N):
        view_name = f'view_{i}'
        dummy_views[view_name] = torch.randn(B, CROP, C, H, W)
    
    dummy_batch = {'views': dummy_views}
    
    # Forward pass
    refined_view_feats, global_image_feat = encoder(dummy_batch)
    
    print("=== MultiViewVisualEncoder Test Results ===")
    print(f"Refined view features shape: {refined_view_feats.shape}")
    print(f"Global image feature shape: {global_image_feat.shape}")
    print(f"Expected shapes: (B, N, D) = ({B}, {N}, 1024) and (B, D) = ({B}, 1024)")
    
    # Check if shapes match expectations
    assert refined_view_feats.shape == (B, N, 1024), "Refined view features shape mismatch"
    assert global_image_feat.shape == (B, 1024), "Global image feature shape mismatch"
    print("All shape checks passed!")
