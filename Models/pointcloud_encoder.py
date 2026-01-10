import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Transform Network (T-Net) for point cloud data"""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        # Shared MLP layers
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Identity initialization
        self.identity = nn.Parameter(torch.eye(k).flatten(), requires_grad=False)
    
    def forward(self, x):
        # x: (B, C, N) where C=3, N=2048
        B = x.size(0)
        
        # Shared MLP layers
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(B, -1)
        
        # Fully connected layers
        x = self.activation(self.bn4(self.fc1(x)))
        x = self.activation(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Add identity transform
        x = x + self.identity
        x = x.view(B, self.k, self.k)
        
        return x


class PointNetEncoder(nn.Module):
    """PointNet Encoder for point cloud feature extraction"""
    def __init__(self, feature_dim=1024, dropout=0.1):
        super(PointNetEncoder, self).__init__()
        
        # Input transformation network
        self.input_transform = TNet(k=3)
        
        # Feature transformation network
        self.feature_transform = TNet(k=64)
        
        # MLP layers for feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, feature_dim, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(feature_dim)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Feature dimension
        self.feature_dim = feature_dim
    
    def forward(self, pointcloud):
        # pointcloud: (B, N, C) where N=2048, C=3
        B, N, C = pointcloud.size()
        
        # ----------------------
        # Stage 1: Point-wise Feature Extraction
        # ----------------------
        
        # Step 1.1: Apply input transformation
        x = pointcloud.permute(0, 2, 1)  # (B, C, N)
        input_transform = self.input_transform(x)
        x = torch.bmm(input_transform, x)
        
        # Step 1.2: First MLP block
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        
        # Step 1.3: Apply feature transformation
        feature_transform = self.feature_transform(x)
        point_features = torch.bmm(feature_transform, x)  # (B, 64, N)
        
        # Step 1.4: Second MLP block
        x = self.activation(self.bn3(self.conv3(point_features)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.activation(self.bn5(self.conv5(x)))
        
        # ----------------------
        # Stage 2: Global Feature Aggregation
        # ----------------------
        
        # Step 2.1: Global max pooling
        global_feature = torch.max(x, 2, keepdim=True)[0]  # (B, D, 1)
        global_feature = global_feature.view(B, -1)  # (B, D)
        
        # ----------------------
        # Stage 3: Output
        # ----------------------
        
        # Reshape point features back to (B, N, D)
        point_features = point_features.permute(0, 2, 1)  # (B, N, 64)
        
        return point_features, global_feature


class PointCloudEncoder(nn.Module):
    """Point Cloud Encoder for 3D object feature extraction"""
    def __init__(self, feature_dim=1024, dropout=0.1):
        super(PointCloudEncoder, self).__init__()
        
        # PointNet encoder
        self.pointnet_encoder = PointNetEncoder(feature_dim, dropout)
    
    def forward(self, batch):
        # Input: batch containing 'pointcloud' tensor
        pointcloud = batch['pointcloud']
        
        # ----------------------
        # Stage 1: PointNet Feature Extraction
        # ----------------------
        
        # Step 1.1: Check pointcloud shape
        # Expected shape: (B, N, 3) where N=2048
        if pointcloud.dim() != 3 or pointcloud.size(2) != 3:
            raise ValueError(f"Expected pointcloud shape (B, N, 3), got {pointcloud.shape}")
        
        # Step 1.2: Extract features using PointNet
        point_features, global_point_feat = self.pointnet_encoder(pointcloud)
        
        return point_features, global_point_feat


if __name__ == "__main__":
    # Test the PointCloudEncoder
    encoder = PointCloudEncoder(feature_dim=1024)
    
    # Create dummy batch data
    B = 2  # Batch size
    N = 2048  # Number of points per sample
    C = 3  # Channels (x, y, z)
    
    # Create dummy pointcloud
    pointcloud = torch.randn(B, N, C)
    
    # Create dummy batch
    dummy_batch = {
        'pointcloud': pointcloud
    }
    
    # Forward pass
    point_features, global_point_feat = encoder(dummy_batch)
    
    # Print output shapes
    print(f"Point features shape: {point_features.shape}")  # Expected: (B, N, 64)
    print(f"Global point feature shape: {global_point_feat.shape}")  # Expected: (B, 1024)
    
    print("PointCloudEncoder test completed successfully!")