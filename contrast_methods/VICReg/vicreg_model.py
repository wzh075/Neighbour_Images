import torch
import torch.nn as nn

from Models.multi_view_visual_encoder import MultiViewVisualEncoder


class VICRegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MultiViewVisualEncoder(feature_dim=1024)
        self.projector = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
        )

    def forward(self, batch, use_projection=True):
        feats = self.encoder(batch, return_predictor=False)
        if not use_projection:
            return feats
        z = self.projector(feats)
        return z
