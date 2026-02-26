import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.hub import load_state_dict_from_url


class SimSiamModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50()
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            map_location="cpu",
        )
        backbone.load_state_dict(state_dict)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
        )
        self.predictor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
        )

    def encode(self, x):
        b, n, m, c, h, w = x.size()
        x = x.view(b * n * m, c, h, w)
        feats = self.backbone(x)
        _, d, fh, fw = feats.size()
        feats = feats.view(b, n, m, d, fh, fw)
        feats = feats.max(dim=2)[0]
        pooled = self.global_pool(feats).flatten(2)
        pooled = pooled.mean(dim=1)
        return pooled

    def forward(self, x1, x2):
        f1 = self.encode(x1)
        f2 = self.encode(x2)
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1, z2

    def forward_encoder(self, x):
        return self.encode(x)
