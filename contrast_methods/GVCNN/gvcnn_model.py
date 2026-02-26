import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.hub import load_state_dict_from_url


class GVCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50()
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            map_location="cpu",
        )
        backbone.load_state_dict(state_dict)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.group_weight = nn.Linear(2048, 1)
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
        )

    def forward(self, x, return_projector=True):
        b, n, m, c, h, w = x.size()
        x = x.view(b * n * m, c, h, w)
        feats = self.backbone(x).flatten(1)
        feats = feats.view(b, n, m, -1)
        weights = self.group_weight(feats).squeeze(-1)
        weights = torch.softmax(weights, dim=2)
        view_feats = (feats * weights.unsqueeze(-1)).sum(dim=2)
        global_feat = view_feats.max(dim=1)[0]
        if not return_projector:
            return global_feat
        z = self.projector(global_feat)
        return global_feat, z
