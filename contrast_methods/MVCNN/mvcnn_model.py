import torch.nn as nn
from torchvision.models import resnet50
from torch.hub import load_state_dict_from_url


class MVCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50()
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            map_location="cpu",
        )
        backbone.load_state_dict(state_dict)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
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
        view_feats = feats.max(dim=2)[0]
        global_feat = view_feats.max(dim=1)[0]
        if not return_projector:
            return global_feat
        z = self.projector(global_feat)
        return global_feat, z
