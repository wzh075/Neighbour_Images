import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.hub import load_state_dict_from_url


class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj):
        h = self.linear(x)
        h = torch.matmul(adj, h)
        return h


class ViewGCNModel(nn.Module):
    def __init__(self, graph_type='full'):
        super().__init__()
        backbone = resnet50()
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            map_location="cpu",
        )
        backbone.load_state_dict(state_dict)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.gcn1 = GCNConv(2048, 2048)
        self.gcn2 = GCNConv(2048, 2048)
        self.act = nn.ReLU(inplace=True)
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
        )
        self.graph_type = graph_type

    def build_adj(self, n, device):
        if self.graph_type == 'full':
            adj = torch.ones(n, n, device=device)
        else:
            adj = torch.zeros(n, n, device=device)
            for i in range(n):
                adj[i, i] = 1.0
                if i - 1 >= 0:
                    adj[i, i - 1] = 1.0
                if i + 1 < n:
                    adj[i, i + 1] = 1.0
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        adj_norm = deg_inv_sqrt.view(n, 1) * adj * deg_inv_sqrt.view(1, n)
        return adj_norm

    def forward(self, x, return_projector=True):
        b, n, m, c, h, w = x.size()
        x = x.view(b * n * m, c, h, w)
        feats = self.backbone(x).flatten(1)
        feats = feats.view(b, n, m, -1)
        view_feats = feats.max(dim=2)[0]
        adj = self.build_adj(n, view_feats.device)
        h = self.act(self.gcn1(view_feats, adj))
        h = self.gcn2(h, adj)
        global_feat = h.mean(dim=1)
        if not return_projector:
            return global_feat
        z = self.projector(global_feat)
        return global_feat, z
