import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torch.hub import load_state_dict_from_url


class MoCoV2Model(nn.Module):
    def __init__(self, queue_size=4096, feature_dim=128, momentum=0.999, temperature=0.07):
        super().__init__()
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.temperature = temperature

        self.encoder_q = self._build_encoder()
        self.encoder_k = self._build_encoder()
        self._init_key_encoder()

        self.register_buffer("queue", F.normalize(torch.randn(queue_size, feature_dim), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def _build_encoder():
        backbone = resnet50()
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/resnet50-0676ba61.pth",
            map_location="cpu",
        )
        backbone.load_state_dict(state_dict)
        encoder = nn.Sequential(*list(backbone.children())[:-2])
        projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128),
        )
        return nn.ModuleDict(
            {
                "backbone": encoder,
                "pool": nn.AdaptiveAvgPool2d((1, 1)),
                "projector": projector,
            }
        )

    def _init_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if batch_size >= self.queue_size:
            self.queue.copy_(keys[-self.queue_size:])
            self.queue_ptr[0] = 0
            return

        end_ptr = ptr + batch_size
        if end_ptr <= self.queue_size:
            self.queue[ptr:end_ptr] = keys
        else:
            first = self.queue_size - ptr
            self.queue[ptr:] = keys[:first]
            self.queue[: end_ptr - self.queue_size] = keys[first:]
        self.queue_ptr[0] = end_ptr % self.queue_size

    def _encode(self, encoder, x, use_projector=True):
        b, n, m, c, h, w = x.size()
        x = x.view(b * n * m, c, h, w)
        feats = encoder["backbone"](x)
        _, d, fh, fw = feats.size()
        feats = feats.view(b, n, m, d, fh, fw)
        feats = feats.max(dim=2)[0]
        pooled = encoder["pool"](feats).flatten(2)
        pooled = pooled.mean(dim=1)
        if not use_projector:
            return pooled
        z = encoder["projector"](pooled)
        z = F.normalize(z, dim=1)
        return z

    def forward(self, x_q, x_k):
        q = self._encode(self.encoder_q, x_q, use_projector=True)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self._encode(self.encoder_k, x_k, use_projector=True)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,kc->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        self._dequeue_and_enqueue(k)
        return logits, labels

    def forward_encoder(self, x):
        return self._encode(self.encoder_q, x, use_projector=False)
