import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x2 = self.self_attn(x, x, x)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x


class MAEDecoder(nn.Module):
    def __init__(
        self,
        feature_dim=1024,
        max_num_views=64,
        depth=2,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.view_pos_embed = nn.Parameter(torch.zeros(1, max_num_views, feature_dim))
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(feature_dim, nhead, dim_feedforward, dropout) for _ in range(depth)]
        )
        self.pred = nn.Linear(feature_dim, feature_dim)
        self._init_parameters()

    def _init_parameters(self):
        if hasattr(nn.init, "trunc_normal_"):
            nn.init.trunc_normal_(self.view_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            nn.init.normal_(self.view_pos_embed, std=0.02)
            nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, latent_tokens, ids_restore):
        B, L, D = latent_tokens.shape
        N = ids_restore.size(1)
        if L > N:
            raise ValueError(f"latent length {L} exceeds num views {N}")

        num_mask = N - L
        if num_mask > 0:
            mask_tokens = self.mask_token.expand(B, num_mask, D)
            x_ = torch.cat([latent_tokens, mask_tokens], dim=1)
        else:
            x_ = latent_tokens

        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        if N > self.view_pos_embed.size(1):
            raise ValueError(f"num_views={N} exceeds max_num_views={self.view_pos_embed.size(1)}")
        x = x + self.view_pos_embed[:, :N, :]

        for blk in self.blocks:
            x = blk(x)

        x = self.pred(x)
        return x
