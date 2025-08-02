from torch import nn
import torch
from cotracker.models.core.cotracker.blocks import Mlp, AttnBlock, Attention


class TimeSformer(nn.Module):
    def __init__(
            self,
            time_depth = 3,
            space_depth = 3,
            input_dim = 1110,
            hidden_size = 384,
            num_heads = 8,
            mlp_ratio = 4.0
        ):
        super().__init__()
        self.input_transform = torch.nn.Linear(input_dim, hidden_size, bias=True)
        self.time_blks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                )
                for _ in range(time_depth)
            ]
        )
        self.space_blks = nn.ModuleList(
            [
                AttnBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_class=Attention,
                )
                for _ in range(space_depth)
            ]
        )
        self.time_depth = time_depth
        self.space_depth = space_depth
    def forward(self, input_tensor):
        tokens = self.input_transform(input_tensor)
        B, N, S, _ = tokens.shape
        for i in range(self.time_depth):
            tokens = tokens.view(B * N, S, -1)
            time_tokens = self.time_blks[i](tokens).view(B, N, S, -1)
            tokens = tokens.view(B, N, S, -1) + time_tokens
            tokens = tokens.permute(0, 2, 1, 3).reshape(B * S, N, -1)
            tokens = self.space_blks[i](tokens)
        return tokens
