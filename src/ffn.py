import torch
import torch.nn as nn
import torch.nn.functional as fun

from src.args import ModelArgs


class FeedForwardNetwork(nn.Module):
    def __init__(self, Param: ModelArgs):
        super().__init__()
        hidden_dim = int(2 * (4 * Param.dims) / 3)

        if Param.ffn_dim_multiplier is not None:
            hidden_dim = int(Param.ffn_dim_multiplier * hidden_dim)

        hidden_dim = Param.multiple_of * (
            (hidden_dim + Param.multiple_of - 1) // Param.multiple_of
        )
        self.w1 = nn.Linear(Param.dims, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, Param.dims, bias=False)
        self.w3 = nn.Linear(Param.dims, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = fun.silu(self.w1(x))
        xv = self.w3(x)
        x = self.w2(swish * xv)
        return x
