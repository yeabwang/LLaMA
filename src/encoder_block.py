from typing import Optional

import torch
import torch.nn as nn
from src.args import ModelArgs
from src.rms_norm import RMSNorm
from src.attn import SelfAttention
from src.ffn import FeedForwardNetwork


class EncoderBlock(nn.Module):
    def __init__(self, Param: ModelArgs):
        super().__init__()
        # self.n_heads = Param.n_heads
        # self.dims = Param.dims
        # self.head_dims = Param.dims // Param.n_heads
        self.attention = SelfAttention(Param)
        self.feed_forward = FeedForwardNetwork(Param)

        self.attention_norm = RMSNorm(Param.dims, Param.norm_eps)
        self.ffn_norm = RMSNorm(Param.dims, Param.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freq_complex: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freq_complex, mask)

        out = h + self.feed_forward(self.ffn_norm(h))

        return out
