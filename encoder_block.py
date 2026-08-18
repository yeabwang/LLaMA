import torch
import torch.nn as nn
from model import ModelArgs
from rms_norm import RMSNorm
from attn import SelfAttention
from ffn import FeedForwardNetwork


class EncoderBlock(nn.Module):
    def __init__(self, Param: ModelArgs):
        super().__init__()
        # self.n_heads = Param.n_heads
        # self.dims = Param.dims
        # self.head_dims = Param.dims // Param.n_heads
        self.attention = SelfAttention(Param)
        self.ffn = FeedForwardNetwork(Param)

        self.attn_norm = RMSNorm(Param.dims, Param.norm_eps)
        self.ffn_norm = RMSNorm(Param.dims, Param.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        h = x + self.attention(self.attn_norm(x), start_pos, freq_complex)

        out = h + self.ffn(self.ffn_norm(h))

        return out
