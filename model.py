import torch
import torch.nn as nn
import torch.nn.functional as fun
import math
from dataclasses import dataclass
from typing import Optional

from encoder_block import EncoderBlock
from rms_norm import RMSNorm
from rope import cal_rope_theta


@dataclass
class ModelArgs:
    dims: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    # will be used to increase back the size of the parameters which was reduced
    # as result of the reduced heads as of kv cahce
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = "cpu"


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        if args.vocab_size > 0:
            self.args = args
            self.vocab_size = args.vocab_size
            self.n_layers = args.n_layers
            self.tok_embeddings: nn.Embedding = nn.Embedding(self.vocab_size, args.dims)
            self.layers = nn.ModuleList()

            for _ in range(args.n_layers):
                self.layers.append(EncoderBlock(args))

            self.norm = RMSNorm(args.dims, eps=args.norm_eps)
            self.output = nn.Linear(args.dims, self.vocab_size, bias=False)
            self.rope_theta = cal_rope_theta(
                self.args.dims // self.args.n_heads,
                self.args.max_seq_len * 2,
                device=self.args.device,
            )

        else:
            print(f"{args.vocab_size} must be greater than zero")

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _, seq_len = tokens.shape

        h = self.tok_embeddings(
            tokens
        )  # batch_size, seq_len -> batch_size, seq_len, emb_dim
        get_pos = self.rope_theta[start_pos : start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, get_pos)
        h = self.norm
        output = self.output(h).float()
        return output
