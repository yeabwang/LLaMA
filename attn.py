import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as fun
from args import ModelArgs
from rope import apply_rotary_embedding


class SelfAttention(nn.Module):
    # Doesn't have parallalization here
    def __init__(self, Param: ModelArgs):
        super().__init__()

        self.n_kv_heads = (
            Param.n_heads if Param.n_kv_heads is None else Param.n_kv_heads
        )
        self.n_q_heads = Param.n_heads

        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError("Incompatiable head counts")
        if Param.dims % Param.n_heads != 0:
            raise ValueError("dims must be divisible by n_heads")

        # indicate the num times each kv head is shared by query heads
        self.h_ratio = self.n_q_heads // self.n_kv_heads
        self.head_dim = Param.dims // Param.n_heads

        self.wq = nn.Linear(Param.dims, Param.n_heads * self.head_dim, bias=False)

        self.wkv = nn.Linear(
            Param.dims, self.n_kv_heads * self.head_dim * 2, bias=False
        )

        self.out = nn.Linear(Param.n_heads * self.head_dim, Param.dims, bias=False)

        self.register_buffer(
            "k_cache",
            torch.zeros(
                (
                    Param.max_batch_size,
                    Param.max_seq_len,
                    self.n_kv_heads,
                    self.head_dim,
                )
            ),
            persistent=False,
        )

        self.register_buffer(
            "v_cache",
            torch.zeros(
                (
                    Param.max_batch_size,
                    Param.max_seq_len,
                    self.n_kv_heads,
                    self.head_dim,
                )
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freq_complex: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):

        batch_size, seq_len, _ = x.shape

        xkv = self.wkv(x)
        xq = self.wq(x)

        xkv = xkv.view(batch_size, seq_len, 2, self.n_kv_heads, self.head_dim)

        xq = xq.view(batch_size, seq_len, self.n_q_heads, self.head_dim)

        # split key and value heads
        xk, xv = xkv.unbind(dim=2)

        xq_rotated = apply_rotary_embedding(xq, freq_complex)

        xk_rotated = apply_rotary_embedding(xk, freq_complex)

        # replacing cache for new entry
        self.k_cache[:batch_size, start_pos : start_pos + seq_len] = xk_rotated

        self.v_cache[:batch_size, start_pos : start_pos + seq_len] = xv

        # retrieve cached keys and vals
        cached_k = self.k_cache[:batch_size, 0 : start_pos + seq_len]

        cached_v = self.v_cache[:batch_size, 0 : start_pos + seq_len]

        # Q has n_q_heads while KV has n_kv_heads
        # instead of repeating KV heads, group query heads
        # and compute attention without expanding KV cache

        xq = xq_rotated.transpose(1, 2)
        # (batch, n_q_heads, seq_len, head_dim)

        keys = cached_k.transpose(1, 2)
        vals = cached_v.transpose(1, 2)
        # (batch, n_kv_heads, seq_len, head_dim)

        # group query heads belonging to each KV head
        xq = xq.reshape(
            batch_size, self.n_kv_heads, self.h_ratio, seq_len, self.head_dim
        )

        # add dimension for broadcasting with grouped queries
        keys = keys.unsqueeze(2)
        vals = vals.unsqueeze(2)

        score = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # score:
        # (batch, n_kv_heads, h_ratio, seq_len, cached_seq_len)

        if mask is not None:
            score = score + mask

        score = fun.softmax(score.float(), dim=-1).type_as(xq)

        out = torch.matmul(score, vals)

        # (batch, n_kv_heads, h_ratio, seq_len, head_dim)

        out = out.reshape(batch_size, self.n_q_heads, seq_len, self.head_dim)

        out = out.transpose(1, 2).contiguous()

        out = out.view(batch_size, seq_len, -1)

        return self.out(out)
