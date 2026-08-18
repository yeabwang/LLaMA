import torch
import torch.nn as nn
from src.args import ModelArgs
from src.encoder_block import EncoderBlock
from src.rms_norm import RMSNorm
from src.rope import cal_rope_freq


class Transformer(nn.Module):
    def __init__(self, Param: ModelArgs):
        super().__init__()

        if Param.vocab_size <= 0:
            raise ValueError(f"{Param.vocab_size} must be greater than zero")

        self.args = Param
        self.vocab_size = Param.vocab_size
        self.n_layers = Param.n_layers
        self.tok_embeddings: nn.Embedding = nn.Embedding(self.vocab_size, Param.dims)
        self.layers = nn.ModuleList()

        for _ in range(Param.n_layers):
            self.layers.append(EncoderBlock(Param))

        self.norm = RMSNorm(Param.dims, eps=Param.norm_eps)
        self.output = nn.Linear(Param.dims, self.vocab_size, bias=False)

        # buffer so it follows .to(device)/.cuda() with the rest of the model
        self.register_buffer(
            "rope_theta",
            cal_rope_freq(
                Param.dims // Param.n_heads,
                Param.max_seq_len * 2,
                Param.device,
            ),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        _, seq_len = tokens.shape

        h = self.tok_embeddings(
            tokens
        )  # batch_size, seq_len -> batch_size, seq_len, emb_dim
        get_pos = self.rope_theta[start_pos : start_pos + seq_len]

        # causal mask; cached positions [0, start_pos) are always visible
        mask = None
        if seq_len > 1:
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=tokens.device, dtype=h.dtype
            )
            mask = torch.triu(mask, diagonal=1)
            mask = torch.cat(
                [
                    torch.zeros(
                        (seq_len, start_pos), device=tokens.device, dtype=h.dtype
                    ),
                    mask,
                ],
                dim=-1,
            )

        for layer in self.layers:
            h = layer(h, start_pos, get_pos, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
