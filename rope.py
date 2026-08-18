import torch


def cal_rope_freq(head_dim: int, max_seq_len: int, device: str, theta: float = 10000.0):
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")

    freq_theta = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    positions = torch.arange(max_seq_len, device=device)

    freq = torch.outer(positions, freq_theta)
    freq_complex = torch.polar(torch.ones_like(freq), freq)

    return freq_complex


"""
x_complex:      (2,128,8,32)
freq_complex:   (1,128,1,32)
                ----------------
result:         (2,128,8,32)
"""


def apply_rotary_embedding(x, freq_complex):

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    freq_complex = freq_complex[: x.shape[1]]
    freq_complex = freq_complex[None, :, None, :]

    x_rotated = x_complex * freq_complex

    x_out = torch.view_as_real(x_rotated)

    x_out = x_out.reshape_as(x)

    return x_out.type_as(x)
