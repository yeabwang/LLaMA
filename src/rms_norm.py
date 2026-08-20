import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    # implmenting (xi / rmsn(xi)) * gamma where rmsn(xi) = sqrt(mean(xi**2))
    def __init__(self, dims, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))  # gamma
        self.eps = eps

    def forward(self, x: torch.Tensor):
        x_float = x.float()

        rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)

        output = x_float * rms

        return self.weight * output.type_as(x)
