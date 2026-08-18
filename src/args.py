from dataclasses import dataclass
from typing import Optional


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
