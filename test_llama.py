"""ponytail: one runnable check, asserts only."""

import torch
import torch.nn.functional as fun

from args import ModelArgs
from model import Transformer
from attn import SelfAttention
from ffn import FeedForwardNetwork
from rms_norm import RMSNorm
from rope import apply_rotary_embedding, cal_rope_freq


def args(**kw):
    base = dict(
        dims=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=17,
        max_batch_size=2,
        max_seq_len=16,
        multiple_of=8,
    )
    base.update(kw)
    return ModelArgs(**base)


def test_rms_norm():
    x = torch.randn(2, 3, 8)
    out = RMSNorm(8)(x)
    assert out.shape == x.shape
    assert torch.allclose(out.pow(2).mean(-1), torch.ones(2, 3), atol=1e-3)


def test_rope_preserves_norm_and_rotates():
    freqs = cal_rope_freq(8, 16, "cpu")
    x = torch.randn(2, 5, 4, 8)
    out = apply_rotary_embedding(x, freqs[:5])
    assert out.shape == x.shape
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4)
    # position 0 is a zero rotation, later positions are not
    assert torch.allclose(out[:, 0], x[:, 0], atol=1e-5)
    assert not torch.allclose(out[:, 1], x[:, 1], atol=1e-3)


def test_attention_is_causal_and_cache_matches_full_pass():
    torch.manual_seed(0)
    p = args()
    attn = SelfAttention(p).eval()
    freqs = cal_rope_freq(p.dims // p.n_heads, p.max_seq_len, "cpu")
    seq = 6
    x = torch.randn(1, seq, p.dims)
    mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)

    with torch.no_grad():
        full = attn(x, 0, freqs[:seq], mask)
    assert full.shape == x.shape

    # causality: changing the last token must not change earlier outputs
    x2 = x.clone()
    x2[:, -1] = torch.randn(p.dims)
    with torch.no_grad():
        attn.k_cache.zero_()
        attn.v_cache.zero_()
        full2 = attn(x2, 0, freqs[:seq], mask)
    assert torch.allclose(full[:, :-1], full2[:, :-1], atol=1e-5)

    # incremental decoding through the KV cache reproduces the full pass
    attn.k_cache.zero_()
    attn.v_cache.zero_()
    steps = []
    with torch.no_grad():
        for i in range(seq):
            steps.append(attn(x[:, i : i + 1], i, freqs[i : i + 1], None))
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-4)


def test_ffn_swiglu():
    torch.manual_seed(0)
    p = args()
    ffn = FeedForwardNetwork(p).eval()

    # hidden_dim = round_up(int(2 * 4 * dims / 3), multiple_of) = round_up(85, 8) = 88
    assert ffn.w1.out_features == 88
    assert ffn.w3.out_features == 88
    assert ffn.w2.in_features == 88
    assert ffn.w2.out_features == p.dims

    x = torch.randn(2, 3, p.dims)
    with torch.no_grad():
        out = ffn(x)
    assert out.shape == x.shape

    # matches the SwiGLU definition: w2(silu(w1 x) * w3 x)
    with torch.no_grad():
        expected = ffn.w2(fun.silu(ffn.w1(x)) * ffn.w3(x))
    assert torch.allclose(out, expected, atol=1e-6)

    # gated -> not a linear map, so f(2x) != 2 f(x)
    with torch.no_grad():
        assert not torch.allclose(ffn(2 * x), 2 * out, atol=1e-3)

    # ffn_dim_multiplier scales the hidden dim, still rounded to multiple_of
    wide = FeedForwardNetwork(args(ffn_dim_multiplier=2.0))
    assert wide.w1.out_features == 176


def test_transformer_forward():
    torch.manual_seed(0)
    model = Transformer(args()).eval()

    tokens = torch.randint(0, 17, (2, 5))
    with torch.no_grad():
        out = model(tokens)
    assert out.shape == (2, 5, 17)
    assert torch.isfinite(out).all()

    with torch.no_grad():  # single-token decode step, mask is None
        assert model(tokens[:, :1], 0).shape == (2, 1, 17)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
