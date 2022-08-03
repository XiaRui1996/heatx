import math
from typing import Any

from flax import linen as nn
import jax.numpy as jnp
import numpy as np


ModuleDef = Any
zero_init = nn.initializers.zeros


def sinusoidal(d_feature, max_len=1024, min_scale=0.5 / math.pi, max_scale=1024.0, dtype=jnp.float32):
    """
    1D Sinusoidal Position Embedding.

    Args:
        max_len: maximum possible length for the input.
        min_scale: float: minimum frequency-scale in sine grating.
        max_scale: float: maximum frequency-scale in sine grating.

    Returns:
        output: `(max_len, d_feature)`
    """
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    return jnp.array(pe, dtype=dtype)


class AttnEncodingLayer(nn.Module):
    features: int
    num_heads: int
    dtype: Any = jnp.float32
    norm: ModuleDef = nn.LayerNorm
    max_len: int = 1024

    def setup(self):
        self.attn = nn.SelfAttention(
            num_heads=self.num_heads, dtype=self.dtype, deterministic=True)
        self.mlp = nn.Sequential([
            nn.Dense(self.features * 4, dtype=self.dtype),
            nn.silu,
            nn.Dense(self.features, dtype=self.dtype),
            nn.LayerNorm(dtype=self.dtype)
        ])
        self.proj = nn.Sequential([
            nn.silu,
            nn.Dense(self.features, kernel_init=zero_init, dtype=self.dtype)
        ])
        self.normalize = self.norm(dtype=self.dtype)
        self.pe = sinusoidal(self.features, max_len=self.max_len, dtype=self.dtype)

    def __call__(self, x):
        '''
        x: (B, L, D) array.
        '''
        h = self.mlp(x + self.pe[:x.shape[-2]])
        h = self.proj(self.attn(h))
        return self.normalize(x + h)
