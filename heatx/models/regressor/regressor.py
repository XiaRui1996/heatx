from typing import Any, Callable
from functools import partial

from flax import linen as nn
import jax
import jax.numpy as jnp

from heatx.models.encoder import ResNet18, CondResNet18
from heatx.models.nnutil import SkipConnCondGatedUnit


ModuleDef = Any


class ImageRegressor(nn.Module):
    targets: int
    encoder_cls: ModuleDef = ResNet18
    dtype: Any = jnp.float32
    fourier_min: int = 5
    fourier_max: int = 8
    act: Callable = None

    def setup(self):
        self.encoder = self.encoder_cls(dtype=self.dtype)
        self.proj = nn.Dense(self.targets, dtype=self.dtype)

        self.fourier_freqs = 2.0 * jnp.pi * \
            (2.0 ** jnp.arange(self.fourier_min, self.fourier_max + 1, dtype=self.dtype))

    def __call__(self, x):
        prefix = x.shape[:-3]
        x = jnp.reshape(x, (-1,) + x.shape[-3:])

        ff = jnp.reshape(x[..., jnp.newaxis] * self.fourier_freqs, x.shape[:-1] + (-1,))
        x = jnp.concatenate([x, jnp.sin(ff), jnp.cos(ff)], axis=-1)

        h = self.encoder(x)
        h = jnp.mean(h, axis=(-2, -3))
        y = self.proj(h)

        y = jnp.reshape(y, prefix + y.shape[1:])

        if self.act is not None:
            y = self.act(y)

        return y


class CondImageRegressor(nn.Module):
    targets: int
    encoder_cls: ModuleDef = CondResNet18
    dtype: Any = jnp.float32
    fourier_min: int = 5
    fourier_max: int = 8
    act: Callable = None

    def setup(self):
        self.encoder = self.encoder_cls(dtype=self.dtype)
        self.gated = SkipConnCondGatedUnit(256, partial(nn.GroupNorm, 32), dtype=self.dtype)
        self.proj = nn.Dense(self.targets, dtype=self.dtype)

        self.x_fourier_freqs = 2.0 * jnp.pi * \
            (2.0 ** jnp.arange(self.fourier_min, self.fourier_max + 1, dtype=self.dtype))
        
        self.t_fourier_freqs = 2.0 * jnp.pi * \
            jax.random.normal(jax.random.PRNGKey(42), (255,)) * 16.0

    def __call__(self, x, t):
        '''
        Args:
          x: (..., H, W, C) array.
          t: (...) array.
        '''
        prefix = jnp.broadcast_shapes(t.shape, x.shape[:-3])
        t = jnp.broadcast_to(t, prefix)
        x = jnp.broadcast_to(x, prefix + x.shape[-3:])
        t = jnp.reshape(t, (-1, 1))
        x = jnp.reshape(x, (-1,) + x.shape[-3:])

        xff = jnp.reshape(x[..., jnp.newaxis] * self.x_fourier_freqs, x.shape[:-1] + (-1,))
        x = jnp.concatenate([x, jnp.sin(xff), jnp.cos(xff)], axis=-1)

        tff = jnp.reshape(t[..., jnp.newaxis] * self.t_fourier_freqs, t.shape[:-1] + (-1,))
        t = jnp.concatenate([t, jnp.exp(-t), jnp.sin(tff), jnp.cos(tff)], axis=-1)


        h = self.encoder(x, t)
        h = jnp.mean(h, axis=(-2, -3))
        h = self.gated(h, t)
        y = self.proj(h)

        y = jnp.reshape(y, prefix + y.shape[1:])

        if self.act is not None:
            y = self.act(y)

        return y
