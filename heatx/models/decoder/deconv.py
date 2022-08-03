"""
Flax implementation of ResNet V1.
Adapted from:
  https://github.com/google/flax/blob/main/examples/imagenet/models.py
"""
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import jax
from flax import linen as nn
import jax.numpy as jnp

from heatx.models.nnutil import SkipConnCondGatedUnit


ModuleDef = Any


class ResNetBlockT(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    act_out: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (3, 3),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act_out(residual + y)


class BottleneckResNetBlockT(nn.Module):
    """Bottleneck ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    act_out: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (3, 3),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act_out(residual + y)


class ResNetT(nn.Module):
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_channels: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.silu
    conv: ModuleDef = nn.ConvTranspose

    @nn.compact
    def __call__(self, x, z):
        '''
        Args:
          x: (B, H, W, C) array.
          z: (B, d) array.
        '''
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.GroupNorm, num_groups=32, dtype=self.dtype)
        gated = partial(SkipConnCondGatedUnit, norm=norm, dtype=self.dtype)
        num_stages = len(self.stage_sizes)

        z = z[..., jnp.newaxis, jnp.newaxis, :]

        for i, block_size in enumerate(reversed(self.stage_sizes)):
            for j in range(block_size):
                strides = (2, 2) if i < (num_stages - 1) and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * (2 ** (num_stages - i - 1)),
                                   strides=strides, conv=conv, norm=norm,
                                   act=self.act, act_out=self.act)(x)
                x = gated(x.shape[-1])(x, z)

        x = self.block_cls(self.num_filters,
                           strides=(2, 2), conv=conv,
                           norm=norm, act=self.act)(x)
        x = gated(x.shape[-1])(x, z)

        x = jnp.concatenate([
            x, jnp.broadcast_to(z, x.shape[:-1] + (z.shape[-1],))
        ], axis=-1)

        x = conv(self.num_channels, (7, 7),
                 strides=(2, 2),
                 name='conv_out')(x)

        return jnp.asarray(x, self.dtype)


ResNetT18 = partial(ResNetT, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlockT)
ResNetT34 = partial(ResNetT, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlockT)
ResNetT50 = partial(ResNetT, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlockT)
ResNetT101 = partial(ResNetT, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlockT)
