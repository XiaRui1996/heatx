"""
Flax implementation of ResNet V1.
Adapted from:
  https://github.com/google/flax/blob/main/examples/imagenet/models.py
"""
from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

from heatx.models.nnutil import SkipConnCondGatedUnit


ModuleDef = Any
Identity = lambda x: x



class ResNetBlock(nn.Module):
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
            residual = self.conv(self.filters, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act_out(residual + y)


class BottleneckResNetBlock(nn.Module):
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
            residual = self.conv(self.filters * 4, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act_out(residual + y)


class ResNet(nn.Module):
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.silu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.GroupNorm, num_groups=32)

        x = conv(self.num_filters, (7, 7),
                 strides=(2, 2),
                 padding=[(3, 3), (3, 3)],
                 name='conv_init')(x)
        x = norm(name='norm_init')(x)
        x = self.act(x)
        x = nn.avg_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=strides, conv=conv, norm=norm,
                                   act=self.act, act_out=self.act)(x)
        return jnp.asarray(x, self.dtype)


class CondResNet(nn.Module):
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.silu
    conv: ModuleDef = nn.Conv

    @nn.compact
    def __call__(self, x, z):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.GroupNorm, num_groups=32)
        gated = partial(SkipConnCondGatedUnit, norm=norm, dtype=self.dtype)

        z = z[..., jnp.newaxis, jnp.newaxis, :]

        x = conv(self.num_filters, (7, 7),
                 strides=(2, 2),
                 padding=[(3, 3), (3, 3)],
                 name='conv_init')(x)
        x = norm(name='norm_init')(x)
        x = gated(x.shape[-1])(x, z)

        x = nn.avg_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=strides, conv=conv, norm=norm,
                                   act=self.act, act_out=Identity)(x)
                x = gated(x.shape[-1])(x, z)

        return x



ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlock)

ResNet18Local = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                        block_cls=ResNetBlock, conv=nn.ConvLocal)



CondResNet18 = partial(CondResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
CondResNet34 = partial(CondResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)
CondResNet50 = partial(CondResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock)
CondResNet101 = partial(CondResNet, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlock)
