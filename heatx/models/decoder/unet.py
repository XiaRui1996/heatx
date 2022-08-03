"""
Flax implementation of UNet.
Adapted from:
  https://github.com/google/flax/blob/main/examples/imagenet/models.py
"""
from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

from heatx.models.encoder import ResNetBlock, BottleneckResNetBlock, AttnEncodingLayer
from heatx.models.decoder.deconv import ResNetBlockT, BottleneckResNetBlockT
from heatx.models.nnutil import SkipConnCondGatedUnit, CondGatedUnit


ModuleDef = Any
Identity = lambda x: x
NoNorm = lambda *args, **kwargs: Identity


class UNetEncoder(nn.Module):
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.silu
    conv: ModuleDef = nn.Conv
    attn_resolution: int = 16

    @nn.compact
    def __call__(self, x, z):
        outputs = []
        glu = partial(CondGatedUnit, dtype=self.dtype)
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.GroupNorm, num_groups=None, group_size=16, dtype=self.dtype)
        gated = partial(SkipConnCondGatedUnit, norm=norm, dtype=self.dtype)
        agg = partial(SkipConnCondGatedUnit, z.shape[-1], norm=nn.LayerNorm, dtype=self.dtype)
        attn = partial(AttnEncodingLayer, num_heads=8, norm=norm, dtype=self.dtype)
        hw = self.attn_resolution
        mean = partial(jnp.mean, axis=(-3, -2), keepdims=True)
        std = partial(jnp.std, axis=(-3, -2), keepdims=True)
        concat = lambda *args: jnp.concatenate([*args], axis=-1)

        z = z[..., jnp.newaxis, jnp.newaxis, :]

        x = conv(self.num_filters, (7, 7),
                 strides=(2, 2),
                 padding=[(3, 3), (3, 3)],
                 name='conv_init')(x)
        a = glu(z.shape[-1], z.shape[-1])(z, concat(mean(x), std(x)))
        x = norm()(x)
        x = gated(x.shape[-1])(x, a)
        outputs.append(x)

        x = nn.avg_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=strides, conv=conv, norm=norm,
                                   act=self.act, act_out=Identity)(x)

                a = agg()(a, concat(mean(x), std(x)))
                x = gated(x.shape[-1])(x, a)

            if x.shape[-2] <= hw and x.shape[-3] <= hw:
                resolution = x.shape[1:-1]
                x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))
                x = jnp.concatenate([
                    jnp.mean(x, axis=-2, keepdims=True), x
                ], axis=-2)
                x = attn(x.shape[-1], max_len=x.shape[-2])(x)
                h, x = x[..., 0, :], x[..., 1:, :]
                h = h[..., jnp.newaxis, jnp.newaxis, :]
                x = jnp.reshape(x, (x.shape[0], *resolution, x.shape[-1]))
                a = agg()(a, h)
                x = gated(x.shape[-1])(x, a)

            outputs.append(x)
        
        return jnp.squeeze(a, axis=(-3, -2)), outputs



class UNetDecoder(nn.Module):
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_channels: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.silu
    conv: ModuleDef = nn.ConvTranspose
    attn_resolution: int = 16

    @nn.compact
    def __call__(self, xs, z):
        '''
        Args:
          xs: List of (B, H, W, C) arrays.
          z: (B, d) array.
        '''
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.GroupNorm, num_groups=None, group_size=16, dtype=self.dtype)
        gated = partial(SkipConnCondGatedUnit, norm=norm, dtype=self.dtype)
        attn = partial(AttnEncodingLayer, num_heads=8, norm=norm, dtype=self.dtype)
        dense = partial(nn.Dense, dtype=self.dtype)
        num_stages = len(self.stage_sizes)
        hw = self.attn_resolution

        z = z[..., jnp.newaxis, jnp.newaxis, :]
        x = xs.pop()

        for i, block_size in enumerate(reversed(self.stage_sizes)):
            if i > 0:
                x = jnp.concatenate([x, xs.pop()], axis=-1)
            for j in range(block_size):
                strides = (2, 2) if i < (num_stages - 1) and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * (2 ** (num_stages - i - 1)),
                                   strides=strides, conv=conv, norm=norm,
                                   act=self.act, act_out=Identity)(x)
                x = gated(x.shape[-1])(x, z)

            if x.shape[-2] <= hw and x.shape[-3] <= hw:
                resolution = x.shape[1:-1]
                x = jnp.reshape(x, (x.shape[0], -1, x.shape[-1]))
                x = jnp.concatenate([
                    jnp.mean(x, axis=-2, keepdims=True), x
                ], axis=-2)
                x = attn(x.shape[-1], max_len=x.shape[-2])(x)
                x = x[..., 1:, :]
                x = jnp.reshape(x, (x.shape[0], *resolution, x.shape[-1]))
                x = gated(x.shape[-1])(x, z)

        x = self.block_cls(self.num_filters,
                           strides=(2, 2), conv=conv, norm=norm,
                           act=self.act, act_out=Identity)(x)
        x = gated(x.shape[-1])(x, z)

        x = jnp.concatenate([x, xs.pop()], axis=-1)

        x = self.block_cls(self.num_filters,
                           strides=(2, 2), conv=conv, norm=norm,
                           act=self.act, act_out=Identity)(x)
        x = norm()(x)
        x = SkipConnCondGatedUnit(x.shape[-1], dtype=self.dtype, norm=NoNorm)(x, z)
        x = dense(self.num_channels, kernel_init=nn.initializers.zeros)(x)

        return jnp.asarray(x, self.dtype)


class UNet(nn.Module):
    stage_sizes: Sequence[int]
    conv_block_cls: ModuleDef
    deconv_block_cls: ModuleDef
    num_channels: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.silu
    conv: ModuleDef = nn.Conv
    deconv: ModuleDef = nn.ConvTranspose
    attn_layers: int = 2

    @nn.compact
    def __call__(self, x, z):
        norm = partial(nn.GroupNorm, num_groups=None, group_size=16, dtype=self.dtype)
        agg = partial(SkipConnCondGatedUnit, z.shape[-1], norm=nn.LayerNorm, dtype=self.dtype)
        attn = partial(AttnEncodingLayer, num_heads=8, norm=norm, dtype=self.dtype)

        a, xs = UNetEncoder(
            stage_sizes=self.stage_sizes,
            block_cls=self.conv_block_cls,
            num_filters=self.num_filters,
            dtype=self.dtype,
            act=self.act,
            conv=self.conv
        )(x, z)

        h = xs.pop()

        resolution = h.shape[1:-1]
        h = jnp.reshape(h, (h.shape[0], -1, h.shape[-1]))
        h = jnp.concatenate([
            jnp.mean(h, axis=-2, keepdims=True), h
        ], axis=-2)

        h = nn.Sequential([
            attn(h.shape[-1], max_len=h.shape[-2])
            for _ in range(self.attn_layers)
        ])(h)
        a = agg()(a, h[..., 0, :])
        h = h[..., 1:, :]
        h = jnp.reshape(h, (h.shape[0], *resolution, -1))

        xs.append(h)

        z = jnp.concatenate([a, z], axis=-1)
        x = UNetDecoder(
            stage_sizes=self.stage_sizes,
            block_cls=self.deconv_block_cls,
            num_channels=self.num_channels,
            num_filters=self.num_filters,
            dtype=self.dtype,
            act=self.act,
            conv=self.deconv
        )(xs, z)
        return x


UNet18 = partial(UNet, stage_sizes=[2, 2, 2, 2],
                 conv_block_cls=ResNetBlock,
                 deconv_block_cls=ResNetBlockT)
UNet34 = partial(UNet, stage_sizes=[3, 4, 6, 3],
                 conv_block_cls=ResNetBlock,
                 deconv_block_cls=ResNetBlockT)
UNet50 = partial(UNet, stage_sizes=[3, 4, 6, 3],
                 conv_block_cls=BottleneckResNetBlock,
                 deconv_block_cls=BottleneckResNetBlockT)
UNet101 = partial(UNet, stage_sizes=[3, 4, 23, 3],
                  conv_block_cls=BottleneckResNetBlock,
                  deconv_block_cls=BottleneckResNetBlockT)
