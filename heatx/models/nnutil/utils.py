from functools import partial
from typing import Any

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any


class GatedUnit(nn.Module):
    hid_features: int
    out_features: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        dense = partial(nn.Dense, dtype=self.dtype)

        x = dense(self.hid_features * 2)(x)
        gate, value = jnp.split(x, 2, axis=-1)
        x = nn.sigmoid(gate) * nn.tanh(value)
        return dense(self.out_features)(x)



class CondGatedUnit(nn.Module):
    hid_features: int
    out_features: int
    dtype: Any = jnp.float32
    out_init: Any = nn.linear.default_kernel_init

    @nn.compact
    def __call__(self, x, y):
        dense = partial(nn.Dense, dtype=self.dtype)

        x = dense(self.hid_features * 2)(x)
        x_gate, x_value = jnp.split(x, 2, axis=-1)

        y = dense(self.hid_features * 2)(y)
        y_gate, y_value = jnp.split(y, 2, axis=-1)

        x = nn.sigmoid(x_gate + y_gate) * nn.tanh(x_value + y_value)
        return dense(self.out_features, kernel_init=self.out_init)(x)


class SkipConnCondGatedUnit(nn.Module):
    hid_features: int
    norm: ModuleDef
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, y):
        r = CondGatedUnit(
            self.hid_features, x.shape[-1],
            out_init=nn.initializers.zeros,
            dtype=self.dtype
        )(x, y)
        return self.norm(dtype=self.dtype)(x + r)
