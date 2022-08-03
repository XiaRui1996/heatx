from typing import Callable

import jax
import jax.numpy as jnp

from heatx.models.generative.sde import SDE


def normal_langevin(rng, x, t):
    prefix = jnp.broadcast_shapes(t.shape, x.shape[:-3])
    alpha = jnp.exp(-t / 2.0)
    sigma = jnp.sqrt(1.0 - jnp.exp(-t))
    axes = (-1, -2, -3)
    mean = jnp.expand_dims(alpha, axes) * x
    noise = jax.random.normal(rng, prefix + x.shape[-3:],  dtype=x.dtype)
    return mean + jnp.expand_dims(sigma, axes) * noise


class ImageScore:
    def __init__(self, sde: SDE, noise_pred_fn: Callable, t0: float=0.001, t1: float=1.0, nfe=32):
        self.sde = sde
        self.noise_pred_fn = noise_pred_fn
        self.t0 = t0
        self.t1 = t1
        self.nfe = nfe

    def brownian(self, rng, x, t):
        '''
        Args:
          x: (..., H, W, C) array.
          t: (...) array.
        '''
        sde, noise_pred_fn = self.sde, self.noise_pred_fn

        prefix = x.shape[:-3]
        x = jnp.reshape(x, (-1,) + x.shape[-3:])
        t0 = jnp.asarray(self.t0, dtype=x.dtype)
        t = jnp.asarray(t, dtype=x.dtype)

        rng, subkey = jax.random.split(rng)
        x0, _ = sde.forward(subkey, x, t0)
        z = sde.ddim_forward(x0, t0, noise_pred_fn, max_steps=self.nfe)

        z = jnp.reshape(z, prefix + x.shape[-3:])

        rng, subkey = jax.random.split(rng)
        zs = normal_langevin(subkey, z, t)

        xs = sde.ddim_inverse(zs, 1.0, noise_pred_fn, t0=t0, max_steps=self.nfe)
        xs = sde.rough_inverse(xs, t0, noise_pred_fn(xs, t0))
        return xs

    def project(self, x, t=0.01):
        prefix = x.shape[:-3]
        x = jnp.reshape(x, (-1,) + x.shape[-3:])
        t = jnp.asarray(t, dtype=x.dtype)

        proj = self.sde.rough_inverse(x, t, self.noise_pred_fn(x, t))

        proj = jnp.reshape(proj, prefix + proj.shape[1:])
        return proj