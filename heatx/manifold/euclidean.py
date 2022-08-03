import jax
import jax.numpy as jnp


class EuclideanImage:
    def brownian(self, rng, x, t):
        '''
        Args:
          x: (..., H, W, C) array.
          t: (...) array.
        '''
        t = jnp.asarray(t, dtype=x.dtype)
        prefix = jnp.broadcast_shapes(t.shape, x.shape[:-3])
        noise = jax.random.normal(rng, prefix + x.shape[-3:])
        xt = x + noise * jnp.expand_dims(t, axis=(-3, -2, -1))
        return xt

    def project(self, x):
        return x
