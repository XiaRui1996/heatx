from typing import Any, Callable
from functools import partial
import math

import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.special import logsumexp
from jax.scipy.optimize import minimize
import jaxopt

from heatx.models.encoder import ResNet18, AttnEncodingLayer
from heatx.models.decoder import ResNetT18
from heatx.models.nnutil import GatedUnit


ModuleDef = Any


def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1)


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std


class ConvVAE(nn.Module):
    channels: int
    latents: int
    hidden_channels: int = 512
    encoder_cls: ModuleDef = ResNet18
    decoder_cls: ModuleDef = ResNetT18
    attn_layers: int = 2
    dtype: Any = jnp.float32
    fourier_min: int = 6
    fourier_max: int = 8

    def setup(self):
        self.encoder = self.encoder_cls(dtype=self.dtype)
        self.decoder = self.decoder_cls(
            num_channels=self.channels, dtype=self.dtype)

        self.attn_enc = nn.Sequential([
            AttnEncodingLayer(self.hidden_channels, num_heads=8, dtype=self.dtype)
            for _ in range(self.attn_layers)
        ])
        self.attn_dec = nn.Sequential([
            AttnEncodingLayer(self.hidden_channels, num_heads=8, dtype=self.dtype)
            for _ in range(self.attn_layers)
        ])
        self.mlp_enc = GatedUnit(self.hidden_channels, self.latents * 2, dtype=self.dtype)
        self.linear_dec = nn.Dense(self.hidden_channels, dtype=self.dtype)

        self.x_fourier_freqs = 2.0 * jnp.pi * \
            (2.0 ** jnp.arange(self.fourier_min, self.fourier_max + 1, dtype=self.dtype))

    def __call__(self, x, z_rng, training=True):
        mean, logvar, resolution = self.encode(x)
        z = mean if not training \
            else reparameterize(z_rng, mean, logvar)
        recon_x = self.decode(z, resolution)
        return recon_x, mean, logvar

    def encode(self, x):
        prefix = x.shape[:-3]
        x = jnp.reshape(x, (-1,) + x.shape[-3:])

        ff = jnp.reshape(x[..., jnp.newaxis] * self.x_fourier_freqs, x.shape[:-1] + (-1,))
        x = jnp.concatenate([x, jnp.sin(ff), jnp.cos(ff)], axis=-1)

        h = self.encoder(x)
        resolution = h.shape[1:-1]

        h = jnp.reshape(h, (h.shape[0], -1, h.shape[-1]))
        h = jnp.concatenate([jnp.mean(h, axis=-2, keepdims=True), h], axis=-2)
        h = self.attn_enc(h)

        agg, h = h[..., 0, :], h[..., 1:, :]
        h = jnp.concatenate([
            agg, jnp.std(h, axis=-2), logsumexp(h, axis=-2), logsumexp(-h, axis=-2)
        ], axis=-1)
        h = self.mlp_enc(h)

        h = jnp.reshape(h, prefix + (-1,))
        mean, logvar = jnp.split(h, 2, axis=-1)
        logvar = 3.0 * jnp.tanh(-nn.elu(-logvar) / 3.0)  # ~ (-3, 1)
        return mean, logvar, resolution

    def decode(self, z, resolution):
        prefix = z.shape[:-1]
        z = jnp.reshape(z, (-1, z.shape[-1]))

        h = self.linear_dec(z)
        h = h[..., jnp.newaxis, :]
        shape = (h.shape[0], np.prod(resolution), h.shape[-1])
        h = jnp.concatenate([h, jnp.zeros(shape, dtype=h.dtype)], axis=-2)
        h = self.attn_dec(h)
        h = jnp.reshape(h[..., 1:, :], (h.shape[0], *resolution, -1))
        h = nn.silu(h)

        x = self.decoder(h, z)
        x = nn.sigmoid(x)

        x = jnp.reshape(x, prefix + x.shape[1:])
        return x
    
    def encode_fn(self):
        return lambda params, x: self.apply(
            {'params': params}, x,
            method=self.encode, mutable=False
        )
    
    def decode_fn(self):
        return lambda params, x, resolution: self.apply(
            {'params': params}, x, resolution,
            method=self.decode, mutable=False
        )

    def iter_infer_fn(self):
        return iter_infer_fn(self)

    def reconstruct_fn(self):
        return partial(self.iter_infer_fn(), reconstruct=True)


def iter_infer_fn(model: ConvVAE):
    def call(params, x, *, reconstruct=False, algo='GD', maxiter=100, lr=0.001):
        algo = algo.upper()

        prefix = x.shape[:-3]
        x = jnp.reshape(x, (-1,) + x.shape[-3:])

        z_init, _, resolution = model.apply(
            {'params': params}, x,
            method=model.encode, mutable=False
        )

        if algo == 'NAIVEGD':
            def f(z):
                recon_x = model.apply(
                    {'params': params}, z, resolution,
                    method=model.decode, mutable=False
                )
                return jnp.sum((x - recon_x) ** 2)
            
            grad_fn = jax.grad(f)

            def body_fn(_, z):
                return z - lr * grad_fn(z)

            z = lax.fori_loop(0, maxiter, body_fn, z_init)
        
        else:
            def rec(xi, zi):
                def f(z):
                    recon_xi = model.apply(
                        {'params': params}, z, resolution,
                        method=model.decode, mutable=False
                    )
                    return jnp.sum((xi - recon_xi) ** 2)

                if algo == 'BFGS':
                    options = {'maxiter': maxiter}
                    rst = minimize(f, zi, method='BFGS', options=options)
                    return rst.x
                elif algo == 'LBFGS-EXP':
                    options = {'maxiter': maxiter}
                    rst = minimize(f, zi, method='L-BFGS-EXPERIMENTAL-DO-NOT-RELY-ON-THIS', options=options)
                    return rst.x
                elif algo == 'LBFGS':
                    solver = jaxopt.LBFGS(fun=f, maxiter=maxiter, tol=1E-5)
                    rst = solver.run(zi)
                    return rst.params
                elif algo == 'GD':
                    solver = jaxopt.GradientDescent(fun=f, maxiter=maxiter)
                    rst = solver.run(zi)
                    return rst.params
                else:
                    raise ValueError(f'Unknown optimization method: {algo}')

            z = jax.vmap(rec)(x, z_init)

        x = model.apply(
            {'params': params}, z, resolution,
            method=model.decode, mutable=False
        )
        z = jnp.reshape(z, prefix + z.shape[-1:])
        x = jnp.reshape(x, prefix + x.shape[1:])
        return (x, z) if reconstruct else z

    return call
