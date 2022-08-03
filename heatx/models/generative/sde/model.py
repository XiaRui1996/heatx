from typing import Any
from dataclasses import dataclass
import math
import abc

import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
import diffrax as dfx

from heatx.models.decoder import UNet18


ModuleDef = Any


class UNetPredictor(nn.Module):
    channels: int
    unet_cls: ModuleDef = UNet18
    dtype: Any = jnp.float32
    x_fourier_min: int = 5
    x_fourier_max: int = 8

    def setup(self):
        self.unet = self.unet_cls(num_channels=self.channels, dtype=self.dtype)

        self.x_fourier_freqs = 2.0 * jnp.pi * \
            (2.0 ** jnp.arange(self.x_fourier_min, self.x_fourier_max + 1, dtype=self.dtype))

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

        x = self.unet(x, t)

        x = jnp.reshape(x, prefix + x.shape[1:])
        return x


@dataclass
class SDE(abc.ABC):
    @abc.abstractmethod
    def drift_diffusion_weights(self, t):
        '''
        Returns:
          (ft, g2t): dx = ft * xt dt + sqrt(g2t) dw
        '''
        pass

    @abc.abstractmethod
    def forward_weights(self, t):
        pass

    def _brdcst_forward_weights(self, t):
        alpha, sigma = self.forward_weights(t)
        alpha = jnp.expand_dims(alpha, (-1, -2, -3))
        sigma = jnp.expand_dims(sigma, (-1, -2, -3))
        return alpha, sigma
    
    def _forward_with_noise(self, x, t, noise):
        alpha, sigma = self._brdcst_forward_weights(t)
        return alpha * x + sigma * noise
    
    def _brdcst_sqrt_NSR(self, t):
        alpha, beta = self._brdcst_forward_weights(t)
        return beta / alpha

    def forward(self, rng, x, t):
        '''
        Args:
          x: (..., H, W, C) array.
          t: (...) array.
        '''
        t = jnp.asarray(t)
        prefix = jnp.broadcast_shapes(t.shape, x.shape[:-3])
        noise = jax.random.normal(rng, prefix + x.shape[-3:])
        return self._forward_with_noise(x, t, noise), noise

    def sample_prior(self, rng, shape):
        _, maxstd = self.forward_weights(1.0)
        return jax.random.normal(rng, shape) * maxstd

    def rough_inverse(self, xt, t, noise):
        t = jnp.asarray(t)
        alpha, sigma = self._brdcst_forward_weights(t)
        return (xt - sigma * noise) / alpha

    def ode(self, xt0, t0, t1, noise_pred_fn, *, max_steps=100):
        def drift(t_, xt_, args):
            ft_, g2t_ = self.drift_diffusion_weights(t_)
            _, std = self.forward_weights(t_)
            score = -noise_pred_fn(xt_, t_) / std
            return ft_ * xt_ - 0.5 * g2t_ * score

        term = dfx.ODETerm(drift)
        solver = dfx.Tsit5()
        sol = dfx.diffeqsolve(
            term, solver, t0, t1, (t1 - t0) / max_steps, xt0,
            saveat=dfx.SaveAt(ts=[t1]), adjoint=dfx.NoAdjoint()
        )
        return sol.ys[-1]

    def ode_inverse(self, xt, t, noise_pred_fn, *, max_steps=100, t0=0.001):
        return self.ode(xt, t, t0, noise_pred_fn, max_steps=max_steps)

    def ode_forward(self, xt, t, noise_pred_fn, *, max_steps=100, t1=1.0):
        return self.ode(xt, t, t1, noise_pred_fn, max_steps=max_steps)

    def ddim(self, xt0, t0, t1, noise_pred_fn, *, max_steps=100):
        def body_fn(x, pair):
            t_now, t_next = pair
            noise = noise_pred_fn(x, t_now)
            x0 = self.rough_inverse(x, t_now, noise)
            x_ = self._forward_with_noise(x0, t_next, noise)
            return x_, None

        ts = jnp.linspace(jnp.sqrt(t0), jnp.sqrt(t1), max_steps + 1) ** 2
        pairs = jnp.stack([ts[:-1], ts[1:]], axis=-1)
        x, _ = lax.scan(body_fn, xt0, pairs)
        return x

    def ddim_inverse(self, xt, t, noise_pred_fn, *, max_steps=100, t0=0.0):
        return self.ddim(xt, t, t0, noise_pred_fn, max_steps=max_steps)

    def ddim_forward(self, xt, t, noise_pred_fn, *, max_steps=100, t1=1.0):
        return self.ddim(xt, t, t1, noise_pred_fn, max_steps=max_steps)

    def sde_inverse(self, rng, xt, t, noise_pred_fn, *, max_steps=100, t0=0.001):
        def drift(t_, xt_, args):
            ft_, g2t_ = self.drift_diffusion_weights(t_)
            _, std = self.forward_weights(t_)
            score = -noise_pred_fn(xt_, t_) / std
            return ft_ * xt_ - g2t_ * score

        def diffusion(t_, _, args):
            _, g2t_ = self.drift_diffusion_weights(t_)
            return jnp.sqrt(g2t_)

        bm = dfx.UnsafeBrownianPath(shape=xt.shape, key=rng)
        drift = dfx.ODETerm(drift)
        diffusion = dfx.WeaklyDiagonalControlTerm(diffusion, bm)
        terms = dfx.MultiTerm(drift, diffusion)
        solver = dfx.Heun()
        sol = dfx.diffeqsolve(
            terms, solver, t, t0, (t0 - t) / max_steps, xt,
            saveat=dfx.SaveAt(ts=[t0]), adjoint=dfx.NoAdjoint()
        )
        return sol.ys[-1]

    def langevin(self, rng, xt, t, noise_pred_fn, *, max_steps=100, time=1.0):
        def drift(_, x_, args):
            _, std = self.forward_weights(t)
            score = -noise_pred_fn(x_, t) / std
            return 0.5 * score

        def diffusion(t_, _, args):
            return jnp.asarray(1.0)

        bm = dfx.UnsafeBrownianPath(shape=xt.shape, key=rng)
        drift = dfx.ODETerm(drift)
        diffusion = dfx.WeaklyDiagonalControlTerm(diffusion, bm)
        terms = dfx.MultiTerm(drift, diffusion)
        solver = dfx.Heun()
        sol = dfx.diffeqsolve(
            terms, solver, 0.0, time, time / max_steps, xt,
            saveat=dfx.SaveAt(ts=[time]), adjoint=dfx.NoAdjoint()
        )
        return sol.ys[-1]

    def cde(self, xt0, t0, t1, noise_pred_fn, *, max_steps=100):
        alpha0, _ = self._brdcst_forward_weights(t0)
        yt0 = xt0 / alpha0

        def drift(t_, yt_, args):
            sigma, dsigma = jax.value_and_grad(self._brdcst_sqrt_NSR)(t_)
            denom = jnp.sqrt(1.0 + sigma ** 2)
            zt_ = yt_ / denom
            noise = noise_pred_fn(zt_, t_)
            return noise * dsigma

        term = dfx.ODETerm(drift)
        solver = dfx.Tsit5()
        sol = dfx.diffeqsolve(
            term, solver, t0, t1, (t1 - t0) / max_steps, yt0,
            saveat=dfx.SaveAt(ts=[t1]), adjoint=dfx.NoAdjoint()
        )

        alpha1, _ = self._brdcst_forward_weights(t1)
        return sol.ys[-1] * alpha1

    def cde_inverse(self, xt, t, noise_pred_fn, *, max_steps=100, t0=0.001):
        return self.ode(xt, t, t0, noise_pred_fn, max_steps=max_steps)

    def cde_forward(self, xt, t, noise_pred_fn, *, max_steps=100, t1=1.0):
        return self.ode(xt, t, t1, noise_pred_fn, max_steps=max_steps)

@dataclass
class VESDE(SDE):
    minSNR: float = 1 / (32.0 ** 2)

    def schedule(self, t):
        maxstd = math.sqrt(1 / self.minSNR)
        return (maxstd + 1) ** t - 1
    
    def drift_diffusion_weights(self, t):
        var_fn = lambda t_: self.schedule(t_) ** 2.0
        _, dvar = jax.jvp(var_fn, (t,), (jnp.ones_like(t),))
        return 0.0, dvar
    
    def forward_weights(self, t):
        return 1.0, self.schedule(t)


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


@dataclass
class VPSDE(SDE):
    minSNR: float = math.exp(-10)

    def integrate_beta(self, t):
        logSNR = math.log(self.minSNR)
        return -math.log(sigmoid(logSNR)) * (t ** 2.0)

    def drift_diffusion_weights(self, t):
        _, beta = jax.jvp(self.integrate_beta, (t,), (jnp.ones_like(t),))
        return -0.5 * beta, beta

    def forward_weights(self, t):
        int_beta = self.integrate_beta(t)
        alpha = jnp.exp(-int_beta)
        std = jnp.sqrt(jnp.maximum(1 - alpha, 1E-6))
        return jnp.sqrt(alpha), std
