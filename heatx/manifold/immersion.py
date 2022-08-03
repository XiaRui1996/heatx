from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap, jacfwd, hessian
import jax.lax as lax
from jax.tree_util import Partial
from jax.scipy.optimize import minimize
import jaxopt
from matplotlib.pyplot import axis


_L_BFGS_EXP = 'L-BFGS-EXPERIMENTAL-DO-NOT-RELY-ON-THIS'


class ImmersedSubmanifold:
    def __init__(self, decoder: Callable, lower_bound=-3.0, upper_bound=3.0):
        self.decoder = decoder
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def decode(self, z):
        return self.decoder(z)

    def jacobian(self, z):
        prefix = z.shape[:-1]
        z = jnp.reshape(z, (-1, z.shape[-1]))
        dxdz = vmap(jacfwd(self.decoder))(z)
        return jnp.reshape(dxdz, prefix + dxdz.shape[-2:])
    
    def jacobian_metric(self, z):
        jac = self.jacobian(z)
        metric = jnp.matmul(jnp.swapaxes(jac, -1, -2), jac)
        return jac, metric

    def metric_tensor(self, z):
        jac = self.jacobian(z)
        return jnp.matmul(jnp.swapaxes(jac, -1, -2), jac)
    
    def metric_christoffel(self, z):
        prefix = z.shape[:-1]
        z = jnp.reshape(z, (-1, z.shape[-1]))

        hess = vmap(hessian(self.decoder))(z)                # (B, D, d, d)
        jac = vmap(jacfwd(self.decoder))(z)                  # (B, D, d)
        metric = jnp.matmul(jnp.swapaxes(jac, -1, -2), jac)  # (B, d, d)
        inv = jnp.linalg.inv(metric)
        chris = jnp.einsum('bim,bojk,bom->bijk', inv, hess, jac)

        metric = jnp.reshape(metric, prefix + metric.shape[-2:])
        chris = jnp.reshape(chris, prefix + chris.shape[-3:])
        return metric, chris

    def normalized_metric_tensor(self, z):
        M = self.metric_tensor(z)
        _, logdet = jnp.linalg.slogdet(M)
        return M / jnp.exp(logdet * 2.0 / z.shape[-1])[..., jnp.newaxis, jnp.newaxis]

    def volume(self, z):
        _, logdet = jnp.linalg.slogdet(self.metric_tensor(z))
        return jnp.exp(logdet / 2.0)

    def random_tangent_direction(self, rng, metric, z):
        noise = jax.random.normal(rng, z.shape)
        norm = jnp.linalg.norm(noise, axis=-1, keepdims=True)
        noise = noise / norm

        w, V = jnp.linalg.eigh(metric)
        scale = jnp.sqrt(jnp.where(w > 0, w, 1.0))
        noise = jnp.where(w > 0, noise / scale, 0.0)
        v = jnp.squeeze(jnp.matmul(V, noise[..., jnp.newaxis]), axis=-1)
        return v

    def random_walk(self, prng, z, *,
                    step_size=0.01, num_steps=10, second_order=False):
        '''
        Args:
          z: (..., d) array.

        Returns:
          zts: (T, ..., d) array.
        '''
        def body(carry, _):
            z_, key = carry
            key, subkey = jax.random.split(key)

            if second_order:
                metric, chris = self.metric_christoffel(z_)
                v = self.random_tangent_direction(subkey, metric, z_)
                v = step_size * v
                ord_2 = jnp.einsum('...ijk,...j,...k->...i', chris, v, v)
                delta = v - 0.5 * ord_2
            else:
                metric = self.metric_tensor(z_)
                v = self.random_tangent_direction(subkey, metric, z_)
                delta = step_size * v

            zt = z_ + delta
            zt = jnp.where(zt < self.upper_bound, zt, self.upper_bound)
            zt = jnp.where(zt > self.lower_bound, zt, self.lower_bound)
            carry = (zt, key)
            return carry, zt

        prefix, dim_z = z.shape[:-1], z.shape[-1]
        z0 = jnp.reshape(z, (-1, dim_z))
        _, zts = lax.scan(body, (z0, prng), jnp.arange(num_steps))
        zts = jnp.reshape(zts, (num_steps, *prefix, dim_z))

        return zts

    def brownian(self, rng, z, t, *,
                 max_num_steps=1024, step_size=0.01, proj_rw=False):
        '''
        Args:
          z: (..., d) array.
          t: (...) array in [0, 1].
        
        Returns:
          zt: (..., d) array.
        '''
        prefix = jnp.broadcast_shapes(t.shape, z.shape[:-1])
        t = jnp.broadcast_to(t, prefix)
        z = jnp.broadcast_to(z, prefix + z.shape[-1:])

        random_walk = self.proj_random_walk if proj_rw else self.random_walk

        zts = jnp.concatenate([
            z[jnp.newaxis, ...],
            random_walk(
                rng, z, step_size=step_size, num_steps=max_num_steps
            )  # (T, ..., d)
        ], axis=0)

        idx = t * float(max_num_steps)
        idx = idx.astype(jnp.int32)
        idx = idx[jnp.newaxis, ..., jnp.newaxis]
        zt = jnp.take_along_axis(zts, idx, axis=0)

        return jnp.squeeze(zt, axis=0)

    def proj(self, x, z_init, algo='GD', maxiter=16, lr=0.001, tol=1E-5):
        algo = algo.upper()

        prefix = x.shape[:-1]
        x = jnp.reshape(x, (-1,) + x.shape[-1:])
        z_init = jnp.reshape(z_init, (-1,) + z_init.shape[-1:])

        def rec_err(_z_, _x_):
            return jnp.sum((_x_ - self.decoder(_z_)) ** 2)

        if algo == 'NAIVEGD':
            grad_fn = jax.grad(Partial(rec_err, _x_=x))

            def body_fn(_, z):
                return z - lr * grad_fn(z)

            z = lax.fori_loop(0, maxiter, body_fn, z_init)

        else:
            def arg_min_z(xi, zi):
                if algo == 'BFGS':
                    options = {'maxiter': maxiter, 'gtol': tol}
                    rst = minimize(Partial(rec_err, _x_=xi), zi, method='BFGS', options=options)
                    return rst.x
                elif algo == 'LBFGS-EXP':
                    options = {'maxiter': maxiter, 'gtol': tol}
                    rst = minimize(Partial(rec_err, _x_=xi), zi, method=_L_BFGS_EXP, options=options)
                    return rst.x
                elif algo == 'LBFGS':
                    solver = jaxopt.LBFGS(fun=rec_err, maxiter=maxiter, tol=tol)
                    rst = solver.run(zi, _x_=xi)
                    return rst.params
                elif algo == 'GD':
                    solver = jaxopt.GradientDescent(fun=rec_err, maxiter=maxiter, tol=tol)
                    rst = solver.run(zi, _x_=xi)
                    return rst.params
                else:
                    raise ValueError(f'Unknown optimization method: {algo}')

            z = jax.vmap(arg_min_z)(x, z_init)

        z = jnp.reshape(z, prefix + z.shape[1:])
        p = self.decoder(z)
        return p, z

    def proj_random_walk(self, prng, z, *, step_size=0.01, num_steps=10):
        '''
        Args:
          z: (..., d) array.

        Returns:
          zts: (T, ..., d) array.
        '''
        def body(carry, _):
            z_, key = carry
            key, subkey = jax.random.split(key)

            jac, metric = self.jacobian_metric(z_)
            v = self.random_tangent_direction(subkey, metric, z_)
            vx = jnp.squeeze(jnp.matmul(jac, v[..., jnp.newaxis]), axis=-1)

            x = self.decode(z_)
            x = x + step_size * vx
            _, zt = self.proj(x, z_, algo='GD', maxiter=16)

            carry = (zt, key)
            return carry, zt

        prefix, dim_z = z.shape[:-1], z.shape[-1]
        z0 = jnp.reshape(z, (-1, dim_z))
        _, zts = lax.scan(body, (z0, prng), jnp.arange(num_steps))
        zts = jnp.reshape(zts, (num_steps, *prefix, dim_z))

        return zts


class ImmersedImage:
    def __init__(self, decoder: Callable, encoder: Callable,
                 lower_bound=-3.0, upper_bound=3.0):
        def dec(z):
            x = decoder(z)
            return jnp.reshape(x, x.shape[:-3] + (-1,))

        self.decoder = decoder
        self.encoder = encoder
        self.manifold = ImmersedSubmanifold(
            dec, lower_bound=lower_bound, upper_bound=upper_bound)

    def jacobian(self, x):
         z = self.encoder(x)
         return self.manifold.metric_tensor(z)

    def metric_tensor(self, x):
        z = self.encoder(x)
        return self.manifold.metric_tensor(z)

    def random_walk(self, rng, x, *, step_size=0.01, num_steps=10):
        z = self.encoder(x)
        zts = self.manifold.random_walk(
            rng, z, step_size=step_size, num_steps=num_steps
        )
        return self.decoder(zts)
    
    def brownian(self, rng, x, t, *,
                 max_num_steps=1024, step_size=0.01, proj_rw=False):
        z = self.encoder(x)
        zt = self.manifold.brownian(
            rng, z, t, max_num_steps=max_num_steps,
            step_size=step_size, proj_rw=proj_rw
        )
        return self.decoder(zt)

    def project(self, x, algo='GD', maxiter=16, lr=0.001, tol=1E-5):
        algo = algo.upper()

        prefix = x.shape[:-3]
        x = jnp.reshape(x, (-1,) + x.shape[-3:])

        z_init = lax.stop_gradient(self.encoder(x))

        p, _ = self.manifold.proj(
            jnp.reshape(x, (x.shape[0], -1)), z_init,
            algo=algo, maxiter=maxiter, lr=lr, tol=tol
        )
        p = jnp.reshape(p, x.shape)
        return jnp.reshape(p, prefix + x.shape[-3:])
