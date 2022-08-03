import math

import jax
import jax.numpy as jnp
import jax.lax as lax


def approx_hessian_diag(rng, fn, x, num_samples=128):
    def hvp(v):
        return jax.jvp(jax.grad(fn), (x,), (v,))[1]

    vs = jax.random.bernoulli(rng, shape=(num_samples,) + x.shape)
    vs = vs.astype(x.dtype)
    vs = vs * 2 - 1
    pushfwd = jax.vmap(hvp)(vs)

    diag = jnp.mean(pushfwd * vs, axis=0) / jnp.mean(vs * vs, axis=0)
    return diag


def hessian_diag(fn, x, batch_size=128):
    shape = x.shape

    def wrapper(_x_):
        return fn(jnp.reshape(_x_, shape))
    
    x = jnp.reshape(x, (-1,))
    D = x.shape[-1]
    num_splits = math.ceil(D / batch_size)

    def hvp(v):
        return jax.jvp(jax.grad(wrapper), (x,), (v,))[1]

    vhvp = jax.vmap(hvp)

    def body_fn(i, diag):
        # basis = jnp.eye(batch_size, D, i * batch_size, dtype=x.dtype)
        basis = jnp.zeros((batch_size, D), dtype=x.dtype)
        rowidx = jnp.arange(batch_size, dtype=i.dtype)
        colidx = rowidx + (i * batch_size)
        basis = basis.at[rowidx, colidx].set(1.0, mode='drop')
        return diag + jnp.sum(basis * vhvp(basis), axis=0)

    diag = lax.fori_loop(0, num_splits, body_fn, jnp.zeros_like(x))
    return jnp.reshape(diag, shape)


def hessian_diag_reverse_mode(fn, x, batch_size=128):
    shape = x.shape

    def wrapper(_x_):
        return fn(jnp.reshape(_x_, shape))
    
    x = jnp.reshape(x, (-1,))
    D = x.shape[-1]
    num_splits = math.ceil(D / batch_size)

    grad_fn = jax.grad(wrapper)

    def grad_of_idx(_x_, _idx_):
        return jnp.take_along_axis(grad_fn(_x_), _idx_, axis=-1)
    
    rowidx = jnp.arange(batch_size, dtype=jnp.int32)

    def grad_grad_of_idx(_x_, _idx_):
        jac = jax.jacrev(grad_of_idx)(_x_, _idx_)
        return jac.at[rowidx, _idx_].get()

    def body_fn(i, diag):
        write_idx = jnp.arange(batch_size, dtype=i.dtype) + i * batch_size
        idx = jnp.minimum(write_idx, D - 1)
        grad_grad = grad_grad_of_idx(x, idx)
        diag = diag.at[write_idx].set(grad_grad, mode='drop')
        return diag

    diag = lax.fori_loop(0, num_splits, body_fn, jnp.zeros_like(x))
    return jnp.reshape(diag, shape)
