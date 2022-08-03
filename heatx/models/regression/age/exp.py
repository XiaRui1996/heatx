#!/usr/bin/env python
from functools import partial
from pathlib import Path
import math

from absl import logging
from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags
from flax import jax_utils
import jax.numpy as jnp
import jax.lax as lax
import ml_collections as mlc
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from heatx.trainutil import restore_checkpoint, save_image
from heatx.datautil.dataloader import FaceDataset
from heatx.models.generative.sde.model import SDE
from heatx.funcutil import hessian_diag_reverse_mode
from heatx.imgutil import to_heatmap


from pde import create_pde_solution_model, \
    create_pde_solution_train_state, \
    create_learning_rate_fn, create_input_iter, \
    EuclideanImage, ImageScore, ImplicitManifold, MAX_AGE_PLUS_1

from heatx.models.generative.sde.train import \
    create_model as create_sde_model, \
    create_train_state as create_sde_train_state


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_integer('grid', 32, 'Grid size.')
flags.DEFINE_string('manifold', 'euclid', 'Manifold.')
flags.DEFINE_string('manifold_workdir', None, 'Directory to store model data.')
flags.DEFINE_float('t', 0.1, 'T.')
flags.DEFINE_integer('n', 4, 'Number of images to explain.')
flags.DEFINE_integer('p', 99, 'Percentile in heatmap.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training hyperparameter configuration.',
    lock_config=True)
config_flags.DEFINE_config_file(
    'manifold_config', None, 'File path to the training hyperparameter configuration.',
    lock_config=True)


DENOISE_T = 0.05


def vec_hessian_diag(rng, state, manifold, x, t, batch_size=64):
    def u(_x_):
        if type(manifold) is ImageScore:
            _x_, _ = manifold.sde.forward(rng, _x_, DENOISE_T)
            _x_ = manifold.project(_x_, t=DENOISE_T)
        return state.apply_fn({'params': state.params}, _x_, t)[..., 0]

    def hdiag(_x_):
        return hessian_diag_reverse_mode(u, _x_, batch_size=batch_size)

    return jax.vmap(hdiag)(x)


def explain_step(rng, state, x, manifold, T, grid_size):
    if type(manifold) is ImplicitManifold:
        def noise_pred_fn(_xt_, _t_):
            return manifold.state.apply_fn({'params': manifold.state.params}, _xt_, _t_)
        M = ImageScore(manifold.sde, noise_pred_fn, nfe=32)
    elif manifold is None:
        M = EuclideanImage()
    else:
        raise ValueError('Unknown manifold')

    def body_fn(key, t):
        key, subkey = jax.random.split(key)
        diag = vec_hessian_diag(subkey, state, M, x, t)
        return key, diag

    ts = jnp.linspace(0.0, T, grid_size + 1)
    _, attrs = lax.scan(body_fn, rng, ts)

    trapezoidal = (attrs[0] + attrs[-1]) / 2.0 + jnp.sum(attrs[1:-1], axis=0)
    trapezoidal = trapezoidal / grid_size  # * T

    attrs = jnp.concatenate([
        trapezoidal[:, jnp.newaxis, ...],
        jnp.swapaxes(attrs, 0, 1)
    ], axis=1)
    hm = jax.vmap(partial(to_heatmap, percentile=FLAGS.p))
    heatmaps = hm(attrs)

    y = state.apply_fn(
        {'params': state.params},
        x, ts[:, jnp.newaxis]
    )[..., 0]
    y = jnp.swapaxes(y, 0, 1) # (B, T)

    x = x[:, jnp.newaxis, ...] / 2 + 0.5  # [-1, 1] -> [0, 1]
    if x.shape[-1] == 1:
        x = jnp.concatenate([x, x, x], axis=-1)

    return jnp.concatenate([
        x,
        hm(trapezoidal)[:, jnp.newaxis, ...],
        heatmaps
    ], axis=1), y


def explain(rng, state, manifold, x, T, grid_size, imgdir):
    p_exp_step = jax.pmap(partial(
        explain_step,
        T=T, grid_size=grid_size
    ), axis_name='batch')
    p_exp_step = partial(p_exp_step, manifold=manifold)

    imgs = []
    ys = []
    ts = np.linspace(0.0, T, grid_size + 1)


    logging.info('Initial compilation, this might take some minutes...')

    # One image per GPU at a time
    for i in tqdm(range(x.shape[1])):
        rst, y = p_exp_step(rng, state, x[:, i:i+1, ...])
        if i == 0:
            logging.info('Initial compilation completed.')
        imgs.append(rst)
        ys.append(y)

    imgs = jnp.concatenate(imgs, axis=1)
    imgs = jnp.reshape(imgs, (-1,) + imgs.shape[-3:])
    save_image(imgs, imgdir / 'attr.png', nrow=1 + 1 + 1 + grid_size + 1)

    ys = jnp.concatenate(ys, axis=1)
    ys = jnp.reshape(ys, (-1,) + ys.shape[-1:])

    ys = np.asarray(ys * MAX_AGE_PLUS_1)

    B = ys.shape[0]
    fig, axes = plt.subplots(B, 1, sharex=True, figsize=(4, 4 * B))
    for i in range(B):
        axes[i].plot(ts, ys[i])

    plt.savefig(imgdir / 'pred.svg')
    plt.savefig(imgdir / 'pred.pdf')


def run(config: mlc.ConfigDict, workdir: str,
        manifold: str, manifold_workdir: str, manifold_config: mlc.ConfigDict):
    ckptdir = Path(workdir) / 'heat' / manifold / 'checkpoints'
    assert ckptdir.exists()
    imgdir = Path(workdir) / 'heat' / manifold / 'images'
    imgdir.mkdir(exist_ok=True, parents=True)

    rng = jax.random.PRNGKey(0)

    if config.batch_size % jax.device_count() > 0:
        raise ValueError(
            'Batch size must be divisible by the number of devices')
    local_batch_size = config.batch_size // jax.process_count()

    datadir = Path(config.datadir)
    input_dtype = np.float16 if config.half_precision else np.float32

    dataset_kwargs = dict(
        width=config.image_size, height=config.image_size,
        dtype=input_dtype, unit_interval=True
    )
    testloader = DataLoader(
        FaceDataset(datadir / 'test', aug=False, **dataset_kwargs),
        batch_size=local_batch_size * 2,
        shuffle=False, num_workers=8, drop_last=True
    )
    testloader = create_input_iter(testloader)
    batch = next(testloader)['image'][:, :FLAGS.n, ...]

    assert manifold in ('euclid', 'score')
    if manifold == 'score':
        sde, score = create_sde_model(manifold_config)
        tmp_lr_fn = create_learning_rate_fn(manifold_config, 100)
        score = create_sde_train_state(rng, manifold_config, score, tmp_lr_fn)
        score = restore_checkpoint(score, Path(manifold_workdir) / 'checkpoints')
        logging.info(f'SDE model: state.step = {int(score.step)}')
        manifold = ImplicitManifold(score, sde)
        manifold = jax_utils.replicate(manifold)
    else:
        manifold = None

    model = create_pde_solution_model(config)
    learning_rate_fn = create_learning_rate_fn(config, 100)
    state = create_pde_solution_train_state(rng, config, model, learning_rate_fn)
    state = restore_checkpoint(state, ckptdir)
    state = jax_utils.replicate(state)
    rng = jax.random.split(rng, jax.local_device_count())

    explain(rng, state, manifold, batch, T=FLAGS.t, grid_size=FLAGS.grid, imgdir=imgdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    run(
        FLAGS.config, FLAGS.workdir,
        manifold=FLAGS.manifold,
        manifold_workdir=FLAGS.manifold_workdir,
        manifold_config=FLAGS.manifold_config
    )

if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)