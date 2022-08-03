#!/usr/bin/env python
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

from absl import logging
from absl import app
from absl import flags
from absl import logging
import jax
from matplotlib.pyplot import axis
from ml_collections import config_flags
from flax import jax_utils
import jax.numpy as jnp
from jax import random
import ml_collections as mlc
from torch.utils.data import DataLoader
import numpy as np

from heatx.trainutil import vsplit, save_image, restore_checkpoint
from heatx.datautil.dataloader import FaceDataset
from heatx.manifold import ImmersedSubmanifold

from train import create_model, create_train_state, \
    create_learning_rate_fn, create_input_iter



FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_bool('rec', False, 'Reconstruct images.')
flags.DEFINE_integer('rw', 16, 'Number of steps to randomly walk.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training hyperparameter configuration.',
    lock_config=True)



def rec_step(state, x, rng, rec_fn):
    x_recon, _, _ = state.apply_fn(
        {'params': state.params},
        x, rng, training=False, mutable=False
    )
    x_recon_opt, _ = rec_fn(state.params, x)

    return jnp.concatenate([
        x[:, jnp.newaxis, ...],
        x_recon[:, jnp.newaxis, ...],
        x_recon_opt[:, jnp.newaxis, ...]
    ], axis=1)


def reconstruct(rng, model, state, x, imgdir):
    p_rec_step = jax.pmap(partial(
        rec_step, rec_fn=model.reconstruct_fn()
    ), axis_name='batch')

    rng, subkey = vsplit(rng)
    comparison = p_rec_step(state, x, subkey)
    comparison = jnp.reshape(comparison, (-1,) + comparison.shape[-3:])
    save_image(comparison, imgdir / 'rec_final.png', nrow=48)


def rw_step(state, x, rng, model, num_steps, stride, step_size=0.01):
    x = jnp.reshape(x, (-1,) + x.shape[-3:])
    mean, _, resolution = model.encode_fn()(state.params, x)
    x_rec = model.decode_fn()(state.params, mean, resolution)

    def decode(z):
        return jnp.reshape(
            model.decode_fn()(state.params, z, resolution),
            z.shape[:-1] + (-1,)
        )

    x_rec_opt, z = model.reconstruct_fn()(state.params, x)

    manifold = ImmersedSubmanifold(decode)
    volume = manifold.volume(z)
    w, _ = jnp.linalg.eigh(manifold.metric_tensor(z))


    zs = manifold.random_walk(rng, z, step_size=step_size, num_steps=num_steps * stride)
    zs = jnp.reshape(zs, (num_steps, stride) + zs.shape[1:])[:, -1, ...]
    zs = jnp.swapaxes(zs, 0, 1)
    xs = decode(zs)
    xs = jnp.reshape(xs, zs.shape[:-1] + x.shape[-3:])

    return jnp.concatenate([
        x[:, jnp.newaxis, ...],
        x_rec[:, jnp.newaxis, ...],
        x_rec_opt[:, jnp.newaxis, ...],
        xs
    ], axis=1), volume, jnp.amin(w, axis=-1), jnp.amax(w, axis=-1), jnp.median(w, axis=-1), jnp.mean(w, axis=-1)


def random_walk(config, rng, model, state, x, imgdir, num_steps=16, stride=64):
    p_rw_step = jax.pmap(partial(
        rw_step, model=model,
        num_steps=num_steps, stride=stride
    ), axis_name='batch')

    rng, subkey = vsplit(rng)
    comparison, volume, min, max, median, mean = p_rw_step(state, x, subkey)
    comparison = jnp.reshape(comparison, (-1,) + comparison.shape[-3:])
    save_image(comparison, imgdir / 'rw.png', nrow=3 + num_steps)
    print('Volumes:', volume)
    print('Min:', min)
    print('Median:', median)
    print('Mean:', mean)
    print('Max:', max)


def run(config: mlc.ConfigDict, workdir: str):
    ckptdir = Path(workdir) / 'checkpoints'
    imgdir = Path(workdir) / 'images'
    imgdir.mkdir(exist_ok=True)

    rng = random.PRNGKey(0)

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
        shuffle=True, num_workers=8, drop_last=True
    )

    testloader = create_input_iter(testloader)

    x = next(testloader)[:, :128, ...]

    model = create_model(config)
    learning_rate_fn = create_learning_rate_fn(config, 100)
    state = create_train_state(rng, config, model, learning_rate_fn)
    state = restore_checkpoint(state, ckptdir)
    state = jax_utils.replicate(state)
    rng = jax.random.split(rng, jax.local_device_count())

    if FLAGS.rec:
        reconstruct(rng, model, state, x, imgdir)
    if FLAGS.rw > 0:
        random_walk(config, rng, model, state, x[:, :8, ...], imgdir, num_steps=FLAGS.rw)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  run(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)