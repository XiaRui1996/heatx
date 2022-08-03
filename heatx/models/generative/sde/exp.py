#!/usr/bin/env python
from functools import partial, reduce
import operator
from pathlib import Path

import jax
import jax.numpy as jnp

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

from heatx.trainutil import vsplit, save_image, restore_checkpoint
from heatx.datautil.dataloader import FaceDataset
from heatx.manifold import ImageScore

from train import create_model, create_train_state, \
    create_learning_rate_fn, create_input_iter
from model import SDE



FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_bool('rec', False, 'Reconstruct images.')
flags.DEFINE_integer('rw', 32, 'Number of steps to randomly walk.')
flags.DEFINE_float('ss', 0.01, 'Step size of random walk.')
flags.DEFINE_integer('stride', 1, 'Stride.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training hyperparameter configuration.',
    lock_config=True)



def rw_step(rng, state, x, sde: SDE, num_steps, step_size, stride, *, t0=0.001, nfe=32):
    x = jnp.reshape(x, (-1,) + x.shape[-3:])
    t0 = jnp.asarray(t0, dtype=x.dtype)
    step_size = jnp.asarray(step_size * stride, dtype=x.dtype)
    grid = (1.0 + jnp.arange(num_steps, dtype=x.dtype)) * step_size
    grid = grid[..., jnp.newaxis]

    def noise_pred_fn(xt, t):
        return state.apply_fn({'params': state.params}, xt, t)

    manifold = ImageScore(sde, noise_pred_fn, t0=t0, nfe=nfe)
    xs = manifold.brownian(rng, x, grid)

    return jnp.concatenate([
        x[:, jnp.newaxis, ...],
        jnp.swapaxes(xs, 0, 1)
    ], axis=1)


def random_walk(config, rng, state, sde, x, imgdir, num_steps=16, step_size=0.01, stride=10):
    p_rw_step = jax.pmap(partial(
        rw_step,
        sde=sde, num_steps=num_steps,
        step_size=step_size, stride=stride
    ), axis_name='batch')

    rng, subkey = vsplit(rng)
    comparison = p_rw_step(subkey, state, x)
    comparison = jnp.reshape(comparison, (-1,) + comparison.shape[-3:])
    comparison = comparison / 2 + 0.5  # [-1, 1] -> [0, 1]
    save_image(comparison, imgdir / 'rw.png', nrow=1 + num_steps)


def run(config: mlc.ConfigDict, workdir: str):
    ckptdir = Path(workdir) / 'checkpoints'
    imgdir = Path(workdir) / 'images'
    imgdir.mkdir(exist_ok=True)

    rng = jax.random.PRNGKey(0)

    if config.batch_size % jax.device_count() > 0:
        raise ValueError(
            'Batch size must be divisible by the number of devices')
    local_batch_size = config.batch_size // jax.process_count()

    input_dtype = np.float16 if config.half_precision else np.float32

    dataset_kwargs = dict(
        width=config.image_size, height=config.image_size,
        dtype=input_dtype, unit_interval=True
    )
    testloader = DataLoader(
        reduce(operator.add, [
            FaceDataset(Path(datadir), aug=config.eval_augs[i], **dataset_kwargs)
            for i, datadir in enumerate(config.eval_datadirs)
        ]),
        batch_size=local_batch_size * 2,
        shuffle=True, num_workers=8, drop_last=True
    )

    testloader = create_input_iter(testloader)

    x = next(testloader)[:, :128, ...]

    sde, model = create_model(config)
    learning_rate_fn = create_learning_rate_fn(config, 100)
    state = create_train_state(rng, config, model, learning_rate_fn)
    state = restore_checkpoint(state, ckptdir)
    state = jax_utils.replicate(state)
    rng = jax.random.split(rng, jax.local_device_count())

    if FLAGS.rw > 0:
        random_walk(
            config, rng, state, sde, x[:, :8, ...], imgdir,
            num_steps=FLAGS.rw, step_size=FLAGS.ss, stride=FLAGS.stride
        )

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