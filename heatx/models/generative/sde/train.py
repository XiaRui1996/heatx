from functools import partial, reduce
import time
import operator
from pathlib import Path

from absl import logging
from flax import jax_utils
from flax.optim import dynamic_scale as dynamic_scale_lib
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from torch.utils.data import DataLoader
import numpy as np

from heatx.trainutil import create_cyclic_sharded_iter, \
    save_checkpoint, restore_checkpoint, TrainState, \
    warmup_cos_decay_lr_schedule_fn, \
    dynamic_scale_update, dereplicate_metrics, \
    save_image, local_sharding
from heatx.models.generative.sde.model import UNetPredictor
import heatx.models.generative.sde.model as sdes
from heatx.datautil.dataloader import FaceDataset
from heatx.metricwriter import LoggingWriter
import heatx.models.decoder.unet as unets


def create_model(config):
    sde = getattr(sdes, config.sde)(minSNR=config.minSNR)
    unet = UNetPredictor(
        channels=config.image_channels,
        dtype=jnp.float16 if config.half_precision else jnp.float32,
        unet_cls=getattr(unets, config.unet)
    )
    return sde, unet


def mse_loss(y_true, y_pred):
    return jnp.sum(jnp.mean((y_true - y_pred) ** 2, axis=0))


def create_learning_rate_fn(config: mlc.ConfigDict, steps_per_epoch: int):
    return warmup_cos_decay_lr_schedule_fn(
        base_learning_rate=config.learning_rate * config.batch_size / 256.,
        num_epochs=config.num_epochs,
        warmup_epochs=config.warmup_epochs,
        steps_per_epoch=steps_per_epoch
    )


def create_input_iter(it):
    return map(
        lambda batch: (batch[0] - 0.5) * 2,  # [0, 1] -> [-1, 1]
        create_cyclic_sharded_iter(it)
    )


def sharded_random_t(rng, x):
    total_batch_size = x.shape[0] * x.shape[1]
    grid = jnp.arange(total_batch_size, dtype=x.dtype) / total_batch_size
    offset = jax.random.uniform(rng, dtype=grid.dtype)
    grid = (grid + offset) % 1.0
    return local_sharding(jax.random.permutation(rng, grid))


def batch_loss(rng, params, apply_fn, sde, x, t):
    rng, subkey = jax.random.split(rng)
    xt, noise = sde.forward(subkey, x, t)
    pred = apply_fn({'params': params}, xt, t)
    return mse_loss(noise, pred)


def train_step(rng, state, x, t, sde, learning_rate_fn):
    step = state.step

    def loss_fn(params):
        loss = batch_loss(rng, params, state.apply_fn, sde, x, t)
        return loss, (loss,)

    dynamic_scale = state.dynamic_scale

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(
            loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_finite, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = lax.pmean(grads, axis_name='batch')

    loss = aux[0]
    metrics = {'loss': loss}
    metrics = lax.pmean(metrics, axis_name='batch')
    metrics['learning_rate'] = learning_rate_fn(step)

    new_state = state.apply_gradients(grads=grads)
    if dynamic_scale:
        new_state = dynamic_scale_update(state, new_state, is_finite)
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics


def eval_step(rng, state, x, t, sde):
    loss = batch_loss(rng, state.params, state.apply_fn, sde, x, t)
    metrics = {'loss': loss}
    return lax.pmean(metrics, axis_name='batch')


def rec_step(rng, state, x, sde, num_steps=10):
    grid = (0.5 + np.arange(num_steps, dtype=x.dtype)) / num_steps
    grid = grid[..., np.newaxis]
    xt, _ = sde.forward(rng, x, grid)
    pred = state.apply_fn({'params': state.params}, xt, grid)
    x_rec = sde.rough_inverse(xt, grid, pred)

    return jnp.concatenate([
        x[:, jnp.newaxis, ...],
        np.swapaxes(xt, 0, 1),
        x[:, jnp.newaxis, ...],
        np.swapaxes(x_rec, 0, 1)
    ], axis=1)


def create_train_state(rng, config: mlc.ConfigDict,
                       model, learning_rate_fn):
    """Create initial training state."""

    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()
    else:
        dynamic_scale = None

    rng, key = jax.random.split(rng)

    input_shape = (
        config.batch_size,
        config.image_size, config.image_size,
        config.image_channels
    )
    init_data = jnp.ones(input_shape, dtype=model.dtype)
    init_scale = jnp.ones(config.batch_size, dtype=model.dtype)

    state = TrainState.create(
        apply_fn=model.apply,
        params=jax.jit(model.init)(key, init_data, init_scale)['params'],
        tx=optax.adamw(
            learning_rate=learning_rate_fn,
            weight_decay=config.weight_decay
        ),
        dynamic_scale=dynamic_scale
    )
    return state


def train_and_evaluate(config: mlc.ConfigDict,
                       workdir: str) -> TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where checkpoints and summaries are written to.

    Returns:
      Final TrainState.
    """
    ckptdir = Path(workdir) / 'checkpoints'
    ckptdir.mkdir(parents=True, exist_ok=True)
    imgdir = Path(workdir) / 'images'
    imgdir.mkdir(exist_ok=True)

    writer = LoggingWriter()
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
    trainloader = DataLoader(
        reduce(operator.add, [
            FaceDataset(Path(datadir), aug=config.train_augs[i], **dataset_kwargs)
            for i, datadir in enumerate(config.train_datadirs)
        ]),
        batch_size=local_batch_size,
        shuffle=True, num_workers=8, drop_last=True
    )
    testloader = DataLoader(
        reduce(operator.add, [
            FaceDataset(Path(datadir), aug=config.eval_augs[i], **dataset_kwargs)
            for i, datadir in enumerate(config.eval_datadirs)
        ]),
        batch_size=local_batch_size * 2,
        shuffle=True, num_workers=8, drop_last=True
    )

    steps_per_epoch = len(trainloader.dataset) // config.batch_size
    steps_per_checkpoint = steps_per_epoch * 10
    num_steps = int(steps_per_epoch * config.num_epochs) \
        if config.num_train_steps == -1 else config.num_train_steps
    steps_per_eval = (len(testloader.dataset) // (config.batch_size * 2)) \
        if config.steps_per_eval == -1 else config.steps_per_eval

    trainloader = create_input_iter(trainloader)
    testloader = create_input_iter(testloader)

    for _ in range(steps_per_eval):
        rec_batch = next(testloader)
    rec_batch = rec_batch[:, :8, ...]

    sde, model = create_model(config)
    learning_rate_fn = create_learning_rate_fn(config, steps_per_epoch)
    state = create_train_state(rng, config, model, learning_rate_fn)
    state = restore_checkpoint(state, ckptdir)
    step_offset = int(state.step)
    state = jax_utils.replicate(state)
    rng = jax.random.split(rng, jax.local_device_count())
    vsplit = lambda x: jax.vmap(jax.random.split)(x).swapaxes(0, 1)

    p_train_step = jax.pmap(partial(
        train_step,
        sde=sde, learning_rate_fn=learning_rate_fn
    ), axis_name='batch')
    p_eval_step = jax.pmap(partial(
        eval_step,
        sde=sde
    ), axis_name='batch')
    p_rec_step = jax.pmap(partial(
        rec_step,
        sde=sde, num_steps=config.rec_grid_size
    ), axis_name='batch')

    train_metrics = []
    hooks = []
    train_metrics_last_t = time.time()

    logging.info('Initial compilation, this might take some minutes...')

    for step, batch in zip(range(step_offset, num_steps), trainloader):
        rng, subkey = vsplit(rng)
        t = sharded_random_t(subkey[0], batch)
        state, metrics = p_train_step(subkey, state, batch, t)
        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info('Initial compilation completed.')

        if config.get('log_every_steps'):
            train_metrics.append(metrics)
            if (step + 1) % config.log_every_steps == 0:
                train_metrics = dereplicate_metrics(train_metrics)
                summary = jax.tree_map(lambda x: x.mean(), train_metrics)
                summary = {f'train_{k}': v for k, v in summary.items()}
                summary['steps_per_second'] = config.log_every_steps / (
                    time.time() - train_metrics_last_t)
                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

        epoch = step // steps_per_epoch

        if (step + 1) % steps_per_epoch == 0 and (
            epoch + 1 == config.num_epochs or \
                epoch % config.eval_every_epochs == 0):

            eval_metrics = []

            for _ in range(steps_per_eval):
                eval_batch = next(testloader)
                rng, subkey = vsplit(rng)
                t = sharded_random_t(subkey[0], eval_batch)
                metrics = p_eval_step(subkey, state, eval_batch, t)
                eval_metrics.append(metrics)

            eval_metrics = dereplicate_metrics(eval_metrics)
            summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info('eval epoch: %d, loss: %.4f', epoch, summary['loss'])
            writer.write_scalars(step + 1, {
                f'eval_{k}': v for k, v in summary.items()
            })
            writer.flush()

            if (epoch % 10 == 0) or (epoch + 1 == config.num_epochs):
                rng, subkey = vsplit(rng)
                comparison = p_rec_step(subkey, state, rec_batch)
                comparison = jnp.reshape(comparison, (-1,) + comparison.shape[-3:])
                save_image(
                    comparison / 2 + 0.5,  # [-1, 1] -> [0, 1],
                    imgdir / f'rec_{epoch}.png',
                    nrow=config.rec_grid_size + 1
                )

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            save_checkpoint(state, ckptdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    logging.info('Done.')

    return state
