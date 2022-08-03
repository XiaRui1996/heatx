from functools import partial
import time
from pathlib import Path

from absl import logging
from flax import jax_utils
from flax.optim import dynamic_scale as dynamic_scale_lib
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from torch.utils.data import DataLoader
import numpy as np

from heatx.trainutil import TrainState, \
    create_cyclic_sharded_iter, \
    save_checkpoint, restore_checkpoint, \
    warmup_cos_decay_lr_schedule_fn, \
    dynamic_scale_update, \
    dereplicate_metrics, vsplit
from heatx.datautil.dataloader import FaceDataset
from heatx.metricwriter import LoggingWriter
from heatx.models.regressor import ImageRegressor
import heatx.models.encoder.resnet as resnets


def create_model(config):
    return ImageRegressor(
        targets=1, act=nn.sigmoid,
        dtype=jnp.float16 if config.half_precision else jnp.float32,
        encoder_cls=getattr(resnets, config.encoder)
    )


def mse_loss(y_true, y_pred):
    return jnp.sum(jnp.mean((y_true - y_pred) ** 2, axis=0))


def create_learning_rate_fn(config: mlc.ConfigDict, steps_per_epoch: int):
    return warmup_cos_decay_lr_schedule_fn(
        base_learning_rate=config.learning_rate * config.batch_size / 256.,
        num_epochs=config.num_epochs,
        warmup_epochs=config.warmup_epochs,
        steps_per_epoch=steps_per_epoch
    )


MAX_AGE_PLUS_1 = 111.0


def create_input_iter(it):
    return map(
        lambda batch: {
            'image': (batch[0] - 0.5) * 2,  # [0, 1] -> [-1, 1]
            'label': batch[1] / MAX_AGE_PLUS_1  # [1, MAX_AGE] -> (0, 1)
        },
        create_cyclic_sharded_iter(it)
    )


def batch_loss(rng, params, apply_fn, batch, smooth=False):
    x, y = batch['image'], batch['label']
    pred = apply_fn({'params': params}, x)[..., 0]
    noise = jnp.zeros_like(y) if not smooth \
        else jax.random.uniform(rng, shape=y.shape, minval=-1.0, maxval=1.0)
    y_smooth = (y * MAX_AGE_PLUS_1 + noise) / MAX_AGE_PLUS_1
    loss = mse_loss(y_smooth, pred)
    mse = mse_loss(y * MAX_AGE_PLUS_1, pred * MAX_AGE_PLUS_1)
    mae = jnp.mean(jnp.abs(y - pred) * MAX_AGE_PLUS_1)
    return loss, mse, mae


def train_step(rng, state, batch, learning_rate_fn):
    step = state.step

    def loss_fn(params):
        loss, mse, mae = batch_loss(rng, params, state.apply_fn, batch, smooth=True)
        return loss, (mse, mae)

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
    mse, mae = aux[1]
    metrics = {'loss': loss, 'mse': mse, 'mae': mae}
    metrics = lax.pmean(metrics, axis_name='batch')
    metrics['learning_rate'] = learning_rate_fn(step)

    new_state = state.apply_gradients(grads=grads)
    if dynamic_scale:
        new_state = dynamic_scale_update(state, new_state, is_finite)
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics


def eval_step(rng, state, batch):
    loss, mse, mae = batch_loss(rng, state.params, state.apply_fn, batch, smooth=False)
    metrics = {'loss': loss, 'mse': mse, 'mae': mae}
    return lax.pmean(metrics, axis_name='batch')


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

    state = TrainState.create(
        apply_fn=model.apply,
        params=jax.jit(model.init)(key, init_data)['params'],
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

    datadir = Path(config.datadir)
    input_dtype = np.float16 if config.half_precision else np.float32

    dataset_kwargs = dict(
        width=config.image_size, height=config.image_size,
        dtype=input_dtype, unit_interval=True
    )
    trainloader = DataLoader(
        FaceDataset(datadir / 'train', aug=config.aug, **dataset_kwargs),
        batch_size=local_batch_size,
        shuffle=True, num_workers=8, drop_last=True
    )
    testloader = DataLoader(
        FaceDataset(datadir / 'test', aug=False, **dataset_kwargs),
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

    model = create_model(config)
    learning_rate_fn = create_learning_rate_fn(config, steps_per_epoch)
    state = create_train_state(rng, config, model, learning_rate_fn)
    state = restore_checkpoint(state, ckptdir)
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    rng = jax.random.split(rng, jax.local_device_count())

    p_train_step = jax.pmap(partial(
        train_step,
        learning_rate_fn=learning_rate_fn
    ), axis_name='batch')
    p_eval_step = jax.pmap(eval_step, axis_name='batch')

    train_metrics = []
    hooks = []
    train_metrics_last_t = time.time()

    logging.info('Initial compilation, this might take some minutes...')

    for step, batch in zip(range(step_offset, num_steps), trainloader):
        rng, subkey = vsplit(rng)
        state, metrics = p_train_step(subkey, state, batch)
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
                metrics = p_eval_step(subkey, state, eval_batch)
                eval_metrics.append(metrics)

            eval_metrics = dereplicate_metrics(eval_metrics)
            summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info(
                'eval epoch: %d, loss: %.4f, mse: %.4f, mae: %.4f',
                epoch, summary['loss'], summary['mse'], summary['mae']
            )
            writer.write_scalars(step + 1, {
                f'eval_{k}': v for k, v in summary.items()
            })
            writer.flush()

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            save_checkpoint(state, ckptdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    logging.info('Done.')

    return state
