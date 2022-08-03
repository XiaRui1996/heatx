from functools import partial
import time
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
    save_image
from heatx.models.generative.vae.model import kl_divergence, ConvVAE
from heatx.datautil.dataloader import FaceDataset
from heatx.metricwriter import LoggingWriter
import heatx.models.encoder as encoders
import heatx.models.decoder as decoders


def create_model(config):
    return ConvVAE(
        channels=config.image_channels,
        latents=config.latents,
        dtype=jnp.float16 if config.half_precision else jnp.float32,
        encoder_cls=getattr(encoders, config.encoder),
        decoder_cls=getattr(decoders, config.decoder)
    )


def vae_loss(x, x_recon, z_mean, z_logvar):
    err = jnp.sum(jnp.mean((x - x_recon) ** 2, axis=0))
    kld = jnp.mean(kl_divergence(z_mean, z_logvar))
    return err, kld


def compute_metrics(x, x_recon, z_mean, z_logvar):
    err, kld = vae_loss(x, x_recon, z_mean, z_logvar)
    metrics = {'err': err, 'kld': kld}
    return lax.pmean(metrics, axis_name='batch')


def create_learning_rate_fn(config: mlc.ConfigDict, steps_per_epoch: int):
    return warmup_cos_decay_lr_schedule_fn(
        base_learning_rate=config.learning_rate * config.batch_size / 256.,
        num_epochs=config.num_epochs,
        warmup_epochs=config.warmup_epochs,
        steps_per_epoch=steps_per_epoch
    )


def create_kl_anneal_fn(config: mlc.ConfigDict, steps_per_epoch: int):
    return optax.linear_schedule(
        init_value=0., end_value=config.kl_anneal_end,
        transition_steps=config.kl_anneal_epochs * steps_per_epoch)


def create_input_iter(it):
    return map(
        lambda batch: batch[0],
        create_cyclic_sharded_iter(it)
    )


def train_step(state, x, rng, learning_rate_fn, kl_anneal_fn):
    step = state.step
    anneal = kl_anneal_fn(step)

    def loss_fn(params):
        x_recon, z_mean, z_logvar = state.apply_fn({'params': params}, x, rng)
        err, kld = vae_loss(x, x_recon, z_mean, z_logvar)
        loss = err + anneal * kld
        return loss, (err, kld)

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
    err, kld = aux[1]
    metrics = {'loss': loss, 'err': err, 'kld': kld}
    metrics = lax.pmean(metrics, axis_name='batch')
    metrics['anneal'] = anneal
    metrics['learning_rate'] = learning_rate_fn(step)

    new_state = state.apply_gradients(grads=grads)
    if dynamic_scale:
        new_state = dynamic_scale_update(state, new_state, is_finite)
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics


def eval_step(state, x, rng):
    x_recon, z_mean, z_logvar = state.apply_fn(
        {'params': state.params},
        x, rng, training=False, mutable=False
    )
    return compute_metrics(x, x_recon, z_mean, z_logvar)


def rec_step(state, x, rng):
    x_recon, _, _ = state.apply_fn(
        {'params': state.params},
        x, rng, training=False, mutable=False
    )
    return jnp.concatenate([
        x[:, jnp.newaxis, ...],
        x_recon[:, jnp.newaxis, ...]
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

    state = TrainState.create(
        apply_fn=model.apply,
        params=jax.jit(model.init)(key, init_data, rng)['params'],
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

    for _ in range(steps_per_eval):
        rec_batch = next(testloader)
    rec_batch = rec_batch[:, :128, ...]

    model = create_model(config)
    kl_anneal_fn = create_kl_anneal_fn(config, steps_per_epoch)
    learning_rate_fn = create_learning_rate_fn(config, steps_per_epoch)
    state = create_train_state(rng, config, model, learning_rate_fn)
    state = restore_checkpoint(state, ckptdir)
    step_offset = int(state.step)
    state = jax_utils.replicate(state)
    rng = jax.random.split(rng, jax.local_device_count())
    vsplit = lambda x: jax.vmap(jax.random.split)(x).swapaxes(0, 1)

    p_train_step = jax.pmap(partial(
        train_step,
        learning_rate_fn=learning_rate_fn,
        kl_anneal_fn=kl_anneal_fn
    ), axis_name='batch')
    p_eval_step = jax.pmap(eval_step, axis_name='batch')
    p_rec_step = jax.pmap(rec_step, axis_name='batch')

    train_metrics = []
    hooks = []
    train_metrics_last_t = time.time()

    logging.info('Initial compilation, this might take some minutes...')

    for step, batch in zip(range(step_offset, num_steps), trainloader):
        rng, subkey = vsplit(rng)
        state, metrics = p_train_step(state, batch, subkey)
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

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            eval_metrics = []

            for _ in range(steps_per_eval):
                eval_batch = next(testloader)
                rng, subkey = vsplit(rng)
                metrics = p_eval_step(state, eval_batch, subkey)
                eval_metrics.append(metrics)

            eval_metrics = dereplicate_metrics(eval_metrics)
            summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info('eval epoch: %d, err: %.4f, kld: %.4f',
                         epoch, summary['err'], summary['kld'])
            writer.write_scalars(step + 1, {f'eval_{k}': v for k, v in summary.items()})
            writer.flush()

            rng, subkey = vsplit(rng)
            comparison = p_rec_step(state, rec_batch, subkey)
            comparison = jnp.reshape(comparison, (-1,) + comparison.shape[-3:])
            save_image(comparison, imgdir / f'rec_{epoch}.png', nrow=32)

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            save_checkpoint(state, ckptdir)
    
    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    logging.info('Done.')

    return state
