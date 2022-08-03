from functools import partial
import time
from pathlib import Path

from absl import logging
from absl import logging
from flax import jax_utils, struct
from flax.optim import dynamic_scale as dynamic_scale_lib
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from torch.utils.data import DataLoader
import numpy as np
from heatx.models.generative.sde.model import SDE

from heatx.trainutil import TrainState, \
    save_checkpoint, restore_checkpoint, \
    dynamic_scale_update, dereplicate_metrics, \
    local_sharding, vsplit
from heatx.datautil.dataloader import FaceDataset
from heatx.metricwriter import LoggingWriter
from heatx.models.regressor import CondImageRegressor
import heatx.models.encoder.resnet as resnets
from heatx.manifold import ImageScore, EuclideanImage

from train import create_model, create_train_state, \
    create_learning_rate_fn, create_input_iter, MAX_AGE_PLUS_1

from heatx.models.generative.sde.train import \
    create_model as create_sde_model, \
    create_train_state as create_sde_train_state



def create_pde_solution_model(config):
    dtype = jnp.float16 if config.half_precision else jnp.float32
    return CondImageRegressor(
        targets=1, act=nn.sigmoid, dtype=dtype,
        encoder_cls=getattr(resnets, 'Cond' + config.encoder)
    )


def mse_loss(y_true, y_pred):
    return jnp.sum(jnp.mean((y_true - y_pred) ** 2, axis=0))


def sharded_random_t(rng, batch, max_scale=1.0):
    x = batch['image']
    total_batch_size = x.shape[0] * x.shape[1]
    grid = jnp.arange(total_batch_size, dtype=x.dtype) / total_batch_size
    offset = jax.random.uniform(rng, dtype=grid.dtype)
    grid = (grid + offset) % 1.0
    grid = grid * max_scale
    return local_sharding(jax.random.permutation(rng, grid))


def batch_loss(rng, params, apply_fn, model, batch, t):
    def model_fn(_x_):
        return model.apply_fn({'params': model.params}, _x_)[..., 0]

    model_grad_fn = jax.vmap(jax.value_and_grad(model_fn))

    def pde_fn(_x_, _t_):
        return apply_fn({'params': params}, _x_, _t_)[..., 0]

    pde_grad_fn = jax.vmap(jax.value_and_grad(pde_fn, argnums=0))

    x = batch['image']
    manifold = EuclideanImage()
    xt = manifold.brownian(rng, x, t)

    xt = jnp.concatenate([x, xt], axis=0)
    x = jnp.concatenate([x, x], axis=0)
    t = jnp.concatenate([jnp.zeros_like(t), t], axis=0)

    yt, gt = model_grad_fn(xt)
    yt_hat, gt_hat = pde_grad_fn(x, t)

    y0, yt = jnp.split(yt, 2, axis=0)
    g0, gt = jnp.split(gt, 2, axis=0)
    y0_hat, yt_hat = jnp.split(yt_hat, 2, axis=0)
    g0_hat, gt_hat = jnp.split(gt_hat, 2, axis=0)

    boundary_mse = mse_loss(y0 * MAX_AGE_PLUS_1, y0_hat * MAX_AGE_PLUS_1)
    solution_mse = mse_loss(yt * MAX_AGE_PLUS_1, yt_hat * MAX_AGE_PLUS_1)
    boundary_grad_mse = mse_loss(g0, g0_hat)
    solution_grad_mse = mse_loss(gt, gt_hat)

    loss = boundary_mse + solution_mse + \
        boundary_grad_mse + solution_grad_mse
    return loss, boundary_mse, solution_mse, boundary_grad_mse, solution_grad_mse


def train_step(rng, state, model, batch, t, learning_rate_fn):
    step = state.step

    def loss_fn(params):
        loss, bmse, smse, bgmse, sgmse = batch_loss(
            rng, params, state.apply_fn, model, batch, t)
        return loss, (bmse, smse, bgmse, sgmse)

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
    bmse, smse, bgmse, sgmse = aux[1]
    metrics = {
        'loss': loss,
        'boundary': bmse, 'solution': smse,
        'grad@boundary': bgmse, 'grad@solution': sgmse
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    metrics['learning_rate'] = learning_rate_fn(step)

    new_state = state.apply_gradients(grads=grads)
    if dynamic_scale:
        new_state = dynamic_scale_update(state, new_state, is_finite)
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics


def eval_step(rng, state, model, batch, t):
    loss, bmse, smse, bgmse, sgmse = batch_loss(
        rng, state.params, state.apply_fn, model, batch, t)
    metrics = {
        'loss': loss,
        'boundary': bmse, 'solution': smse,
        'grad@boundary': bgmse, 'grad@solution': sgmse
    }
    return lax.pmean(metrics, axis_name='batch')


class ImplicitManifold(struct.PyTreeNode):
    state: TrainState
    sde: SDE = struct.field(pytree_node=False)


def manifold_batch_loss(rng, params, apply_fn, model, manifold, batch, t):
    def model_fn(_x_):
        return model.apply_fn({'params': model.params}, _x_)[..., 0]

    def pde_fn(_x_, _t_):
        return apply_fn({'params': params}, _x_, _t_)[..., 0]

    def noise_pred_fn(_xt_, _t_):
        return manifold.state.apply_fn({'params': manifold.state.params}, _xt_, _t_)

    implicit = ImageScore(manifold.sde, noise_pred_fn, nfe=32)

    x = batch['image']
    xt = implicit.brownian(rng, x, t)

    xt = jnp.concatenate([x, xt], axis=0)
    x = jnp.concatenate([x, x], axis=0)
    t = jnp.concatenate([jnp.zeros_like(t), t], axis=0)

    yt = model_fn(xt)
    yt_hat = pde_fn(x, t)

    y0, yt = jnp.split(yt, 2, axis=0)
    y0_hat, yt_hat = jnp.split(yt_hat, 2, axis=0)

    boundary_mse = mse_loss(y0 * MAX_AGE_PLUS_1, y0_hat * MAX_AGE_PLUS_1)
    solution_mse = mse_loss(yt * MAX_AGE_PLUS_1, yt_hat * MAX_AGE_PLUS_1)

    loss = boundary_mse + solution_mse
    return loss, boundary_mse, solution_mse


def manifold_train_step(rng, state, model, batch, t, manifold, learning_rate_fn):
    step = state.step

    def loss_fn(params):
        loss, bmse, smse = manifold_batch_loss(
            rng, params, state.apply_fn, model, manifold, batch, t)
        return loss, (bmse, smse)

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
    bmse, smse = aux[1]
    metrics = {
        'loss': loss,
        'boundary': bmse, 'solution': smse
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    metrics['learning_rate'] = learning_rate_fn(step)

    new_state = state.apply_gradients(grads=grads)
    if dynamic_scale:
        new_state = dynamic_scale_update(state, new_state, is_finite)
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics


def manifold_eval_step(rng, state, model, batch, t, manifold):
    loss, bmse, smse = manifold_batch_loss(
        rng, state.params, state.apply_fn, model, manifold, batch, t)
    metrics = {
        'loss': loss,
        'boundary': bmse, 'solution': smse
    }
    return lax.pmean(metrics, axis_name='batch')


def create_pde_solution_train_state(rng, config: mlc.ConfigDict,
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
    init_t = jnp.ones(config.batch_size, dtype=model.dtype)

    state = TrainState.create(
        apply_fn=model.apply,
        params=jax.jit(model.init)(key, init_data, init_t)['params'],
        tx=optax.adamw(
            learning_rate=learning_rate_fn,
            weight_decay=config.weight_decay
        ),
        dynamic_scale=dynamic_scale
    )
    return state


def train_and_evaluate(
        config: mlc.ConfigDict, workdir: str,
        model_config: mlc.ConfigDict, model_workdir: str,
        manifold_config: mlc.ConfigDict, manifold_workdir: str) -> TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where checkpoints and summaries are written to.

    Returns:
      Final TrainState.
    """
    model_ckpt_dir = Path(model_workdir) / 'checkpoints'
    imgdir = Path(workdir) / 'images'
    imgdir.mkdir(parents=True, exist_ok=True)
    ckptdir = Path(workdir) / 'checkpoints'
    ckptdir.mkdir(parents=True, exist_ok=True)

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
        FaceDataset(datadir / 'train', aug=config.aug, **dataset_kwargs) +
        FaceDataset(datadir / 'test', aug=config.aug, **dataset_kwargs),
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

    model = create_model(model_config)
    tmp_lr_fn = create_learning_rate_fn(model_config, steps_per_epoch)
    model = create_train_state(rng, model_config, model, tmp_lr_fn)
    model = restore_checkpoint(model, model_ckpt_dir)
    logging.info(f'Model: state.step = {int(model.step)}')
    model = jax_utils.replicate(model)

    if manifold_workdir is not None:
        sde, score = create_sde_model(manifold_config)
        tmp_lr_fn = create_learning_rate_fn(manifold_config, steps_per_epoch)
        score = create_sde_train_state(rng, manifold_config, score, tmp_lr_fn)
        score = restore_checkpoint(score, Path(manifold_workdir) / 'checkpoints')
        logging.info(f'SDE model: state.step = {int(score.step)}')
        manifold = ImplicitManifold(score, sde)
        manifold = jax_utils.replicate(manifold)
    else:
        manifold = None

    solution = create_pde_solution_model(config)
    learning_rate_fn = create_learning_rate_fn(config, steps_per_epoch)
    state = create_pde_solution_train_state(rng, config, solution, learning_rate_fn)
    state = restore_checkpoint(state, ckptdir)
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    rng = jax.random.split(rng, jax.local_device_count())

    if manifold is None:
        p_train_step = jax.pmap(partial(
            train_step,
            learning_rate_fn=learning_rate_fn
        ), axis_name='batch')
        p_eval_step = jax.pmap(eval_step, axis_name='batch')
    else:
        p_train_step = jax.pmap(partial(
            manifold_train_step,
            learning_rate_fn=learning_rate_fn
        ), axis_name='batch')
        p_train_step = partial(p_train_step, manifold=manifold)

        p_eval_step = jax.pmap(manifold_eval_step, axis_name='batch')
        p_eval_step = partial(p_eval_step, manifold=manifold)

    def eval(_state_, _rng_, _step_):
        epoch = _step_ // steps_per_epoch
        eval_metrics = []

        for _ in range(steps_per_eval):
            eval_batch = next(testloader)
            t = sharded_random_t(_rng_[0], eval_batch)
            metrics = p_eval_step(_rng_, _state_, model, eval_batch, t)
            eval_metrics.append(metrics)

        eval_metrics = dereplicate_metrics(eval_metrics)
        summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
        logging.info(
            'eval epoch: %d, loss: %.6f, boundary mse: %.6f, solution mse: %.6f',
            epoch, summary['loss'], summary['boundary'], summary['solution']
        )
        writer.write_scalars(_step_ + 1, {
            f'eval_{k}': v for k, v in summary.items()
        })
        writer.flush()

    train_metrics = []
    hooks = []
    train_metrics_last_t = time.time()

    logging.info('Initial compilation, this might take some minutes...')

    for step, batch in zip(range(step_offset, num_steps), trainloader):
        epoch = step // steps_per_epoch

        rng, subkey = vsplit(rng)
        t = sharded_random_t(subkey[0], batch)
        state, metrics = p_train_step(subkey, state, model, batch, t)

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

        if (step + 1) % steps_per_epoch == 0 and (
            epoch + 1 == config.num_epochs or
                epoch % config.eval_every_epochs == 0):
            rng, subkey = vsplit(rng)
            eval(state, subkey, step)

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            save_checkpoint(state, ckptdir)

    eval(state, rng, num_steps)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    logging.info('Done.')

    return state
