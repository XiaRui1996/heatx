from functools import partial
import time
from pathlib import Path
from typing import Union

from absl import logging
from absl import logging
from flax import jax_utils, struct
from flax.optim import dynamic_scale as dynamic_scale_lib
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from torch.utils.data import DataLoader
import numpy as np

from heatx.trainutil import TrainState, \
    save_checkpoint, restore_checkpoint, \
    dynamic_scale_update, dereplicate_metrics, \
    local_sharding, vsplit, save_image
from heatx.datautil.dataloader import SyntheticDataset
from heatx.metricwriter import LoggingWriter
from heatx.models.regressor import CondImageRegressor
import heatx.models.encoder.resnet as resnets
from heatx.manifold import ImmersedImage, EuclideanImage

from train import create_model, create_train_state, \
    create_learning_rate_fn, create_input_iter, make_worker_init_fn


def create_pde_solution_model(config):
    dtype = jnp.float16 if config.half_precision else jnp.float32
    return CondImageRegressor(
        targets=1, dtype=dtype,
        encoder_cls=getattr(resnets, 'Cond' + config.encoder)
    )


def mse_loss(y_true, y_pred):
    return jnp.sum(jnp.mean((y_true - y_pred) ** 2, axis=0))


def sharded_random_t(rng, batch, max_scale=1.0):
    x = batch['image']
    total_batch_size = x.shape[0] * x.shape[1]
    n = total_batch_size - 1
    grid = jnp.arange(n, dtype=x.dtype) / n
    offset = jax.random.uniform(rng, dtype=grid.dtype)
    grid = (grid + offset) % 1.0
    grid = jnp.concatenate([
        jnp.zeros((1,), dtype=grid.dtype), grid
    ], axis=0)
    grid = (grid ** 2) * max_scale
    return local_sharding(jax.random.permutation(rng, grid))


class Manifold(struct.PyTreeNode):
    M: Union[EuclideanImage,ImmersedImage] = struct.field(pytree_node=False)


def brownian(manifold, rng, x, t):
    if type(manifold.M) is ImmersedImage:
        # [-1, 1] -> [0, 1] -> [-1, 1]
        xt = manifold.M.brownian(rng, x / 2 + 0.5, t, step_size=1.0)
        xt = (xt - 0.5) * 2
    elif type(manifold.M) is EuclideanImage:
        xt = manifold.M.brownian(rng, x, t)
    else:
        raise ValueError('Unknown manifold.')

    return xt


def batch_loss_match_grad(rng, params, apply_fn, model, batch, t, manifold):
    assert type(manifold.M) in (EuclideanImage, ImmersedImage)

    def model_fn(_x_):
        return model.apply_fn({'params': model.params}, _x_)[..., 0]

    model_grad_fn = jax.vmap(jax.value_and_grad(model_fn))

    def pde_fn(_x_, _t_):
        return apply_fn({'params': params}, _x_, _t_)[..., 0]

    pde_grad_fn = jax.vmap(jax.value_and_grad(pde_fn, argnums=0))

    x = batch['image']

    rng, subkey = jax.random.split(rng)
    xt = brownian(manifold, subkey, x, t)

    yt, gt = model_grad_fn(xt)
    y_hat, g_hat = pde_grad_fn(
        jnp.concatenate([x, xt], axis=0),
        jnp.concatenate([t, jnp.zeros_like(t)], axis=0)
    )
    yt_hat, y0_hat = jnp.split(y_hat, 2, axis=0)
    gt_hat, g0_hat = jnp.split(g_hat, 2, axis=0)

    boundary_mse = mse_loss(yt, y0_hat)
    solution_mse = mse_loss(yt, yt_hat)
    boundary_grad_mse = mse_loss(gt, g0_hat)
    solution_grad_mse = mse_loss(gt, gt_hat)

    return (
        boundary_mse, solution_mse,
        boundary_grad_mse, solution_grad_mse, xt
    )


def batch_loss(rng, params, apply_fn, model, batch, t, manifold):
    assert type(manifold.M) in (EuclideanImage, ImmersedImage)

    def model_fn(_x_):
        return model.apply_fn({'params': model.params}, _x_)[..., 0]

    def pde_fn(_x_, _t_):
        return apply_fn({'params': params}, _x_, _t_)[..., 0]

    x = batch['image']

    rng, subkey = jax.random.split(rng)
    xt = brownian(manifold, subkey, x, t)

    yt = model_fn(xt)
    y_hat = pde_fn(
        jnp.concatenate([x, xt], axis=0),
        jnp.concatenate([t, jnp.zeros_like(t)], axis=0)
    )
    yt_hat, y0_hat = jnp.split(y_hat, 2, axis=0)
    boundary_mse = mse_loss(yt, y0_hat)
    solution_mse = mse_loss(yt, yt_hat)

    return boundary_mse, solution_mse


def train_step(rng, state, model, batch, t, manifold,
               learning_rate_fn, grad_weight=0.0, boundary_weight=0.1):
    step = state.step

    def loss_fn(params):
        if grad_weight > 0:
            bmse, smse, bgmse, sgmse, _ = batch_loss_match_grad(
                rng, params, state.apply_fn, model, batch, t, manifold)
            loss = (smse + grad_weight * sgmse) + boundary_weight * (bmse + grad_weight * bgmse)
            return loss, (bmse, smse, bgmse, sgmse)
        else:
            bmse, smse = batch_loss(
                rng, params, state.apply_fn, model, batch, t, manifold)
            loss = smse + boundary_weight * bmse
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
    if grad_weight > 0:
        bmse, smse, bgmse, sgmse = aux[1]
        metrics = {
            'loss': loss, 'boundary': bmse, 'solution': smse,
            'grad@boundary': bgmse, 'grad@solution': sgmse
        }
    else:
        bmse, smse = aux[1]
        metrics = {'loss': loss, 'boundary': bmse, 'solution': smse}
    metrics = lax.pmean(metrics, axis_name='batch')
    metrics['learning_rate'] = learning_rate_fn(step)

    new_state = state.apply_gradients(grads=grads)
    if dynamic_scale:
        new_state = dynamic_scale_update(state, new_state, is_finite)
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics


def eval_step(rng, state, model, batch, t, manifold):
    bmse, smse, bgmse, sgmse, xt = batch_loss_match_grad(
        rng, state.params, state.apply_fn, model, batch, t, manifold)
    metrics = {
        'boundary': bmse, 'solution': smse,
        'grad@boundary': bgmse, 'grad@solution': sgmse
    }
    return lax.pmean(metrics, axis_name='batch'), xt


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
        config: mlc.ConfigDict, workdir: str, manifold: bool,
        model_config: mlc.ConfigDict, model_workdir: str,
        save_rw_images: bool=True) -> TrainState:
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

    input_dtype = np.float16 if config.half_precision else np.float32

    dataset_kwargs = dict(
        width=config.image_size, height=config.image_size,
        dtype=input_dtype, unit_interval=True
    )
    dataset = SyntheticDataset(**dataset_kwargs)

    trainloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=True, num_workers=8, drop_last=True,
        worker_init_fn=make_worker_init_fn(41),
        persistent_workers=True
    )
    testloader = DataLoader(
        dataset,
        batch_size=local_batch_size * 2,
        shuffle=True, num_workers=8, drop_last=True,
        worker_init_fn=make_worker_init_fn(1021),
        persistent_workers=True
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

    solution = create_pde_solution_model(config)
    learning_rate_fn = create_learning_rate_fn(config, steps_per_epoch)
    state = create_pde_solution_train_state(rng, config, solution, learning_rate_fn)
    state = restore_checkpoint(state, ckptdir)
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    rng = jax.random.split(rng, jax.local_device_count())

    M = EuclideanImage() if not manifold \
        else ImmersedImage(dataset.decode_jax, dataset.encode_jax)
    manifold = Manifold(M)

    p_train_step = jax.pmap(partial(
        train_step,
        manifold=manifold,
        learning_rate_fn=learning_rate_fn,
        boundary_weight=config.boundary_weight,
        grad_weight=config.grad_weight
    ), axis_name='batch')

    p_eval_step = jax.pmap(partial(
        eval_step,
        manifold=manifold
    ), axis_name='batch')

    def eval(_state_, _rng_, _step_):
        epoch = _step_ // steps_per_epoch
        eval_metrics = []

        for i in range(steps_per_eval):
            eval_batch = next(testloader)
            t = sharded_random_t(_rng_[0], eval_batch)
            metrics, xt = p_eval_step(_rng_, _state_, model, eval_batch, t)
            eval_metrics.append(metrics)

            if i == 0 and save_rw_images:
                x = eval_batch['image']
                images = jnp.reshape(
                    jnp.stack([x, xt], axis=-4),
                    (-1,) + x.shape[-3:]
                ) / 2 + 0.5  # [-1, 1] -> [0, 1]
                save_image(images, imgdir / f'rw_{_step_}.png', nrow=16)

        eval_metrics = dereplicate_metrics(eval_metrics)
        summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
        logging.info(
            'eval epoch: %d, boundary mse: %.6f, solution mse: %.6f',
            epoch, summary['boundary'], summary['solution']
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
