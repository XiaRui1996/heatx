import functools
import math


from flax import jax_utils
from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.training import checkpoints, train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
import cv2


def vsplit(key):
    return jax.vmap(jax.random.split)(key).swapaxes(0, 1)


def local_sharding(xs):
    local_device_count = jax.local_device_count()

    def reshape(x):
        x = np.asarray(x)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(reshape, xs)


def cycle(iterable):
    while True:
        for element in iterable:
            yield element


def create_cyclic_sharded_iter(it):
    it = map(local_sharding, cycle(it))
    return jax_utils.prefetch_to_device(it, 2)


class TrainState(train_state.TrainState):
    dynamic_scale: dynamic_scale_lib.DynamicScale = None


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


def warmup_cos_decay_lr_schedule_fn(
        base_learning_rate: float,
        num_epochs: int,
        warmup_epochs: int,
        steps_per_epoch: int):
    warmup_fn = optax.linear_schedule(
        init_value=0., end_value=base_learning_rate,
        transition_steps=warmup_epochs * steps_per_epoch)
    cosine_epochs = max(num_epochs - warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_epochs * steps_per_epoch])
    return schedule_fn


def dynamic_scale_update(state, new_state, is_finite):
    # if is_finite == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
    return new_state.replace(
        opt_state=jax.tree_map(
            functools.partial(jnp.where, is_finite),
            new_state.opt_state,
            state.opt_state),
        params=jax.tree_map(
            functools.partial(jnp.where, is_finite),
            new_state.params,
            state.params)
    )


def stack_forest(forest):
    stack_args = lambda *args: np.stack(args)
    return jax.tree_map(stack_args, *forest)


def dereplicate_metrics(device_metrics):
    # We select the first element of x in order to get a single copy of a
    # device-replicated metric.
    device_metrics = jax.tree_map(lambda x: x[0], device_metrics)
    metrics_np = jax.device_get(device_metrics)
    return stack_forest(metrics_np)


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0):
    """
    Make a grid of images and Save it into an image file.

    Args:
      ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
        fp - A filename(string) or file object
      nrow (int, optional): Number of images displayed in each row of the grid.
        The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
      padding (int, optional): amount of padding. Default: ``2``.
      scale_each (bool, optional): If ``True``, scale each image in the batch of
        images separately rather than the (min, max) over all images. Default: ``False``.
      pad_value (float, optional): Value for the padded pixels. Default: ``0``.
  """
    if not (isinstance(ndarray, jnp.ndarray) or
            (isinstance(ndarray, list) and all(isinstance(t, jnp.ndarray) for t in ndarray))):
        raise TypeError(
            'array_like of tensors expected, got {}'.format(type(ndarray)))

    if type(fp) is not str:
        fp = str(fp)

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] +
                        padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = jnp.full((height * ymaps + padding, width * xmaps +
                     padding, num_channels), pad_value).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid = grid.at[y * height + padding:(y + 1) * height,
                           x * width + padding:(x + 1) * width].set(ndarray[k])
            k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
    cv2.imwrite(fp, np.asarray(ndarr))
