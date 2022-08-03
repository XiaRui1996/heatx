import numpy as np
import jax.numpy as jnp


def to_heatmap_numpy(img, percentile=100):
    if img.ndim >= 3 and img.shape[-1] in (1, 3):
        img = np.sum(img, axis=-1)
    
    span = np.percentile(np.abs(img), percentile)
    blue, red, green = [np.ones_like(img) for _ in range(3)]

    pos = np.minimum(1, 1 - img[img > 0] / span)
    neg = np.minimum(1, 1 + img[img < 0] / span)

    blue[img > 0] = pos
    red[img < 0] = neg
    green[img > 0] = pos
    green[img < 0] = neg

    return np.stack([blue, green, red], axis=-1)


def to_heatmap(img, percentile=100):
    if img.ndim >= 3 and img.shape[-1] in (1, 3):
        img = jnp.sum(img, axis=-1)
    
    span = jnp.percentile(jnp.abs(img), percentile)
    blue, red, green = [jnp.ones_like(img) for _ in range(3)]

    pos = jnp.minimum(1, 1 - img / span)
    neg = jnp.minimum(1, 1 + img / span)

    blue = jnp.where(img > 0, pos, blue)
    red = jnp.where(img < 0, neg, red)
    green = jnp.where(img > 0, pos, neg)

    return jnp.stack([blue, green, red], axis=-1)
