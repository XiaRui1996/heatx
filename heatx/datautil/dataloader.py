from pathlib import Path

from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, \
    Resize, ToPILImage, ColorJitter, \
    RandomResizedCrop, RandomHorizontalFlip, RandomPosterize, \
    RandomAdjustSharpness, RandomGrayscale, RandomAutocontrast

import jax
from jax import nn
import jax.numpy as jnp
import numpy as np
import cv2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def inverse_sigmoid_jax(x):
    return -jnp.log(jnp.maximum((1 / x) - 1, 1E-10))


class SyntheticDataset(Dataset):
    def __init__(self, folder=None, *,
                 width=128, height=128, latents=6, size=10000,
                 seed=1021, dtype=np.float32, yrange=4, unit_interval=True):
        self.width = width
        self.height = height
        self.latents = latents
        self.size = size
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)
        self.yrange = yrange
        self.unit_interval = unit_interval
        self.backgrounds = self.rng.uniform(size=(self.latents, width, height))

    def reset_rng(self, seed):
        self.rng = np.random.default_rng(seed)

    def decode(self, z):
        code = sigmoid(z)
        W, H = self.width, self.height
        pic = np.zeros((W, H), dtype=self.dtype)
        
        wgrid = W * np.linspace(0, 1, 2 * self.latents)
        wgrid = wgrid.astype(int)
        
        hgrid = H * np.linspace(0, 1, 2 * self.latents)
        hgrid = hgrid.astype(int)
        
        for i, v in enumerate(code):
            x0, y0 = wgrid[i], hgrid[i]
            x1, y1 = wgrid[-(i+1)], hgrid[-(i+1)]
            pic[x0:x1, y0:y1] = v * self.backgrounds[i][x0:x1, y0:y1]

        return pic.T[:, :, np.newaxis]

    def _decode_jax(self, z):
        code = nn.sigmoid(z)

        W, H = self.width, self.height
        pic = jnp.zeros((W, H), dtype=self.dtype)
        
        wgrid = W * np.linspace(0, 1, 2 * self.latents)
        wgrid = wgrid.astype(int)
        
        hgrid = H * np.linspace(0, 1, 2 * self.latents)
        hgrid = hgrid.astype(int)

        for i in range(self.latents):
            x0, y0 = wgrid[i], hgrid[i]
            x1, y1 = wgrid[-(i+1)], hgrid[-(i+1)]
            pic = pic.at[x0:x1, y0:y1].set(code[i] * self.backgrounds[i][x0:x1, y0:y1])

        return pic.T[:, :, jnp.newaxis]
    
    def decode_jax(self, z):
        prefix = z.shape[:-1]
        z = jnp.reshape(z, (-1, self.latents))
        x = jax.vmap(self._decode_jax)(z)
        x = jnp.reshape(x, prefix + x.shape[1:])
        return x

    def encode_jax(self, x):
        x = jnp.squeeze(x, axis=-1)

        W, H = self.width, self.height
        wgrid = W * np.linspace(0, 1, 2 * self.latents)
        wgrid = wgrid.astype(int)

        hgrid = H * np.linspace(0, 1, 2 * self.latents)
        hgrid = hgrid.astype(int)

        code = []

        for i in range(self.latents):
            x0, y0 = wgrid[i], hgrid[i]
            x1, y1 = wgrid[i+1], hgrid[i+1]
            v = jnp.divide(
                jnp.sum(x[..., x0:x1, y0:y1], axis=(-1, -2)),
                np.sum(self.backgrounds[i][x0:x1, y0:y1])
            )
            code.append(v)
        
        code = jnp.stack(code, axis=-1)
        z = inverse_sigmoid_jax(code)
        return z

    def label(self, z):
        lower, upper = -self.yrange, self.yrange
        for x in z:
            mid = (lower + upper) / 2
            if x <= 0.0:
                lower = mid
            else:
                upper = mid
        return self.rng.normal(
            loc=(lower + upper) / 2,
            scale=np.abs(upper - lower) / 6 
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        z = self.rng.normal(scale=2.0, size=self.latents)
        x = self.decode(z)
        if not self.unit_interval:
            x *= 255
        y = self.label(z)
        return x, y


class ImageFolderDataset(Dataset):
    def __init__(self, folder, dtype=np.float32, glob='*.jpg',
                 transform=None, jitter=False, unit_interval=True, seed=42):
        self.folder = Path(folder)
        self.files = list(self.folder.glob(glob))
        self.dtype = dtype
        self.transform = transform
        self.jitter = jitter
        self.unit_interval = unit_interval
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        im = cv2.imread(str(path))

        if self.transform is not None:
            im = self.transform(im)
        
        im = np.asarray(im, dtype=self.dtype)

        if self.jitter:
            noise = self.rng.uniform(low=-0.5, high=0.5, size=im.shape)
            im += noise

        if self.unit_interval:
            im = im / 255.0

        return im.astype(self.dtype)


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )



class FaceDataset(ImageFolderDataset):
    def __init__(self, folder, height=128, width=128,
                 aug=False, transform=None, **kwargs):
        transforms = [ToPILImage()]
        if transform is not None:
            transforms.append(transform)

        if not aug:
            transforms += [Resize((height, width))]
        elif str(aug).lower() == 'celeba':
            cx, cy, half = 89, 121, 70
            x1, x2 = cy - half, cy + half
            y1, y2 = cx - half, cx + half
            transforms += [
                Crop(x1, x2, y1, y2),
                RandomHorizontalFlip(p=0.5),
                RandomResizedCrop((height, width), scale=(0.9, 1), ratio=(0.9, 1.1))
            ]
        elif str(aug).lower() == 'weak':
            transforms += [
                ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0),
                RandomGrayscale(p=0.05),
                RandomHorizontalFlip(p=0.5),
                RandomResizedCrop((height, width), scale=(0.9, 1))
            ]
        elif str(aug).lower() == 'strong':
            transforms += [
                ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0),
                RandomAdjustSharpness(2, p=0.5),
                RandomAutocontrast(p=0.5),
                RandomPosterize(5, p=0.5),
                RandomGrayscale(p=0.25),
                RandomHorizontalFlip(p=0.5),
                RandomResizedCrop((height, width), scale=(0.85, 1))
            ]
        else:
            raise ValueError(f'Unknown augmentation: {aug}')
        super().__init__(folder, transform=Compose(transforms), **kwargs)


    def __getitem__(self, idx):
        name = self.files[idx].name
        label = 0.0 if '_' not in name else float(name.split('_')[0])
        return super().__getitem__(idx), label
