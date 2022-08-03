"""Default Hyperparameter configuration."""
from pathlib import Path
import math

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    datadir = Path(__file__).parent / '../../../../..' / 'datasets'

    config.train_datadirs = [
        datadir / 'UTKFace' / 'train'
    ]
    config.eval_datadirs = [
        datadir / 'UTKFace' / 'test'
    ]
    config.train_augs = ['weak']
    config.eval_augs = [False]

    config.image_size = 128
    config.image_channels = 3

    config.unet = 'UNet18'
    config.sde = 'VESDE'
    config.minSNR = 1 / (16.0 ** 2)

    config.learning_rate = 0.001
    config.weight_decay = 0.0001
    config.warmup_epochs = 5.0
    config.batch_size = 256

    config.num_epochs = 400.0
    config.log_every_steps = 200

    config.half_precision = False

    config.num_train_steps = -1
    config.steps_per_eval = -1
    config.eval_every_epochs = 5

    config.rec_grid_size = 10

    return config
