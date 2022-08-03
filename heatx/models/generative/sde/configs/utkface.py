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
    config.sde = 'VPSDE'
    config.minSNR = math.exp(-10)


    config.learning_rate = 0.001
    config.weight_decay = 0.0001
    config.warmup_epochs = 10.0
    config.batch_size = 64 * 4

    config.num_epochs = 800.0
    config.log_every_steps = 100

    config.half_precision = False

    config.num_train_steps = -1
    config.steps_per_eval = -1
    config.eval_every_epochs = 20

    config.rec_grid_size = 10

    return config
