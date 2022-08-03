"""Default Hyperparameter configuration."""
from pathlib import Path

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    datadir = Path(__file__).parent / '../../../../..' / 'datasets' / 'UTKFace'

    config.datadir = str(datadir.resolve())
    config.image_size = 128
    config.image_channels = 3
    config.aug = 'strong'

    config.encoder = 'ResNet18'
    config.decoder = 'ResNetT18'
    config.latents = 64

    config.learning_rate = 0.001
    config.weight_decay = 0.0001
    config.warmup_epochs = 5.0
    config.batch_size = 64

    config.kl_anneal_end = 1.0
    config.kl_anneal_epochs = 25

    config.num_epochs = 200.0
    config.log_every_steps = 100

    config.half_precision = False

    config.num_train_steps = -1
    config.steps_per_eval = -1

    return config
