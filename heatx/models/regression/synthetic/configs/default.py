"""Default Hyperparameter configuration."""
from pathlib import Path
import math

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.image_size = 128
    config.image_channels = 1

    config.encoder = 'ResNet18'

    config.learning_rate = 0.001
    config.weight_decay = 0.0001
    config.warmup_epochs = 5.0
    config.batch_size = 128

    config.num_epochs = 200.0
    config.log_every_steps = 200

    config.half_precision = False

    config.num_train_steps = -1
    config.steps_per_eval = -1
    config.eval_every_epochs = 10

    return config
