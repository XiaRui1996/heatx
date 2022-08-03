from absl import app
from absl import flags
from absl import logging
import jax
from ml_collections import config_flags

from pde import train_and_evaluate


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_bool('manifold', False, 'Solve on-manifold heat equation.')
flags.DEFINE_bool('rw', False, 'Save images from random walk.')
flags.DEFINE_string('model_workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config', None, 'File path to the training hyperparameter configuration.',
    lock_config=True)
config_flags.DEFINE_config_file(
    'model_config', None, 'File path to the hyperparameter configuration of target model.',
    lock_config=True)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    logging.info('JAX process: %d / %d',
                 jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    train_and_evaluate(
        FLAGS.config, FLAGS.workdir,
        manifold=FLAGS.manifold,
        save_rw_images=FLAGS.rw,
        model_config=FLAGS.model_config,
        model_workdir=FLAGS.model_workdir
    )


if __name__ == '__main__':
    flags.mark_flags_as_required(
        ['config', 'workdir', 'model_config', 'model_workdir'])
    app.run(main)
