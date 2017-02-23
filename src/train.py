import argparse
import discriminator
import generator
from helpers import Helpers
import tensorflow as tf
from trainer import Trainer

# Model hyperparamaters
opts = {
    'batch_size': 64,
    'iterations': 1200000,
    'learning_rate': 2e-6,
    'model_path': None,  # path of previously trained model to continue training from
    'print_every': 100,
    'save_every': 10000,
    'training_height': 256,
    'training_width': 256,
}


def parse_args():
    """
    Creates command line arguments with the same name and default values as those in the global opts variable
    Then updates opts using their respective argument values
    """

    # Parse command line arguments to assign to the global opt variable
    parser = argparse.ArgumentParser(description='colorize images using conditional generative adversarial networks')
    for opt_name, value in opts.items():
        parser.add_argument("--%s" % opt_name, default=value)

    # Update global opts variable using flag values
    args = parser.parse_args()
    for opt_name, _ in opts.items():
        opts[opt_name] = getattr(args, opt_name)

parse_args()
with tf.Session() as sess:
    # Initialize networks
    gen = generator.Generator()
    disc = discriminator.Discriminator()

    # Train them
    t = Trainer(sess, gen, disc, opts)
    t.train()
