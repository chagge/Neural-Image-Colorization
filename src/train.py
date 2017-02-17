import argparse
import discriminator
import generator
from helpers import Helpers
import tensorflow as tf
from trainer import Trainer

# Model hyperparamaters
BATCH_SIZE = 64
EPOCHS = 10000
LEARNING_RATE = 0.00005

# Training related paramaters
TRAINING_DIMS = {'height': 256, 'width': 256}
PRINT_TRAINING_STATUS = True
PRINT_EVERY_N = 100


def parse_args():
    argparse.ArgumentParser(description='colorize images using conditional generative adversarial networks')

with tf.Session() as sess:
    parse_args()

    # Initialize networks
    gen = generator.Generator()
    disc = discriminator.Discriminator()

    # Train them
    Helpers.check_for_examples()
    t = Trainer(sess, gen, disc, BATCH_SIZE, TRAINING_DIMS, PRINT_TRAINING_STATUS, PRINT_EVERY_N)
    t.train(EPOCHS, LEARNING_RATE)
