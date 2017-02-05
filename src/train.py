import argparse
import discriminator
import generator
import tensorflow as tf
from trainer import Trainer

# Model hyperparamaters
EPOCHS = 1200000
LEARNING_RATE = .1
BATCH_SIZE = 1

# Training related paramaters
TRAINING_DIMS = {'height': 256, 'width': 256}
PRINT_TRAINING_STATUS = True
PRINT_EVERY_N = 100


def parse_args():
    argparse.ArgumentParser(description='colorize images using conditional generative adversarial networks')

with tf.Session() as sess:
    parse_args()
    gen = generator.Generator()
    disc = discriminator.Discriminator()
    t = Trainer(sess, gen, disc, TRAINING_DIMS, PRINT_TRAINING_STATUS, PRINT_EVERY_N)
    t.train(EPOCHS, LEARNING_RATE, BATCH_SIZE)
