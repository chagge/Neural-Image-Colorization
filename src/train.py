import argparse
import discriminator
import generator
import os
import tensorflow as tf
import trainer

# Model hyperparamaters
EPOCHS = 1200000
LEARNING_RATE = .005

# Training related paramaters
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
TRAIN_DIR = os.path.abspath(DIR_PATH + '/../lib/train2014/')
TRAINING_DIMS = {'height': 256, 'width': 256}
PRINT_TRAINING_STATUS = True
PRINT_EVERY_N = 100


def parse_args():
    global TRAIN_DIR
    parser = argparse.ArgumentParser(description='colorize images using conditional generative adversarial networks.')
    parser.add_argument('--trainingdata', help='directory containing training images', default=TRAIN_DIR)
    args = parser.parse_args()
    TRAIN_DIR = os.path.abspath(args.trainingdata)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    parse_args()
    gen = generator.Generator()
    disc = discriminator.Discriminator()
    t = trainer.Trainer(sess, gen, disc, TRAIN_DIR, TRAINING_DIMS, PRINT_TRAINING_STATUS, PRINT_EVERY_N)
    t.train(EPOCHS, LEARNING_RATE)
