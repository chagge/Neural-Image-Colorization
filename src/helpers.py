import numpy as np
import os
import tensorflow as tf
import time
from scipy.misc import toimage
import urllib.request as request
import zipfile

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
OUT_PATH = os.path.abspath(DIR_PATH + '/../output/out_%.0f.jpg' % time.time())
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
LIB_DIR = os.path.abspath(CURRENT_PATH + '/../lib/')
TRAINING_DIR = os.path.abspath(CURRENT_PATH + '/../lib/train2014/')
TRAINING_URL = 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip'


class Helpers:
    def __init__(self):
        pass

    @staticmethod
    def add_gradient_summary(grad, var):
        if grad is not None:
            #tf.summary.histogram(var.op.name + "/gradient", grad)
            pass

    # Checks for training data to see if it's missing or not. Asks to download if missing.
    @staticmethod
    def check_for_examples():
        # Ask to unzip training data if a previous attempt was made
        zip_path = os.path.abspath(CURRENT_PATH + '/../lib/train2014.zip')
        if os.path.isfile(zip_path):
            Helpers.__ask_to_unzip(zip_path)

        # Ask to download training data if the training dir does not exist or does not contain the needed files
        if not os.path.isdir(TRAINING_DIR):
            Helpers.ask_to_download()
        else:
            training_files = os.listdir(TRAINING_DIR)
            num_training_files = len(training_files)
            if num_training_files <= 1:
                Helpers.ask_to_download()

    @staticmethod
    def exit_program(rc=0, message="Exiting the program.."):
        print(message)
        tf.get_default_session().close()
        exit(rc)

    @staticmethod
    def get_lib_dir():
        if not os.path.isdir(LIB_DIR):
            os.makedirs(LIB_DIR)
        return LIB_DIR

    @staticmethod
    def get_training_dir():
        return TRAINING_DIR

    # Renders the generated image given a tensorflow session and a variable image (x)
    @staticmethod
    def render_img(img, display=False, path_out=None):
        if not path_out:
            path_out = os.path.abspath(DIR_PATH + '/../output/out_%.0f.jpg' % time.time())

        clipped_img = np.clip(img, 0., 1.)
        shaped_img = np.reshape(clipped_img, img.shape[1:])

        if display:
            toimage(shaped_img).show()

        if path_out:
            toimage(shaped_img).save(path_out)

    # Asks on stdout to download MSCOCO data. Downloads if response is 'y'
    @staticmethod
    def ask_to_download():
        print("You've requested to train a new model. However, you've yet to download the training data.")

        answer = 0
        while answer is not 'y' and answer is not 'N':
            answer = input("Would you like to download the 13 GB file? [y/N] ").replace(" ", "")

        # Download weights if yes, else exit the program
        if answer == 'y':
            print("Downloading from %s. Please be patient..." % TRAINING_URL)

            if not os.path.isdir(LIB_DIR):
                os.makedirs(LIB_DIR)

            zip_save_path = LIB_DIR + 'train2014.zip'
            request.urlretrieve(TRAINING_URL, zip_save_path)
            Helpers.ask_to_unzip(zip_save_path)
        elif answer == 'N':
            Helpers.exit_program()

    # Asks on stdout to unzip a given zip file path. Unizips if response is 'y'
    @staticmethod
    def ask_to_unzip(path):
        answer = 0
        while answer is not 'y' and answer is not 'N':
            answer = input("The application requires the file to be unzipped. Unzip? [y/N] ").replace(" ", "")

        if answer == 'y':
            if not os.path.isdir(TRAINING_DIR):
                os.makedirs(TRAINING_DIR)

            print("Unzipping file..")
            zip_ref = zipfile.ZipFile(path, 'r')
            zip_ref.extractall(CURRENT_PATH + '/../lib/')
            zip_ref.close()
            os.remove(path)
        else:
            Helpers.exit_program()
