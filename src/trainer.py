import numpy as np
import os
from scipy.misc import toimage
import tensorflow as tf
import time
import urllib.request as request
import zipfile

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
OUT_PATH = DIR_PATH + '/../output/out_%.0f.jpg' % time.time()


class Trainer:
    def __init__(self, session, gen, disc, train_dir, training_dims, print_training_status=True, print_every_n=100):
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.disc = disc
        self.gen = gen
        self.paths = {
            'out_dir': self.current_path + '/../output/',
            'trained_generators_dir': self.current_path + '/../lib/generators/',
            'training_dir': train_dir,
            'training_url': 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip'
        }
        self.session = session
        self.print_training_status = print_training_status
        self.print_n = print_every_n
        self.train_height = training_dims['height']
        self.train_width = training_dims['width']

    def train(self, epochs, learning_rate):
        self.__check_for_examples()

        with tf.Session() as sess:
            bw_shape = [1, self.train_height, self.train_width, 1]
            color_shape = bw_shape[:3] + [3]
            gen_placeholder = tf.placeholder(dtype=tf.float32, shape=bw_shape)
            disc_placeholder = tf.placeholder(dtype=tf.float32, shape=color_shape)

            # Build the generator
            self.gen.build(gen_placeholder)

            # Generate a sample and attain the probability that the sample and the target are from the real distribution
            sample = self.gen.output
            prob_sample = self.disc.predict(sample)
            prob_target = self.disc.predict(disc_placeholder)

            # Generator training ops
            gen_loss = -tf.reduce_mean(tf.log(prob_sample))
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
            gen_update = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=gen_vars)

            # Discriminator training ops
            disc_loss = -tf.reduce_mean(tf.log(prob_target) + tf.log(1. - prob_sample))
            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
            disc_update = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss, var_list=disc_vars)

            # Begin optimization process
            example = self.next_example()
            example_bw, example_color = self.__split_img(example, self.train_height, self.train_width)
            print("Initializing session and begin threading..")
            tf.initialize_all_variables().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            start_time = time.time()

            for i in range(epochs):
                bw = example_bw.eval()
                color = example_color.eval()
                _, d_loss = sess.run([disc_update, disc_loss], feed_dict={gen_placeholder: bw, disc_placeholder: color})
                _, g_loss = sess.run([gen_update, gen_loss], feed_dict={gen_placeholder: bw})

                # Print current epoch number and errors if warranted
                if self.print_training_status and i % self.print_n == 0:
                    total_loss = g_loss + d_loss
                    fields = (i, total_loss, g_loss, d_loss)
                    print("Epoch %06d | Total Loss %.05f | Generator Loss %.05f | Discriminator Loss %.05f" % fields)
                    self.__render_img(sample.eval(feed_dict={gen_placeholder: bw}), path_out=OUT_PATH)

            # Alert that training has been completed and print the run time
            elapsed = time.time() - start_time
            print("Training complete. The session took %.2f seconds to complete." % elapsed)
            coord.request_stop()
            coord.join(threads)

    # Returns an image in both its grayscale and rgb formats
    def next_example(self):
        # Ops for getting training images, from retrieving the filenames to reading the data
        regex = self.paths['training_dir'] + '/*.jpg'
        filenames = tf.train.match_filenames_once(regex)
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        _, file = reader.read(filename_queue)
        return file

    # Asks on stdout to download MSCOCO data. Downloads if response is 'y'
    def __ask_to_download(self):
        print("You've requested to train a new model. However, you've yet to download the training data.")

        answer = 0
        while answer is not 'y' and answer is not 'N':
            answer = input("Would you like to download the 13 GB file? [y/N] ").replace(" ", "")

        # Download weights if yes, else exit the program
        if answer == 'y':
            print("Downloading from %s. Please be patient..." % self.paths['training_url'])

            lib_dir = self.current_path + '/../lib/'
            if not os.path.isdir(lib_dir):
                os.makedirs(lib_dir)
            zip_save_path = lib_dir + 'train2014.zip'
            request.urlretrieve(self.paths['training_url'],  zip_save_path)
            self.__ask_to_unzip(zip_save_path)
        elif answer == 'N':
            self.__exit()

    # Asks on stdout to unzip a given zip file path. Unizips if response is 'y'
    def __ask_to_unzip(self, path):
        answer = 0
        while answer is not 'y' and answer is not 'N':
            answer = input("The application requires the file to be unzipped. Unzip? [y/N] ").replace(" ", "")

        if answer == 'y':
            if not os.path.isdir(self.paths['training_dir']):
                os.makedirs(self.paths['training_dir'])

            print("Unzipping file..")
            zip_ref = zipfile.ZipFile(path, 'r')
            zip_ref.extractall(self.current_path + '/../lib/')
            zip_ref.close()
            os.remove(path)
        else:
            self.__exit()

    # Checks for training data to see if it's missing or not. Asks to download if missing.
    def __check_for_examples(self):
        # Ask to unzip training data if a previous attempt was made
        zip_path = os.path.abspath(self.current_path + '/../lib/train2014.zip')
        if os.path.isfile(zip_path):
            self.__ask_to_unzip(zip_path)

        # Ask to download training data if the training dir does not exist or does not contain the needed files
        if not os.path.isdir(self.paths['training_dir']):
            self.__ask_to_download()
        else:
            training_files = os.listdir(self.paths['training_dir'])
            num_training_files = len(training_files)
            if num_training_files <= 1:
                self.__ask_to_download()

    # Returns whether or not there is a checkpoint available
    def __is_trained(self):
        return os.path.isfile('model.meta')

    # Renders the generated image given a tensorflow session and a variable image (x)
    def __render_img(self, img, display=False, path_out=None):
        clipped_img = np.clip(img, 0., 1.)
        shaped_img = np.reshape(clipped_img, img.shape[1:])

        if display:
            toimage(shaped_img).show()

        if path_out:
            toimage(shaped_img).save(path_out)

    # Returns a pair of decoded image with a 1 and 3 channels respectively
    def __split_img(self, file, height, width):
        with tf.control_dependencies([file]):
            def get_image(x, channels):
                img = tf.image.decode_jpeg(x, channels=channels)
                img = tf.image.resize_images(img, [height, width])
                img = tf.div(img, 255.)
                img = tf.expand_dims(img, dim=0)
                return img

            # Get a single channel and a three-channel copy of the image and return them
            example_bw = get_image(file, 1)
            example_color = get_image(file, 3)
            return example_bw, example_color

    def __exit(self, rc=0, message="Exiting the program.."):
        print(message)
        self.session.close()
        exit(rc)
