from helpers import Helpers
import os
import tensorflow as tf
import time

import skimage
import skimage.io
import skimage.transform
import numpy as np

EPSILON = 1e-10
MOMENTUM_B1 = .5


class Trainer:
    def __init__(self, session, gen, disc, training_dims, print_training_status=True, print_every_n=100):
        self.disc = disc
        self.gen = gen
        self.session = session
        self.print_training_status = print_training_status
        self.print_n = print_every_n
        self.train_height = training_dims['height']
        self.train_width = training_dims['width']

    def train(self, epochs, learning_rate, batch_size):
        Helpers.check_for_examples()

        bw_shape = [None, self.train_height, self.train_width, 1]
        color_shape = bw_shape[:3] + [3]
        gen_placeholder = tf.placeholder(dtype=tf.float32, shape=bw_shape)
        disc_placeholder = tf.placeholder(dtype=tf.float32, shape=color_shape)

        # Build the generator
        self.gen.build(gen_placeholder)

        # Generate a sample and attain the probability that the sample and the target are from the real distribution
        sample = self.gen.output
        prob_sample = self.disc.predict(sample) + EPSILON
        prob_target = self.disc.predict(disc_placeholder) + EPSILON

        # Losses
        gen_loss = -tf.reduce_mean(tf.log(prob_sample))
        disc_loss = -tf.reduce_mean(tf.log(prob_target) + tf.log(1 - prob_sample))

        # Optimization
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        disc_opt = tf.train.AdamOptimizer(learning_rate, beta1=MOMENTUM_B1)
        gen_opt = tf.train.AdamOptimizer(learning_rate, beta1=MOMENTUM_B1)

        gen_grads = gen_opt.compute_gradients(gen_loss, gen_vars)
        gen_update = gen_opt.apply_gradients(gen_grads)

        disc_grads = disc_opt.compute_gradients(disc_loss, disc_vars)
        disc_update = disc_opt.apply_gradients(disc_grads)

        # Training data retriever ops
        example = self.next_example(height=self.train_height, width=self.train_width)
        batch = tf.train.batch([example], batch_size=batch_size)

        # delete this when done
        CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
        filenames_ = tf.train.match_filenames_once(CURRENT_PATH + '/../nyc.jpg')
        filename_queue_ = tf.train.string_input_producer(filenames_)
        r = tf.WholeFileReader()
        fn_, f_ = r.read(filename_queue_)
        rgb_ = tf.image.decode_jpeg(f_, channels=1)
        img_ = tf.image.resize_images(rgb_, [self.train_height, self.train_width])
        img_ = tf.div(img_, 255.)
        #f_ = tf.image.rgb_to_hsv(img_)
        f = tf.expand_dims(img_, dim=0)

        print("Initializing session and begin threading..")
        self.session.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()

        for i in range(epochs):
            #example_bw_ = tf.image.rgb_to_grayscale(example)
            #example_bw = tf.expand_dims(example_bw_, dim=0)
            example_color = tf.image.rgb_to_hsv(batch)
            #example_color = tf.expand_dims(example_color_, dim=0)
            example_bw = tf.slice(
                example_color,
                [0, 0, 0, 2],
                [batch_size, self.train_height, self.train_width, 1])

            bw = example_bw.eval()
            color = example_color.eval()
            #s = sample.eval(feed_dict={gen_placeholder: bw})

            _, d_loss, d_pred = self.session.run(
                [disc_update, disc_loss, prob_target],
                feed_dict={gen_placeholder: bw, disc_placeholder: color})

            _, g_loss, g_pred, g = self.session.run(
                [gen_update, gen_loss, prob_sample, gen_grads],
                feed_dict={gen_placeholder: bw})
            print(1)
            # Print current epoch number and errors if warranted
            if self.print_training_status and i % self.print_n == 0:
                total_loss = g_loss + d_loss
                log1 = "Epoch %06d | Total Loss %.010f | " % (i, total_loss)
                log2 = "Generator Loss %.010f | " % g_loss
                log3 = "Discriminator Loss %.010f" % d_loss
                print(log1 + log2 + log3)

                # test out
                self.gen.is_training = False
                rgb = self.session.run(
                    tf.image.hsv_to_rgb(sample),
                    feed_dict={gen_placeholder: f.eval()})
                Helpers.render_img(rgb)
                self.gen.is_training = True

        # Alert that training has been completed and print the run time
        elapsed = time.time() - start_time
        print("Training complete. The session took %.2f seconds to complete." % elapsed)
        coord.request_stop()
        coord.join(threads)

        self.__save_model(gen_vars)

    @staticmethod
    # Returns an image in both its grayscale and rgb formats
    def next_example(height, width):
        # Ops for getting training images, from retrieving the filenames to reading the data
        regex = Helpers.get_training_dir() + '/*.jpg'
        filenames = tf.train.match_filenames_once(regex)
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        _, file = reader.read(filename_queue)

        img = tf.image.decode_jpeg(file, channels=3)
        img = tf.image.resize_images(img, [height, width])
        img = tf.div(img, 255.)
        return img

    # Returns whether or not there is a checkpoint available
    def __is_trained(self):
        lib_dir = Helpers.get_lib_dir
        return os.path.isfile(lib_dir + '/generator.meta')

    def __save_model(self, variables):
        print("Proceeding to save weights..")
        lib_dir = Helpers.get_lib_dir()
        if not os.path.isdir(lib_dir):
            os.makedirs(lib_dir)
        saver = tf.train.Saver(variables)
        saver.save(self.session, lib_dir + '/generator')

    # Returns a resized numpy array of an image specified by its path
    def load_img_to(self, path, height=None, width=None):
        # Load image
        img = skimage.io.imread(path) / 255.0
        if height is not None and width is not None:
            ny = height
            nx = width
        elif height is not None:
            ny = height
            nx = img.shape[1] * ny / img.shape[0]
        elif width is not None:
            nx = width
            ny = img.shape[0] * nx / img.shape[1]
        else:
            ny = img.shape[0]
            nx = img.shape[1]

        if len(img.shape) < 3:
            img = np.dstack((img, img, img))

        return skimage.transform.resize(img, (ny, nx)), [ny, nx, 3]
