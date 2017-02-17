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
NOISE_DECAY = 1e-8
DISC_GRAD_CLIP = .01
NUM_CRITIC = 5


class Trainer:
    def __init__(self, session, gen, disc, batch_size, training_dims, print_training_status=True, print_every_n=100):
        self.batch_size = batch_size
        self.disc = disc
        self.gen = gen
        self.session = session
        self.print_training_status = print_training_status
        self.print_n = print_every_n
        self.train_height = training_dims['height']
        self.train_width = training_dims['width']

    def train(self, epochs, learning_rate):
        # Set initial training shapes and placeholders
        x_shape = [None, self.train_height, self.train_width, 1]
        y_shape = x_shape[:3] + [2]
        z_shape = [1] + x_shape[1:]

        x_ph = tf.placeholder(dtype=tf.float32, shape=x_shape, name='input_placeholder')
        y_ph = tf.placeholder(dtype=tf.float32, shape=y_shape, name='condition_placeholder')
        z_ph = tf.placeholder(dtype=tf.float32, shape=x_shape, name='noise_placeholder')

        # Build the generator to setup layers and variables
        self.gen.build(x_ph, z_ph)

        # Generate a sample and attain the probability that the sample and the target are from the real distribution
        sample = self.gen.output

        gen_noise = tf.random_normal(z_shape, stddev=.02)
        gen_noise_batch = gen_noise
        for i in range(self.batch_size - 1):
            gen_noise_batch = tf.concat(0, [gen_noise_batch, gen_noise])

        prob_sample = self.disc.predict(sample, x_ph)
        prob_target = self.disc.predict(y_ph, x_ph)

        # Optimization ops for the discriminator
        disc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(prob_target, tf.ones_like(prob_target)))
        disc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(prob_sample, tf.zeros_like(prob_sample)))
        disc_loss = disc_loss_real + disc_loss_fake
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        disc_opt = tf.train.RMSPropOptimizer(learning_rate)
        disc_grads_ = disc_opt.compute_gradients(disc_loss, disc_vars)
        disc_grads = [(tf.clip_by_value(grad, -DISC_GRAD_CLIP, DISC_GRAD_CLIP), var) for grad, var in disc_grads_]
        disc_update = disc_opt.apply_gradients(disc_grads)

        # Optimization ops for the generator
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(prob_sample, tf.ones_like(prob_sample)))
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        gen_opt = tf.train.RMSPropOptimizer(learning_rate)
        gen_grads_ = gen_opt.compute_gradients(gen_loss, gen_vars)
        gen_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gen_grads_]
        gen_update = gen_opt.apply_gradients(gen_grads)

        # Training data retriever ops
        example = self.next_example(height=self.train_height, width=self.train_width)
        example_condition = tf.slice(example, [0, 0, 2], [self.train_height, self.train_width, 1])
        example_condition = tf.div(example_condition, 255.)
        example_label = tf.slice(example, [0, 0, 0], [self.train_height, self.train_width, 2])

        min_queue_examples = 100
        batch_condition, batch_label = tf.train.batch([example_condition, example_label], self.batch_size,
                                                      num_threads=4, capacity=min_queue_examples + 2 * self.batch_size)

        # delete this when done (retrieves image to render while training)
        CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
        filenames_ = tf.train.match_filenames_once(CURRENT_PATH + '/../nyc.jpg')
        filename_queue_ = tf.train.string_input_producer(filenames_)
        r = tf.WholeFileReader()
        fn_, f_ = r.read(filename_queue_)
        rgb_ = tf.image.decode_jpeg(f_, channels=3)
        rgb_ = tf.image.resize_images(rgb_, [self.train_height, self.train_width])
        img_ = tf.image.rgb_to_hsv(rgb_)
        img_ = tf.expand_dims(img_, dim=0)
        v_ = tf.slice(img_, [0, 0, 0, 2], [1, self.train_height, self.train_width, 1])
        v_ = tf.div(v_, 255.)
        colored_sample = tf.image.hsv_to_rgb(tf.concat(3, [sample, tf.mul(v_, 255.)]))
        colored_sample = tf.div(colored_sample, 255.)

        # Start session and begin threading
        print("Initializing session and begin threading..")
        self.session.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()

        for i in range(epochs):

            # Update steps
            for _ in range(NUM_CRITIC):
                conditional_batch = batch_condition.eval()
                label_batch = batch_label.eval()
                _gen_noise_batch = gen_noise_batch.eval()
                __, d_loss = self.session.run([disc_update, disc_loss],
                                              feed_dict={x_ph: conditional_batch, y_ph: label_batch, z_ph: _gen_noise_batch})

            conditional_batch = batch_condition.eval()
            _gen_noise_batch = gen_noise_batch.eval()
            _, g_loss = self.session.run([gen_update, gen_loss],
                                         feed_dict={x_ph: conditional_batch, z_ph: _gen_noise_batch})

            # Print current epoch number and errors if warranted
            if self.print_training_status and i % self.print_n == 0:
                total_loss = g_loss + d_loss
                log1 = "Epoch %06d || Total Loss %.010f || " % (i, total_loss)
                log2 = "Generator Loss %.010f || " % (g_loss)
                log3 = "Discriminator Loss %.010f" % (d_loss)
                print(log1 + log2 + log3)

                # test out delete when done
                _gen_noise_i = gen_noise.eval()
                rgb = self.session.run(
                    colored_sample,
                    feed_dict={x_ph: v_.eval(),
                               z_ph: _gen_noise_i})
                Helpers.render_img(rgb)

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
        img = tf.image.rgb_to_hsv(img)
        #img = tf.expand_dims(img, dim=0)
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

    def norm(self, inputs):
        _shifted = tf.sub(inputs, tf.reduce_min(inputs)),
        _range = tf.sub(tf.reduce_max(inputs), tf.reduce_min(inputs))
        normed = tf.div(_shifted, _range)

        return normed
