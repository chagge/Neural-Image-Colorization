from helpers import Helpers
import os
import tensorflow as tf
import time

EPSILON = 1e-10
LEARNING_DECAY = 40e-10
MOMENTUM_B1 = .5
MOMENTUM_GROWTH = .0008
MOMENTUM_LIMIT = .8


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

        bw_shape = [batch_size, self.train_height, self.train_width, 1]
        color_shape = bw_shape[:3] + [3]
        gen_placeholder = tf.placeholder(dtype=tf.float32, shape=bw_shape)
        disc_placeholder = tf.placeholder(dtype=tf.float32, shape=color_shape)

        # Build the generator
        self.gen.build(gen_placeholder)

        # Generate a sample and attain the probability that the sample and the target are from the real distribution
        sample = self.gen.output
        prob_sample, prob_sample_logit = self.disc.predict(sample)
        prob_sample_logit += EPSILON
        prob_target, prob_target_logit = self.disc.predict(disc_placeholder)
        prob_target_logit += EPSILON

        # Losses
        gen_loss = -tf.reduce_mean(tf.log(prob_sample))
        disc_loss = -tf.reduce_mean(tf.log(prob_target) + tf.log(1 - prob_sample))

        # Optimization
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

        learning_var = tf.Variable(tf.constant(learning_rate), trainable=False, name='learning_rate')
        momentum_var = tf.Variable(tf.constant(.5), trainable=False, name='momentum')
        #tf.variables_initializer([learning_var, momentum_var]).run()
        self.session.run(tf.initialize_all_variables())
        disc_opt = tf.train.AdamOptimizer(learning_var, beta1=momentum_var)
        gen_opt = tf.train.AdamOptimizer(learning_var, beta1=momentum_var)

        gen_grads = gen_opt.compute_gradients(gen_loss, gen_vars)
        gen_update = gen_opt.apply_gradients(gen_grads)

        disc_grads = disc_opt.compute_gradients(disc_loss, disc_vars)
        disc_update = disc_opt.apply_gradients(disc_grads)

        # Training data retriever ops
        example = self.next_example(height=self.train_height, width=self.train_width)
        min_after_dequeue = 1000
        num_threads = 2
        capacity = 30000
        batch = tf.train.shuffle_batch(
            [example],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)

        print("Initializing session and begin threading..")
        self.session.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()

        for i in range(epochs):
            batch_bw = tf.image.rgb_to_grayscale(batch)
            batch_color = tf.image.rgb_to_hsv(batch)

            bw = batch_bw.eval()
            color = batch_color.eval()
            #s = sample.eval(feed_dict={gen_placeholder: bw})

            _, d_loss, d_pred = self.session.run(
                [disc_update, disc_loss, prob_target],
                feed_dict={gen_placeholder: bw, disc_placeholder: color})

            _, g_loss, g_pred, g = self.session.run(
                [gen_update, gen_loss, prob_sample, gen_grads],
                feed_dict={gen_placeholder: bw})
            #print(1)
            # Print current epoch number and errors if warranted
            if self.print_training_status and i % self.print_n == 0:
                total_loss = g_loss + d_loss
                log1 = "Epoch %06d | Total Loss %.010f | " % (i, total_loss)
                log2 = "Generator Loss %.010f | " % g_loss
                log3 = "Discriminator Loss %.010f" % d_loss
                print(log1 + log2 + log3)

                learning_var.assign(learning_var - LEARNING_DECAY)
                momentum_var.assign(
                    tf.cond(
                        tf.less_equal(momentum_var, MOMENTUM_LIMIT),
                        lambda: momentum_var + MOMENTUM_GROWTH,
                        lambda: tf.identity(momentum_var)
                    ))

                single_sample = tf.slice(sample, [0, 0, 0, 0], [1, 256, 256, 3])
                rgb = self.session.run(tf.image.hsv_to_rgb(single_sample), feed_dict={gen_placeholder: bw})
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

