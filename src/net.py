import tensorflow as tf

DROPOUT_RATE = .5
EPSILON = 1e-8
FILTER_SHAPE = [4, 4]
SAMPLE_LEVEL = 2
STRIDE = [1, SAMPLE_LEVEL, SAMPLE_LEVEL, 1]


class Net(object):
    def __init__(self):
        self.dropout_keep = DROPOUT_RATE
        self.epsilon = EPSILON
        self.filter_shape = FILTER_SHAPE
        self.sample_level = SAMPLE_LEVEL
        self.stride = STRIDE

    # Constructs and returns a network layer tailored to one's specifications
    def conv_layer(self, inputs, out_size, name,
                   activation=tf.nn.relu,
                   dropout=False,
                   noisy=False,
                   normalize=True,
                   pad='SAME',
                   shape=None,
                   stride=STRIDE,
                   is_training=False):

        """
        Performs a convolution and auxiliary operation for training related stability on a given input
        :param inputs: input tensor to perform convolution on
        :param out_size:
        :param name:
        :param activation:
        :param dropout:
        :param noisy:
        :param normalize:
        :param pad:
        :param shape:
        :param stride:
        :param is_training:
        :return:
        """

        with tf.variable_scope(name):
            if not shape:
                in_size = inputs.get_shape().as_list()[3]
                shape = self.filter_shape + [in_size] + [out_size]

            # Create filters and perform convolution
            filters = tf.get_variable(
                'filters',
                initializer=tf.truncated_normal(shape, mean=0., stddev=.02))
            x = tf.nn.conv2d(inputs, filters, padding=pad, strides=stride)

            # Add bias
            num_out_maps = shape[3]
            bias = tf.get_variable(
                'bias',
                initializer=tf.constant(.1, shape=[num_out_maps]))
            x = tf.nn.bias_add(x, bias)

            # Training related ops
            x = self.batch_normalize(x, is_training) if normalize else x
            x = tf.nn.dropout(x, keep_prob=self.dropout_keep) if dropout else x
            x = activation(x) if activation is not None else x
            x = self.add_noise(x) if noisy else x

            return x

    @staticmethod
    def add_noise(inputs):
        """
        Adds gaussian noise
        :param inputs:
        :return:
        """

        noise = tf.random_normal(
            tf.shape(inputs),
            mean=0.,
            stddev=.02)
        inputs = tf.add(inputs, noise)
        return inputs

    # Batch normalize inputs to reduce covariate shift and improve the efficiency of training
    @staticmethod
    def batch_normalize(inputs, is_training):
        """

        :param inputs:
        :param is_training:
        :return:
        """

        batch_size = inputs.get_shape()[0]
        if batch_size is 1 or True:
            return Net.instance_normalize(inputs)

        with tf.variable_scope("batch_normalization"):
            shape = inputs.get_shape().as_list()
            num_maps = shape[3]

            # Trainable variables for scaling and offsetting our inputs
            scale = tf.Variable(tf.ones([num_maps], dtype=tf.float32))
            offset = tf.Variable(tf.zeros([num_maps], dtype=tf.float32))

            # Mean and variances related to our current batch
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])

            # Create an optimizer to maintain a 'moving average'
            ema = tf.train.ExponentialMovingAverage(decay=.9)

            def ema_retrieve():
                return ema.average(batch_mean), ema.average(batch_var)

            # If the net is being trained, update the average every training step
            def ema_update():
                ema_apply = ema.apply([batch_mean, batch_var])

                # Make sure to compute the new means and variances prior to returning their values
                with tf.control_dependencies([ema_apply]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            # Retrieve the means and variances and apply the BN transformation
            mean, var = tf.cond(tf.equal(is_training, True), ema_update, ema_retrieve)
            bn_inputs = tf.nn.batch_normalization(inputs, mean, var, offset, scale, EPSILON)

        return bn_inputs

    # Instance normalize inputs to reduce covariate shift and reduce dependency on input contrast to improve results
    @staticmethod
    def instance_normalize(inputs):
        """

        :param inputs:
        :return:
        """

        with tf.variable_scope('instance_normalization'):
            batch, height, width, channels = [_.value for _ in inputs.get_shape()]
            mu, sigma_sq = tf.nn.moments(inputs, [1, 2], keep_dims=True)

            shift = tf.Variable(tf.constant(.1, shape=[channels]), name='shift')
            scale = tf.Variable(tf.ones([channels]), name='scale')
            normalized = (inputs - mu) / (sigma_sq + EPSILON) ** .5

            return tf.add(tf.mul(scale, normalized), shift)

    @staticmethod
    def leaky_relu(inputs):
        return tf.maximum(.2 * inputs, inputs)

