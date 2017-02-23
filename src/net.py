import tensorflow as tf

DROPOUT_RATE = .5
EPSILON = 1e-12
FILTER_SHAPE = [4, 4]
SAMPLE_LEVEL = 2
STRIDE = [1, SAMPLE_LEVEL, SAMPLE_LEVEL, 1]


class Net(object):
    def __init__(self):
        self.dropout_keep = DROPOUT_RATE
        self.epsilon = EPSILON
        self.filter_shape = FILTER_SHAPE
        self.is_training = True
        self.noise_decay = 1e-8
        self.noise_multiplier = 1.
        self.sample_level = SAMPLE_LEVEL
        self.stride = STRIDE

    # Constructs and returns a network layer tailored to one's specifications
    def conv_layer(self, t, nmaps, name, act=tf.nn.relu, drop=False, norm=True, pad='SAME', shape=None, stride=STRIDE):

        """
        Performs a convolution and auxiliary operation for training related stability on a given input

        :param t: input tensor to perform convolution on
        :param nmaps: number of filter maps
        :param name: operation name
        :param act: nonlinear function
        :param drop: whether or not to use dropout regularization
        :param norm: whether or not to normalize input distribution
        :param pad: padding type
        :param shape: the 4D shape of the filters
        :param stride: convolution stride
        :return:
        """

        with tf.variable_scope(name):
            if not shape:
                in_size = t.get_shape().as_list()[3]
                shape = self.filter_shape + [in_size] + [nmaps]

            # Create filters and perform convolution
            filters = tf.get_variable('weights', initializer=tf.truncated_normal(shape, stddev=.02))
            tf.summary.histogram('weights', filters)
            maps_ = tf.nn.conv2d(t, filters, padding=pad, strides=stride)

            # Add bias
            nmaps = shape[3]
            bias = tf.get_variable('bias', initializer=tf.constant(.0, shape=[nmaps]))
            tf.summary.histogram('bias', bias)
            maps = tf.nn.bias_add(maps_, bias)

            # Training related ops
            maps = self.batch_normalize(maps, nmaps) if norm else maps
            maps = tf.nn.dropout(maps, keep_prob=self.dropout_keep) if drop else maps
            maps = act(maps) if act is not None else maps

            return maps

    # Batch normalize inputs to reduce covariate shift and improve the efficiency of training
    def _batch_normalize(self, inputs, num_maps, decay=.9):
        with tf.variable_scope("batch_normalization"):
            # Trainable variables for scaling and offsetting our inputs
            scale = tf.get_variable('scale', initializer=tf.ones([num_maps], dtype=tf.float32), trainable=True)
            tf.summary.histogram('scale', scale)
            offset = tf.get_variable('offset', initializer=tf.constant(.1, shape=[num_maps]), trainable=True)
            tf.summary.histogram('offset', offset)

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                with tf.name_scope(None):
                    # Mean and variances related to our current batch
                    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])

                    # Create an optimizer to maintain a 'moving average'
                    ema = tf.train.ExponentialMovingAverage(decay=decay)

                    def ema_retrieve():
                        return ema.average(batch_mean), ema.average(batch_var)

                    # If the net is being trained, update the average every training step
                    def ema_update():
                        ema_apply = ema.apply([batch_mean, batch_var])

                        # Make sure to compute the new means and variances prior to returning their values
                        with tf.control_dependencies([ema_apply]):
                            return tf.identity(batch_mean), tf.identity(batch_var)

                    # Retrieve the means and variances and apply the BN transformation
                    mean, var = tf.cond(tf.equal(self.is_training, True), ema_update, ema_retrieve)
                    bn_inputs = tf.nn.batch_normalization(inputs, mean, var, offset, scale, EPSILON)

        return bn_inputs

    # Instance normalize inputs to reduce covariate shift and reduce dependency on input contrast to improve results
    @staticmethod
    def batch_normalize(inputs, x):
        """
        Normalize the distribution of a given input tensor

        :param inputs: inputs to normalize
        :return: normalized inputs
        """

        with tf.variable_scope('instance_normalization'):
            batch, height, width, channels = [_.value for _ in inputs.get_shape()]
            mu, sigma_sq = tf.nn.moments(inputs, [1, 2], keep_dims=True)

            shift = tf.get_variable('shift', initializer=tf.constant(.01, shape=[channels]))
            scale = tf.get_variable('scale', initializer=tf.ones([channels]), dtype=tf.float32)

            tf.summary.histogram('shift', shift)
            tf.summary.histogram('scale', scale)

            normalized = (inputs - mu) / (sigma_sq + EPSILON) ** .5
            normalized = tf.add(tf.multiply(scale, normalized), shift)

        return normalized

    @staticmethod
    def leaky_relu(inputs, slope=.2):
        return tf.maximum(slope * inputs, inputs)

