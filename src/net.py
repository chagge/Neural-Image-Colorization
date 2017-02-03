import tensorflow as tf

DROPOUT_RATE = .5
EPSILON = 1e-8
FILTER_SHAPE = [4, 4]
SAMPLE_LEVEL = 2
STRIDE = [1, SAMPLE_LEVEL, SAMPLE_LEVEL, 1]


class Net(object):
    def __init__(self):
        self.dropout_rate = DROPOUT_RATE
        self.epsilon = EPSILON
        self.filter_shape = FILTER_SHAPE
        self.sample_level = SAMPLE_LEVEL
        self.stride = STRIDE

    # Constructs and returns a network layer tailored to one's specifications
    def conv_layer(self, inputs, out_size, name, shape=None, norm=True, dropout=False, act=tf.nn.relu, stride=STRIDE):
        with tf.variable_scope(name):
            if not shape:
                in_size = inputs.get_shape().as_list()[3]
                shape = self.filter_shape + [in_size] + [out_size]

            filters = tf.Variable(tf.truncated_normal(mean=0., stddev=.1, shape=shape))
            x = tf.nn.conv2d(inputs, filters, padding='SAME', strides=stride)
            num_out_maps = shape[3]
            bias = tf.Variable(tf.constant(.1, shape=[num_out_maps]), name='bias')
            x = tf.nn.bias_add(x, bias)

            if norm:
                x = self.instance_normalize(x)

            if dropout:
                x = tf.nn.dropout(x, keep_prob=self.dropout_rate)

            x = act(x)
            return x

    # Instance normalize inputs to reduce covariate shift and reduce dependency on input contrast to improve results
    @staticmethod
    def instance_normalize(inputs):
        with tf.variable_scope('instance_normalization'):
            batch, height, width, channels = [_.value for _ in inputs.get_shape()]
            mu, sigma_sq = tf.nn.moments(inputs, [1, 2], keep_dims=True)

            shift = tf.Variable(tf.constant(.1, shape=[channels]))
            scale = tf.Variable(tf.ones([channels]))
            normalized = (inputs - mu) / (sigma_sq + EPSILON) ** .5

            return scale * normalized + shift

    @staticmethod
    def leaky_relu(inputs):
        return tf.maximum(.2 * inputs, inputs)

