import tensorflow as tf

EPSILON = 1e-8


class Net(object):
    def __init__(self):
        self.dropout_rate = .5
        self.filter_shape = [4, 4]
        self.sample_level = 2
        self.stride = [1, self.sample_level, self.sample_level, 1]

    # Constructs and returns a network layer tailored to one's specifications
    def _conv_layer(self, inputs, out_size, filters_shape=None, leaky=False, norm=True, dropout=False):
        if not filters_shape:
            in_size = inputs.get_shape().as_list()[3]
            filters_shape = self.filter_shape + [in_size] + [out_size]

        filters = tf.Variable(tf.truncated_normal(mean=0., stddev=.1, shape=filters_shape))
        x = tf.nn.conv2d(inputs, filters, padding='SAME', strides=self.stride)
        num_out_maps = filters_shape[3]
        bias = tf.Variable(tf.constant(.1, shape=[num_out_maps]))
        x = tf.nn.bias_add(x, bias)

        if norm:
            x = self._instance_normalize(x)

        if dropout:
            x = tf.nn.dropout(x, keep_prob=self.dropout_rate)

        if leaky:
            x = tf.maximum(.2 * x, x)
        else:
            x = tf.nn.relu(x)

        return x

    # Instance normalize inputs to reduce covariate shift and reduce dependency on input contrast to improve results
    @staticmethod
    def _instance_normalize(inputs):
        with tf.variable_scope('instance_normalization'):
            batch, height, width, channels = [_.value for _ in inputs.get_shape()]
            mu, sigma_sq = tf.nn.moments(inputs, [1, 2], keep_dims=True)

            shift = tf.Variable(tf.constant(.1, shape=[channels]))
            scale = tf.Variable(tf.ones([channels]))
            normalized = (inputs - mu) / (sigma_sq + EPSILON) ** .5

            return scale * normalized + shift

