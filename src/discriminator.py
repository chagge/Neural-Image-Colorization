import net
import tensorflow as tf


class Discriminator(net.Net):
    def __init__(self):
        net.Net.__init__(self)
        print("Initialized new 'Discriminator' instance")

    def predict(self, inputs):
        with tf.variable_scope('discriminator') as scope:
            conv_e1 = self._conv_layer(inputs, 64, filters_shape=[1, 1, 3, 3], leaky=True, norm=False)
            conv_e2 = self._conv_layer(conv_e1, 128, filters_shape=[1, 1, 3, 3], leaky=True)
            conv_e3 = self._conv_layer(conv_e2, 3, filters_shape=[1, 1, 3, 3], norm=False)
            output = tf.nn.sigmoid(tf.reduce_mean(conv_e3))
            scope.reuse_variables()
            return output
