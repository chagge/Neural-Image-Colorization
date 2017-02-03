import net
import tensorflow as tf


class Discriminator(net.Net):
    def __init__(self):
        net.Net.__init__(self)
        print("Initialized new 'Discriminator' instance")

    def predict(self, inputs):
        with tf.variable_scope('discriminator') as scope:
            conv_e1 = self.conv_layer(inputs, 64, shape=[1, 1, 3, 3], act=self.leaky_relu, norm=False, name='conv_e1')
            conv_e2 = self.conv_layer(conv_e1, 128, shape=[1, 1, 3, 3], act=self.leaky_relu, name='conv_e2')

            output = tf.nn.sigmoid(conv_e2)
            scope.reuse_variables()
            return output
