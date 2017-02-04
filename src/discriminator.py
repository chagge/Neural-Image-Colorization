import net
import tensorflow as tf


class Discriminator(net.Net):
    def __init__(self):
        net.Net.__init__(self)
        print("Initialized new 'Discriminator' instance")

    def predict(self, inputs):
        with tf.variable_scope('discriminator') as scope:
            stride = [1, 1, 1, 1]
            conv_e1 = self.conv_layer(inputs, 64, shape=[1, 1, 3, 3], act=self.leaky_relu, stride=stride, norm=False, name='conv_e1')
            conv_e2 = self.conv_layer(conv_e1, 128, shape=[1, 1, 3, 1], act=self.leaky_relu, stride=stride, name='conv_e2')
            conv_e3 = self.conv_layer(conv_e2, 128, shape=[1, 1, 1, 1], act=None, stride=stride, name='conv_e2')

            prediction = tf.nn.sigmoid(conv_e3)
            scope.reuse_variables()
            return prediction, conv_e3
