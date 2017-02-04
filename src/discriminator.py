import net
import tensorflow as tf


class Discriminator(net.Net):
    def __init__(self):
        net.Net.__init__(self)
        print("Initialized new 'Discriminator' instance")

    def predict(self, inputs):
        filter_size = 70
        stride = [1, 2, 2, 1]

        with tf.variable_scope('discriminator') as scope:
            conv_e1 = self.conv_layer(inputs, 64,
                                      shape=[filter_size, filter_size, 3, 3], act=self.leaky_relu,
                                      stride=stride, norm=False, name='conv1')

            conv_e2 = self.conv_layer(conv_e1, 128,
                                      shape=[filter_size, filter_size, 3, 3], act=self.leaky_relu,
                                      stride=stride, name='conv2')

            conv_e3 = self.conv_layer(conv_e2, 256,
                                      shape=[filter_size, filter_size, 3, 3], act=self.leaky_relu,
                                      stride=stride, name='conv3')

            conv_e4 = self.conv_layer(conv_e3, 512,
                                      shape=[filter_size, filter_size, 3, 3], act=self.leaky_relu,
                                      stride=stride, name='conv4')

            conv_e5 = self.conv_layer(conv_e4, 1,
                                      shape=[filter_size, filter_size, 3, 1], act=None,
                                      stride=stride, name='conv5')

            prediction = tf.nn.sigmoid(conv_e5, name='prediction')
            scope.reuse_variables()
            return prediction, conv_e5
