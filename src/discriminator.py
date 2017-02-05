import net
import tensorflow as tf


class Discriminator(net.Net):
    def __init__(self):
        net.Net.__init__(self)
        print("Initialized new 'Discriminator' instance")
        self.noise_multiplier = 1.
        self.noise_decay = 1e-8

    def predict(self, inputs):
        filter_size = 70
        stride = [1, 2, 2, 1]

        with tf.variable_scope('discriminator') as scope:
            inputs_ = self.add_noise(inputs, self.noise_multiplier)

            conv_e1_ = self.conv_layer(inputs_, 64,
                                      shape=[filter_size, filter_size, 3, 3], act=self.leaky_relu,
                                      stride=stride, norm=False, name='conv1')
            conv_e1 = self.add_noise(conv_e1_, self.noise_multiplier)

            conv_e2_ = self.conv_layer(conv_e1, 128,
                                      shape=[filter_size, filter_size, 3, 3], act=self.leaky_relu,
                                      stride=stride, name='conv2')
            conv_e2 = self.add_noise(conv_e2_, self.noise_multiplier)

            conv_e3_ = self.conv_layer(conv_e2, 256,
                                      shape=[filter_size, filter_size, 3, 3], act=self.leaky_relu,
                                      stride=stride, name='conv3')
            conv_e3 = self.add_noise(conv_e3_, self.noise_multiplier)

            conv_e4_ = self.conv_layer(conv_e3, 512,
                                      shape=[filter_size, filter_size, 3, 3], act=self.leaky_relu,
                                      stride=stride, name='conv4')
            conv_e4 = self.add_noise(conv_e4_, self.noise_multiplier)

            conv_e5 = self.conv_layer(conv_e4, 1,
                                      shape=[filter_size, filter_size, 3, 1], act=None,
                                      stride=stride, name='conv5')

            prediction = tf.nn.sigmoid(conv_e5, name='prediction')
            self.noise_multiplier = self.noise_multiplier - self.noise_decay
            scope.reuse_variables()
            return prediction, conv_e5
