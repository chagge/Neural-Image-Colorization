import net
import tensorflow as tf


class Discriminator(net.Net):
    def __init__(self, filter_size=4):
        net.Net.__init__(self)
        print("Initialized new 'Discriminator' instance")
        self.filter_size = filter_size
        self.stride = [1, 2, 2, 1]
        self.is_training = True

    def predict(self, inputs):
        """
        Predicts the probability a given input belongs to a targeted sample distribution
        :param inputs: input tensor to predict with
        :return: output tensor predicting the probability the input belongs to targeted sample distribution
        """

        with tf.variable_scope('discriminator'):
            inputs_ = self.add_noise(inputs)

            conv1 = self.conv_layer(
                inputs_, 64,
                activation=self.leaky_relu,
                stride=self.stride,
                normalize=False,
                noisy=True,
                name='conv1')

            conv2 = self.conv_layer(
                conv1, 128,
                activation=self.leaky_relu,
                stride=self.stride,
                noisy=True,
                name='conv2')

            conv3 = self.conv_layer(
                conv2, 256,
                activation=self.leaky_relu,
                stride=self.stride,
                noisy=True,
                name='conv3')

            conv4 = self.conv_layer(
                conv3, 512,
                activation=self.leaky_relu,
                stride=self.stride,
                noisy=True,
                name='conv4')

            output = self.conv_layer(
                conv4, 1,
                activation=tf.nn.sigmoid,
                stride=self.stride,
                noisy=False,
                name='conv5')

        output = tf.reduce_mean(output)
        tf.get_variable_scope().reuse_variables()
        return output
