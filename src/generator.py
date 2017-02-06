import net
import tensorflow as tf


class Generator(net.Net):
    def __init__(self):
        net.Net.__init__(self)
        print("Initialized new 'Generator' instance")
        self.is_training = True

    def build(self, inputs):
        with tf.variable_scope('generator'):

            # Encoder
            self.conv1e = self.conv_layer(
                inputs, 64,
                activation=self.leaky_relu,
                normalize=False,
                name='conv1e')

            self.conv2e = self.conv_layer(
                self.conv1e, 128,
                activation=self.leaky_relu,
                name='conv2e')

            self.conv3e = self.conv_layer(
                self.conv2e, 256,
                activation=self.leaky_relu,
                name='conv3e')

            self.conv4e = self.conv_layer(
                self.conv3e, 512,
                activation=self.leaky_relu,
                name='conv4e')

            self.conv5e = self.conv_layer(
                self.conv4e, 512,
                activation=self.leaky_relu,
                name='conv5e')

            self.conv6e = self.conv_layer(
                self.conv5e, 512,
                activation=self.leaky_relu,
                name='conv6e')

            self.conv7e = self.conv_layer(
                self.conv6e, 512,
                activation=self.leaky_relu,
                name='conv7e')

            self.conv8e = self.conv_layer(
                self.conv7e, 512,
                activation=self.leaky_relu,
                name='conv8e')

            # U-Net decoder
            self.conv1d = self.__resid_layer(
                self.conv8e, self.conv7e, 512,
                dropout=True,
                name='conv1d')

            self.conv2d = self.__resid_layer(
                self.conv1d, self.conv6e, 512,
                dropout=True,
                name='conv2d')

            self.conv3d = self.__resid_layer(
                self.conv2d, self.conv5e, 512,
                dropout=True,
                name='conv3d')

            self.conv4d = self.__resid_layer(
                self.conv3d, self.conv4e, 512,
                name='conv4d')

            self.conv5d = self.__resid_layer(
                self.conv4d, self.conv3e, 256,
                name='conv5d')

            self.conv6d = self.__resid_layer(
                self.conv5d, self.conv2e, 128,
                name='conv6d')

            self.conv7d = self.__resid_layer(
                self.conv6d, self.conv1e, 64,
                name='conv7d')

            self.output = self.__resid_layer(
                self.conv7d, inputs, 2,
                activation=tf.nn.sigmoid,
                name='conv8d')

    # Restore a trained generator
    @staticmethod
    def restore(file_path):
        saver = tf.train.import_meta_graph(file_path + '.meta')
        saver.restore(tf.get_default_session(), file_path)

    def __deconv_layer(
            self, inputs, out_size, activation, name,
            normalize=True,
            dropout=False):

        with tf.variable_scope(name):
            in_size = inputs.get_shape().as_list()[3]
            filters_shape = self.filter_shape + [out_size] + [in_size]
            filters = tf.Variable(
                tf.truncated_normal(
                    mean=0.,
                    stddev=.02,
                    shape=filters_shape),
                name='filters')

            # Get dimensions to use for the deconvolution operator
            shape = tf.shape(inputs)
            out_height = shape[1] * self.sample_level
            out_width = shape[2] * self.sample_level
            out_size = filters_shape[2]
            out_shape = tf.pack([shape[0], out_height, out_width, out_size])

            # Deconvolve and normalize the biased outputs
            deconv = tf.nn.conv2d_transpose(
                inputs, filters,
                output_shape=out_shape,
                strides=self.stride)
            bias = tf.Variable(
                tf.constant(
                    .1,
                    shape=[out_size]),
                name='bias')
            deconv = tf.nn.bias_add(deconv, bias)

            deconv = self.batch_normalize(deconv, self.is_training) if normalize else deconv
            deconv = tf.nn.dropout(deconv, keep_prob=self.dropout_keep) if dropout else deconv
            return activation(deconv)

    def __resid_layer(self, inputs, skip_activations, out_size, name,
                      activation=tf.nn.relu,
                      normalize=True,
                      dropout=False):

        conv_ = self.__deconv_layer(
            inputs, out_size,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
            name=name)

        conv = tf.concat(3, [conv_, skip_activations])
        return conv
