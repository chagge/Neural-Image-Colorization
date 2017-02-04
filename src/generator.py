import net
import tensorflow as tf


class Generator(net.Net):
    def __init__(self):
        net.Net.__init__(self)
        print("Initialized new 'Generator' instance")

    def build(self, inputs):
        with tf.variable_scope('generator'):
            # Encoder
            self.conv_e1 = self.conv_layer(inputs, 64, act=self.leaky_relu, norm=False, name='conv_e1')
            self.conv_e2 = self.conv_layer(self.conv_e1, 128, act=self.leaky_relu, name='conv_e2')
            self.conv_e3 = self.conv_layer(self.conv_e2, 256, act=self.leaky_relu, name='conv_e3')
            self.conv_e4 = self.conv_layer(self.conv_e3, 512, act=self.leaky_relu, name='conv_e4')
            self.conv_e5 = self.conv_layer(self.conv_e4, 512, act=self.leaky_relu, name='conv_e5')
            self.conv_e6 = self.conv_layer(self.conv_e5, 512, act=self.leaky_relu, name='conv_e6')
            self.conv_e7 = self.conv_layer(self.conv_e6, 512, act=self.leaky_relu, name='conv_e7')
            self.conv_e8 = self.conv_layer(self.conv_e7, 512, act=self.leaky_relu, name='conv_e8')

            self.bottleneck = self.conv_layer(self.conv_e8, 1, act=self.leaky_relu, norm=False, name='bottleneck')

            # U-Net decoder
            self.conv_d1 = self.__resid_layer(self.bottleneck, self.conv_e8, 512, dropout=True, name='conv_d1')
            self.conv_d2 = self.__resid_layer(self.conv_d1, self.conv_e7, 512, dropout=True, name='conv_d2')
            self.conv_d3 = self.__resid_layer(self.conv_d2, self.conv_e6, 512, dropout=True, name='conv_d3')
            self.conv_d4 = self.__resid_layer(self.conv_d3, self.conv_e5, 512, name='conv_d4')
            self.conv_d5 = self.__resid_layer(self.conv_d4, self.conv_e4, 512, name='conv_d5')
            self.conv_d6 = self.__resid_layer(self.conv_d5, self.conv_e3, 256, name='conv_d6')
            self.conv_d7 = self.__resid_layer(self.conv_d6, self.conv_e2, 128, name='conv_d7')
            self.conv_d8 = self.__resid_layer(self.conv_d7, self.conv_e1, 64, name='conv_d8')
            self.output_ = self.conv_layer(self.conv_d8, 2, stride=[1, 1, 1, 1], act=tf.nn.sigmoid, name='output')
            self.output = tf.concat(3, [self.output_, inputs])

    # Restore a trained generator
    @staticmethod
    def restore(file_path):
        saver = tf.train.import_meta_graph(file_path + '.meta')
        saver.restore(tf.get_default_session(), file_path)

    def __deconv_layer(self, inputs, out_size, name, norm=False, dropout=False):
        with tf.variable_scope(name):
            in_size = inputs.get_shape().as_list()[3]
            filters_shape = self.filter_shape + [out_size] + [in_size]
            filters = tf.Variable(tf.truncated_normal(mean=0., stddev=.1, shape=filters_shape), name='filters')

            # Get dimensions to use for the deconvolution operator
            shape = tf.shape(inputs)
            out_height = shape[1] * self.sample_level
            out_width = shape[2] * self.sample_level
            out_size = filters_shape[2]
            out_shape = tf.pack([shape[0], out_height, out_width, out_size])

            # Deconvolve and normalize the biased outputs
            deconv = tf.nn.conv2d_transpose(inputs, filters, output_shape=out_shape, strides=self.stride)
            bias = tf.Variable(tf.constant(.1, shape=[out_size]), name='bias')
            deconv = tf.nn.bias_add(deconv, bias)

            if norm:
                deconv = self.instance_normalize(deconv)

            if dropout:
                deconv = tf.nn.dropout(deconv, keep_prob=self.dropout_keep)

            return tf.nn.relu(deconv)

    def __resid_layer(self, inputs, skip_activations, out_size, name, norm=True, dropout=False):
        concat = tf.concat(3, [inputs, skip_activations])
        conv = self.__deconv_layer(concat, out_size, name, norm, dropout)
        return conv