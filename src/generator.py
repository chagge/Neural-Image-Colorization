import net
import tensorflow as tf


class Generator(net.Net):
    def __init__(self):
        net.Net.__init__(self)
        print("Initialized new 'Generator' instance")

    def build(self, inputs):
        with tf.variable_scope('generator'):
            # Encoder
            self.conv_e1 = self._conv_layer(inputs, 64)
            self.conv_e2 = self._conv_layer(self.conv_e1, 128, leaky=True, norm=False)
            self.conv_e3 = self._conv_layer(self.conv_e2, 256, leaky=True)
            self.conv_e4 = self._conv_layer(self.conv_e3, 512, leaky=True)
            self.conv_e5 = self._conv_layer(self.conv_e4, 512, leaky=True)
            self.conv_e6 = self._conv_layer(self.conv_e5, 512, leaky=True)
            self.conv_e7 = self._conv_layer(self.conv_e6, 512, leaky=True)
            self.conv_e8 = self._conv_layer(self.conv_e7, 512, leaky=True)

            # U-Net decoder
            self.conv_d1 = self._resid_layer(self.conv_e8, self.conv_e7, 512, dropout=True)
            self.conv_d2 = self._resid_layer(self.conv_d1, self.conv_e6, 512, dropout=True)
            self.conv_d3 = self._resid_layer(self.conv_d2, self.conv_e5, 512, dropout=True)
            self.conv_d4 = self._resid_layer(self.conv_d3, self.conv_e4, 512)
            self.conv_d5 = self._resid_layer(self.conv_d4, self.conv_e3, 512)
            self.conv_d6 = self._resid_layer(self.conv_d5, self.conv_e2, 256)
            self.conv_d7 = self._resid_layer(self.conv_d6, self.conv_e1, 128)
            self.output = self._get_output(self.conv_d7)

    # Restore a trained generator
    @staticmethod
    def restore(file_path):
        saver = tf.train.import_meta_graph(file_path + '.meta')
        saver.restore(tf.get_default_session(), file_path)

    def _deconv_layer(self, inputs, out_size, dropout=False):
        in_size = inputs.get_shape().as_list()[3]
        filters_shape = self.filter_shape + [out_size] + [in_size]
        filters = tf.Variable(tf.truncated_normal(mean=0., stddev=.1, shape=filters_shape))

        # Get dimensions to use for the deconvolution operator
        batch, height, width, channels = inputs.get_shape().as_list()
        out_height = height * self.sample_level
        out_width = width * self.sample_level
        out_size = filters_shape[2]
        out_shape = tf.pack([batch, out_height, out_width, out_size])

        # Deconvolve and normalize the biased outputs
        deconv = tf.nn.conv2d_transpose(inputs, filters, output_shape=out_shape, strides=self.stride)
        bias = tf.Variable(tf.constant(.1, shape=[out_size]))
        deconv = tf.nn.bias_add(deconv, bias)
        bn_maps = self._instance_normalize(deconv)

        if dropout:
            bn_maps = tf.nn.dropout(bn_maps, keep_prob=self.dropout_rate)

        return tf.nn.relu(bn_maps)

    def _get_output(self, inputs):
        conv = self._deconv_layer(inputs, 3)
        activation = tf.nn.sigmoid(conv)
        return activation

    def _resid_layer(self, inputs, skip_activations, out_size, dropout=False):
        conv = self._deconv_layer(inputs, out_size, dropout=dropout)
        return tf.concat(3, [conv, skip_activations])
