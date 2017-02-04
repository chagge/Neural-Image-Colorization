import tensorflow as tf


def rgb2yuv(rgb):
    c = tf.constant([
        [.299, .587, .114],
        [-.14713, -.28886, .436],
        [.615, -.51499, -.10001]
    ])

    # Transform the pixel intensity values
    rgb_shape = rgb.get_shape().as_list()
    rgb = tf.reshape(rgb, [3, rgb_shape[1] * rgb_shape[2]])
    yuv = tf.matmul(c, rgb)
    yuv = tf.reshape(yuv, [1, rgb_shape[1], rgb_shape[2], 3])

    # Split the luminance channel and fit it into the appropriate shape
    y_ = tf.reshape(yuv, [3, rgb_shape[1], rgb_shape[2]])
    y = tf.reshape(y_[0], [1, rgb_shape[1], rgb_shape[2], 1])

    return yuv, y


def yuv2rgb(yuv):
    c = tf.constant([
        [1., 0., 1.13983],
        [1., -.39465, -.58060],
        [1., 2.03211, 0.]
    ])

    # Transform the pixel intensity values
    yuv_shape = yuv.get_shape().as_list()
    yuv = tf.reshape(yuv, [3, yuv_shape[1] * yuv_shape[2]])
    rgb = tf.matmul(c, yuv)
    rgb = tf.reshape(rgb, [1, yuv_shape[1], yuv_shape[2], 3])

    return rgb


def y_uv(y, uv):
    shape = y.get_shape().as_list()
    y_ = tf.reshape(y, [1, shape[1], shape[2]])
    uv_ = tf.reshape(uv, [2, shape[1], shape[2]])
    yuv_ = tf.concat(0, [y_, uv_])
    yuv = tf.reshape(yuv_, [1, shape[1], shape[2], 3])
    return yuv


def exit_program(rc=0, message="Exiting the program.."):
    print(message)
    tf.get_default_session().close()
    exit(rc)
