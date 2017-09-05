import tensorflow as tf

# convolution/pool stride
_CONV_KERNEL_STRIDES_ = [1, 2, 2, 1]
_DECONV_KERNEL_STRIDES_ = [1, 2, 2, 1]
_REGULAR_FACTOR_ = 1.0e-4

def conv2d_layer(input_layer, output_dim, kernel_size = 3, stddev = 0.02, name = 'conv2d'):
    with tf.variable_scope(name):
        init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
        filter_size = [kernel_size, kernel_size, input_layer.get_shape()[-1], output_dim]
        weight = tf.get_variable(
            name = name + 'weight',
            shape = filter_size,
            initializer = init_weight,
            regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
        bias = tf.get_variable(
            name = name + 'bias',
            shape = [output_dim],
            initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_layer, weight, _CONV_KERNEL_STRIDES_, padding = 'SAME')
        conv = tf.nn.bias_add(conv, bias)
        return conv

def deconv2d_layer(input_layer, output_shape, kernel_size = 2, stddev = 0.02, name = 'deconv'):
    with tf.variable_scope(name):
        init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
        filter_size = [kernel_size, kernel_size, output_shape[-1], input_layer.get_shape()[-1]]
        weight = tf.get_variable(
            name = name + 'weight',
            shape = filter_size,
            initializer = init_weight,
            regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
        bias = tf.get_variable(
            name = name + 'bias',
            shape = [output_shape[-1]],
            initializer = tf.constant_initializer(0.0))
        deconv = tf.nn.conv2d_transpose(input_layer, weight, output_shape, strides = _DECONV_KERNEL_STRIDES_, padding = 'SAME')
        deconv = tf.nn.bias_add(deconv, bias)
        return deconv

def lrelu(input_layer, leak = 0.2, name = 'lrelu'):
    with tf.variable_scope(name):
        alpha1 = 0.5 * (1 + leak)
        alpha2 = 0.5 * (1 - leak)
        return alpha1 * input_layer + alpha2 * abs(input_layer)

def full_connection_layer(input_layer, output_dim, stddev = 0.02, name = 'fc'):
    # calculate input_layer dimension and reshape to batch * dimension
    input_dimension = 1
    for dim in input_layer.get_shape().as_list()[1:]:
        input_dimension *= dim

    with tf.variable_scope(name):
        init_weight = tf.truncated_normal_initializer(mean = 0.0, stddev = stddev, dtype = tf.float32)
        filter_size = [input_dimension, output_dim]
        weight = tf.get_variable(
            name = name + 'weight',
            shape = filter_size,
            initializer = init_weight,
            regularizer = tf.contrib.layers.l2_regularizer(_REGULAR_FACTOR_))
        bias = tf.get_variable(
            name = name + 'bias',
            shape = [output_dim],
            initializer = tf.constant_initializer(0.0))
        input_layer_reshape = tf.reshape(input_layer, [-1, input_dimension])
        fc = tf.matmul(input_layer_reshape, weight)
        tc = tf.nn.bias_add(fc, bias)
        return fc

class BatchNormalization:
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                decay=self.momentum,
                updates_collections=None,
                epsilon=self.epsilon,
                scale=True,
                is_training=train,
                scope=self.name)
