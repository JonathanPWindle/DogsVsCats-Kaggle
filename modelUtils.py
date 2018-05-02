import tensorflow as tf


########################################################################################################################
#   Initialised with a slight bit of noise to break symmetry and prevent 0 gradients
#   shape: array to define matrix shape holding weights
########################################################################################################################
def weightVariable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


########################################################################################################################
#   Slightly positive bias avoids "dead neurons"
#   shape: array to define matrix shape holding weights
#######################################################################################################################
def biasVariable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


########################################################################################################################
#   Convolutional layer definition
#   x: inputs
#   W: weights related to layer
########################################################################################################################
def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


########################################################################################################################
#   Use Max pooling over 2x2 blocks
#   x: inputs
########################################################################################################################
def maxPool2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")