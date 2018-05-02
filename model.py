# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.python.ops import variable_scope


vgg = nets.vgg


def grasp_net(images):
    """
    A net taking advantage of pretrained VGG16.

    :param images: size of 224x224
    :return: logits with shape of [batch_size, 18]
    """

    vgg_output, _ = vgg.vgg_16(images, is_training=True)
    tf.summary.scalar('vgg_output', tf.reduce_mean(vgg_output))

    # Add extra layers to the end of VGG16
    with variable_scope.variable_scope('extra_layers'):
        # Use sigmoid for the final layer
        net = slim.fully_connected(vgg_output, 18, scope='fc1', activation_fn=tf.sigmoid)

    return net


def custom_loss_function(logits, labels):
    """
    Custom loss function according to the paper.

    :param logits: should be shape of [batch_size, 18]
    :param labels: denoted by one-hot vector
    :return: reduce sum of loss for a batch
    """
    # Get the inner product of two
    inner_product = logits * labels
    # Get the indices of non-zero value
    zero = tf.constant(0, dtype=tf.float32)
    indices = tf.where(tf.not_equal(inner_product, zero))
    # Gather the non-zero values
    non_zeros = tf.gather_nd(inner_product, indices)
    # Calculate entropy elementwise
    entropys = -tf.log(tf.clip_by_value(non_zeros, 1e-8, 1.0))
    # Sum the entropy and get the final loss
    loss = tf.reduce_sum(entropys)

    return loss
