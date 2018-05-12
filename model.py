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


def custom_loss_function(logits, thetas, labels):
    """
    Custom loss function according to the paper.

    :param logits: should be shape of [batch_size, 18]
    :param thetas: each denoted by one-hot vector
    :param labels: each denoted by 0 or 1
    :return: reduce sum of loss for a batch
    """
    filtered_activation = tf.reduce_sum(logits*thetas, 1)
    # Reshape the activation, such that it shares the shape with labels
    filtered_activation = tf.reshape(filtered_activation, [-1, 1])
    entropys = - labels * tf.log(tf.clip_by_value(filtered_activation, 1e-8, 1.0))\
               - (1-labels) * tf.log(tf.clip_by_value((1-filtered_activation), 1e-8, 1.0))
    loss = tf.reduce_sum(entropys)

    return loss
