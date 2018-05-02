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


def cross_entropy(label, logit):
    """
    Calculate cross entropy.

    :param label: ground truth
    :param logit: prediction
    :return: cross entropy
    """
    # In order to avoid causing loss to NaN
    clip_min = 1e-7
    clip_max = 1e7
    return -label * tf.log(tf.clip_by_value(logit, clip_min, clip_max))\
           - (1-label) * tf.log(tf.clip_by_value(1-logit, clip_min, clip_max))


def custom_loss_function(logits, labels):
    """
    Custom loss function according to the paper.

    :param logits: should be shape of [batch_size, 18]
    :param labels: denoted by one-hot vector
    :return: reduce sum of loss for a batch
    """
    loss = 0.0
    for i in range(logits.shape[0]):
        index = tf.argmax(labels[i])
        loss += cross_entropy(labels[i][index], logits[i][index])

    return loss
