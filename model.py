# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from tensorflow.python.ops import variable_scope
from train import add_summary_op


vgg = nets.vgg


def grasp_net(images, summary_ops):
    """A net taking advantage of pretrained VGG16."""

    vgg_output, _ = vgg.vgg_16(images, is_training=True)
    add_summary_op(summary_ops, tf.reduce_mean(vgg_output), 'vgg_output')

    # Add extra layers to the end of VGG16
    with variable_scope.variable_scope('extra_layers'):
        # Use sigmoid for the final layer
        net = slim.fully_connected(vgg_output, 18, scope='fc1', activation_fn=tf.nn.sigmoid)

    return net
