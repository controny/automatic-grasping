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


def custom_loss_function(logits, theta_labels, class_labels):
    """
    Custom loss function according to the paper.

    :param logits: should be shape of [batch_size, 18]
    :param theta_labels: each denoted by a int, should be converted to one-hot vector
    :param class_labels: each denoted by 0 or 1, should be converted to float
    :return: reduce sum of loss for a batch
    """
    theta_labels = tf.one_hot(theta_labels, 18)
    class_labels = tf.reshape(tf.cast(class_labels, tf.float32), [-1, 1])
    print('after processing')
    print('class_labels', class_labels)
    print('theta_labels', theta_labels)
    filtered_activation = tf.reduce_sum(logits*theta_labels, 1)
    # Reshape the activation, such that it shares the shape with labels
    filtered_activation = tf.reshape(filtered_activation, [-1, 1])
    entropys = - class_labels * tf.log(tf.clip_by_value(filtered_activation, 1e-8, 1.0))\
               - (1-class_labels) * tf.log(tf.clip_by_value((1-filtered_activation), 1e-8, 1.0))
    loss = tf.reduce_sum(entropys)

    return loss


def get_num_correctness(logits, theta_labels, class_labels):
    """
    Function to calculate accuracy.

    :param logits: should be shape of [batch_size, 18]
    :param theta_labels: each denoted by a int
    :param class_labels: each denoted by 0 or 1
    :return: number of correctness
    """
    theta_labels = tf.one_hot(theta_labels, 18)
    # Extract outputs of coresponding angles
    angle_outputs = tf.reduce_sum(theta_labels * logits, 1)
    # Get positive and negative indexes of labels
    # Remember to cast bool to int for later computing
    p_label_indexes = tf.cast(tf.equal(class_labels, tf.ones(class_labels.shape, dtype=tf.int32)), tf.int32)
    n_label_indexes = tf.cast(tf.equal(class_labels, tf.zeros(class_labels.shape, dtype=tf.int32)), tf.int32)
    threshold = tf.constant(0.5, shape=angle_outputs.shape)
    # Among the outputs of angles, those >= threshold will be considered positive and vice versa
    p_logits_indexes = tf.cast(tf.greater_equal(angle_outputs, threshold), tf.int32)
    n_logits_indexes = tf.cast(tf.less(angle_outputs, threshold), tf.int32)
    # Finally, we can calculate numbers of true positive and true negative
    num_true_p = tf.reduce_sum(p_label_indexes * p_logits_indexes)
    num_true_n = tf.reduce_sum(n_label_indexes * n_logits_indexes)

    return num_true_p + num_true_n
