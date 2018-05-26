# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets


vgg = nets.vgg


def grasp_net(images, is_training=True):
    """
    A net taking advantage of pretrained VGG16.

    :param images: size of 224x224
    :param is_training: whether training or not
    :return: logits with shape of [batch_size, 18]
    """

    vgg_output, _ = vgg.vgg_16(images, num_classes=18, is_training=is_training)
    # Apply sigmoid to the output
    net = tf.sigmoid(vgg_output)

    return net


def custom_loss_function(logits, theta_labels, class_labels):
    """
    Custom loss function according to the paper.

    :param logits: should be shape of [batch_size, 18]
    :param theta_labels: each denoted by an one-hot vector
    :param class_labels: each denoted by 0 or 1, should be converted to float
    :return: reduce sum of loss for a batch
    """
    class_labels = tf.cast(class_labels, tf.float32)
    filtered_scores = tf.reduce_sum(logits*theta_labels, 1)
    # Reshape the scores, such that it shares the shape with labels
    filtered_scores = tf.reshape(filtered_scores, [-1, 1])
    clipped_scores = tf.clip_by_value(filtered_scores, 0.001, 0.999)
    entropys = - class_labels * tf.log(clipped_scores)\
               - (1-class_labels) * tf.log(1-clipped_scores)
    loss = tf.reduce_sum(entropys)

    return loss


def get_num_correctness(logits, theta_labels, class_labels):
    """
    Function to calculate accuracy.

    :param logits: should be shape of [batch_size, 18]
    :param theta_labels: each denoted by an one-hot vector
    :param class_labels: each denoted by 0 or 1
    :return: number of correctness
    """
    # Extract outputs of coresponding angles
    angle_outputs = tf.reduce_sum(theta_labels * logits, 1)
    # Convert class labels to 1-D array
    class_labels = tf.cast(tf.squeeze(class_labels), tf.int32)
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
