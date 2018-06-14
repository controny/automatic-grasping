# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

resnet = nets.resnet_v2


def grasp_net(images, is_training=True, lmbda=0.0, base_model=''):
    num_classes = 18
    if base_model == 'alexnet':
        with slim.arg_scope(alexnet_v2_arg_scope(lmbda)):
            return alexnet_v2(images, is_training, num_classes)
    elif base_model.startswith('resnet'):
        with slim.arg_scope(resnet.resnet_arg_scope()):
            if base_model == 'resnet200':
                net, _ = resnet.resnet_v2_200(images, num_classes=1024, is_training=is_training, reuse=tf.AUTO_REUSE)
            elif base_model == 'resnet50':
                net, _ = resnet.resnet_v2_50(images, num_classes=1024, is_training=is_training, reuse=tf.AUTO_REUSE)
            else:
                raise ValueError('Unexpected base model')
        with tf.variable_scope('extra', reuse=tf.AUTO_REUSE):
            net = slim.fully_connected(net, num_classes, tf.sigmoid)
            net = tf.squeeze(net)
        return net
    else:
        raise ValueError('Unexpected base model')


def vgg_16(images, is_training, num_classes):
    """
    A net taking advantage of VGG16

    :param images: size of 224x224
    :param is_training: whether training or not
    :param num_classes: number of final classes
    :return: logits with shape of [batch_size, 18]
    """
    with tf.variable_scope('vgg_16', 'vgg_16', [images], reuse=tf.AUTO_REUSE) as sc:
        dropout_keep_prob = 0.5
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected, slim.max_pool2d]):
            net = slim.repeat(
                images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected slim.
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            net = slim.dropout(
                net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(
                net, dropout_keep_prob, is_training=is_training, scope='dropout7')
            net = slim.conv2d(
                net,
                num_classes, [1, 1],
                activation_fn=tf.sigmoid,
                normalizer_fn=None,
                scope='fc8')
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')

            return net


def alexnet_v2_arg_scope(weight_decay=0.0005):
    """Used when calculating predictions."""
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                return arg_sc


def alexnet_v2(images, is_training, num_classes):
    """
    A net taking advantage of AlexNet V2.

    :param images: size of 224x224
    :param is_training: whether training or not
    :param num_classes: number of final classes
    :return: logits with shape of [batch_size, 18]
    """
    dropout_keep_prob = 0.5
    with tf.variable_scope('alexnet_v2', 'alexnet_v2', [images], reuse=tf.AUTO_REUSE) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected, slim.max_pool2d],
                outputs_collections=[end_points_collection]):
            net = slim.conv2d(
                images, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

            # Use conv2d instead of fully_connected slim.
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.005),
                    biases_initializer=tf.constant_initializer(0.1)):
                net = slim.conv2d(net, 4096, [5, 5], padding='VALID', scope='fc6')
                net = slim.dropout(
                    net, dropout_keep_prob, is_training=is_training, scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(
                    net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                net = slim.conv2d(
                    net,
                    num_classes, [1, 1],
                    activation_fn=tf.sigmoid,
                    normalizer_fn=None,
                    biases_initializer=tf.zeros_initializer(),
                    scope='fc8')

            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')

    return net


def custom_loss_function(logits, theta_labels, class_labels):
    """
    Custom loss function according to the paper.

    :param logits: should be shape of [batch_size, 18]
    :param theta_labels: each denoted by an one-hot vector
    :param class_labels: each denoted by 0 or 1, should be converted to float
    :return: reduce mean of loss for a batch
    """
    class_labels = tf.cast(class_labels, tf.float32)
    filtered_scores = tf.reduce_sum(logits*theta_labels, 1)
    # Reshape the scores, such that it shares the shape with labels
    filtered_scores = tf.reshape(filtered_scores, [-1, 1])
    clipped_scores = tf.clip_by_value(filtered_scores, 10e-8, 1.0-10e-8)
    entropys = - class_labels * tf.log(clipped_scores)\
               - (1-class_labels) * tf.log(1-clipped_scores)
    loss = tf.reduce_mean(entropys)

    return loss


def get_metrics(logits, theta_labels, class_labels):
    """
    Function to calculate accuracy.

    :param logits: should be shape of [batch_size, 18]
    :param theta_labels: each denoted by an one-hot vector
    :param class_labels: each denoted by 0 or 1
    :return: accuray, precision and recall
    """
    # Extract outputs of coresponding angles
    angle_outputs = tf.reduce_sum(theta_labels * logits, 1)
    # angle_outputs = tf.Print(angle_outputs, [angle_outputs], 'predition: ', summarize=10)
    # Convert class labels to 1-D array
    class_labels = tf.cast(tf.squeeze(class_labels), tf.int32)
    # class_labels = tf.Print(class_labels, [class_labels], 'label: ', summarize=10)
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
    # Compute number of false positive and false negative
    num_positive_labels = tf.reduce_sum(p_label_indexes)
    num_negative_labels = tf.reduce_sum(n_label_indexes)
    num_false_p = num_positive_labels - num_true_p
    num_false_n = num_negative_labels - num_true_n
    # Compute accuracy, precision and recall
    accuracy = (num_true_p + num_true_n) / (num_positive_labels + num_negative_labels)
    precison = num_true_p / (num_true_p + num_false_p)
    recall = num_true_p / (num_true_p + num_false_n)

    return accuracy, precison, recall
