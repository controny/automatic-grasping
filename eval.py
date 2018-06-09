# coding=utf-8
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import model
import read_TFRecord

flags = tf.flags

flags.DEFINE_string('log_dir', '../log/', 'log directory')
flags.DEFINE_string('model_name', 'model', 'model name')

# State the batch_size to evaluate each time, which can be a lot more than the training batch
flags.DEFINE_integer('batch_size', 50, 'batch size')

# State the number of epochs to evaluate
flags.DEFINE_integer('num_epochs', 1, 'number of epochs')

FLAGS = flags.FLAGS


def evaluate():
    """Main evaluating function to set out training."""

    with tf.Graph().as_default():
        images, class_labels, theta_labels, num_samples = read_TFRecord.get_batch_data('Test', FLAGS.batch_size)

        num_steps_per_epoch = int(num_samples / FLAGS.batch_size)

        # Define the loss functions and get the total loss
        with slim.arg_scope(model.alexnet_v2_arg_scope()):
            predictions = model.grasp_net(images, is_training=False)
        loss = model.custom_loss_function(predictions, theta_labels, class_labels)
        tf.losses.add_loss(loss)
        total_loss_op = tf.losses.get_total_loss()
        # Compute number of correctness
        num_correctness_op = model.get_num_correctness(predictions, theta_labels, class_labels)

        # Where model logs are stored
        model_log_dir = os.path.join(FLAGS.log_dir, FLAGS.model_name)
        # Get the latest checkpoint file
        checkpoint_file = tf.train.latest_checkpoint(model_log_dir)
        # Get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(session):
            """A Saver function to later restore our model."""
            return saver.restore(session, checkpoint_file)

        # Define a supervisor for running a managed session
        sv = tf.train.Supervisor(summary_op=None, init_fn=restore_fn)

        total_num_correctness = 0
        total_examples = 0
        total_loss = 0
        # Run the managed session
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * FLAGS.num_epochs):
                current_loss, num_correctness = sess.run([total_loss_op, num_correctness_op])
                accuracy = 1.0 * num_correctness / FLAGS.batch_size
                print('Step %d: loss = %.4f, accuracy = %.4f (%d / %d)' %
                      (step, current_loss, accuracy, num_correctness, FLAGS.batch_size))
                total_num_correctness += num_correctness
                total_examples += FLAGS.batch_size
                total_loss += current_loss

            average_loss = total_loss / (num_steps_per_epoch * FLAGS.num_epochs)
            final_accuracy = 1.0 * total_num_correctness / total_examples
            print('Final: loss = %.4f, accuracy = %.4f (%d / %d)' %
                  (average_loss, final_accuracy, total_num_correctness, total_examples))


if __name__ == '__main__':
    evaluate()
