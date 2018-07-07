# coding=utf-8
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import model
import read_TFRecord
import shutil

flags = tf.flags

flags.DEFINE_string('log_dir', '../log/', 'log directory')
flags.DEFINE_string('model_name', 'model', 'model name')
flags.DEFINE_string('class_label', '', 'positive or negative class label')
flags.DEFINE_string('base_model', '', 'base model')

# State the batch_size to evaluate each time, which can be a lot more than the training batch
flags.DEFINE_integer('batch_size', 50, 'batch size')

# State the number of epochs to evaluate
flags.DEFINE_integer('num_epochs', 1, 'number of epochs')

flags.DEFINE_integer('gpu_id', 0, 'the id of gpu to use')

FLAGS = flags.FLAGS


def evaluate():
    """Main evaluating function to set out training."""

    with tf.Graph().as_default():
        if FLAGS.class_label == '':
            images, class_labels, theta_labels, num_samples = read_TFRecord.get_batch_data('Test', FLAGS.batch_size)
        else:
            images, class_labels, theta_labels, num_samples = read_TFRecord.get_one_class_batch_data(
                'Test', FLAGS.class_label, FLAGS.batch_size)

        num_steps_per_epoch = int(num_samples / FLAGS.batch_size)

        with tf.device('/device:GPU:' + str(FLAGS.gpu_id)):

            # Define the loss functions and get the total loss
            predictions = model.grasp_net(images, is_training=False, base_model=FLAGS.base_model)
            loss = model.custom_loss_function(predictions, theta_labels, class_labels)
            tf.losses.add_loss(loss)
            total_loss_op = tf.losses.get_total_loss()
        # Compute number of correctness
        accuracy_op, precision_op, recall_op = model.get_metrics(predictions, theta_labels, class_labels)

        tf.summary.scalar('eval/accuracy: ', accuracy_op)
        tf.summary.scalar('eval/precision: ', precision_op)
        tf.summary.scalar('eval/recall: ', recall_op)
        summary_op = tf.summary.merge_all()

        # Where model logs are stored
        model_log_dir = os.path.join(FLAGS.log_dir, FLAGS.model_name)
        # Get the latest checkpoint file
        checkpoint_file = tf.train.latest_checkpoint(model_log_dir)
        # Get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Where eval logs are stored
        eval_log_dir = os.path.join(
            FLAGS.log_dir, '%s_eval_%s_with_batch_of_%d' % (FLAGS.model_name, FLAGS.class_label, FLAGS.batch_size))
        # Make sure to overwrite the folders
        if os.path.exists(eval_log_dir):
            shutil.rmtree(eval_log_dir)
        os.mkdir(eval_log_dir)

        # Define a supervisor for running a managed session
        sv = tf.train.Supervisor(
            summary_op=None, logdir=eval_log_dir, init_fn=lambda session: saver.restore(session, checkpoint_file))

        total_num_correctness = 0
        total_examples = 0
        total_loss = 0
        # Run the managed session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            for step in range(num_steps_per_epoch * FLAGS.num_epochs):
                current_loss, accuracy, precision, recall, summaries = sess.run(
                    [total_loss_op, accuracy_op, precision_op, recall_op, summary_op])
                num_correctness = int(accuracy * FLAGS.batch_size)
                print('Step %d: loss = %.4f, accuracy = %.4f, precision = %.4f, recall = %.4f' %
                      (step, current_loss, accuracy, precision, recall))
                sv.summary_computed(sess, summaries, global_step=step)
                total_num_correctness += num_correctness
                total_examples += FLAGS.batch_size
                total_loss += current_loss

            average_loss = total_loss / (num_steps_per_epoch * FLAGS.num_epochs)
            final_accuracy = 1.0 * total_num_correctness / total_examples
            print('Final: loss = %.4f, accuracy = %.4f (%d / %d)' %
                  (average_loss, final_accuracy, total_num_correctness, total_examples))


if __name__ == '__main__':
    evaluate()
