import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import os
import shutil
import model
import read_TFRecord

# Input setting
image_size = 224
channel_num = 3
theta_label_size = 18
class_label_size = 1

flags = tf.flags

# File paths
flags.DEFINE_string('log_dir', '../log/', 'log directory')
flags.DEFINE_string('model_name', 'model', 'model name')
flags.DEFINE_string('pretrained_model_path', '../pretrained_model/vgg_16.ckpt', 'pretrained mode path')

# Training parameters
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('validation_batch_size', 1000, 'batch size for validation')
flags.DEFINE_integer('num_epochs', 3, 'number of epochs')
flags.DEFINE_integer('logging_gap', 50, 'logging gap')
flags.DEFINE_integer('num_epochs_before_decay', 10, 'number of epochs before decay')
flags.DEFINE_float('initial_learning_rate', 0.0001, 'initial learning rate')
flags.DEFINE_float('learning_rate_decay_factor', 0.7, 'learning rate decay factor')

FLAGS = flags.FLAGS


def train():
    """Main training function to set out training."""

    with tf.Graph().as_default():
        training_images, training_class_labels, training_theta_labels, num_training_samples =\
            read_TFRecord.get_batch_data('Train', FLAGS.batch_size)
        validation_images, validation_class_labels, validation_theta_labels, num_validation_samples = \
            read_TFRecord.get_batch_data('Validation', FLAGS.validation_batch_size)

        num_steps_per_epoch = int(num_training_samples / FLAGS.batch_size)
        decay_steps = int(FLAGS.num_epochs_before_decay * num_steps_per_epoch)

        # Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()

        # Define exponentially decaying learning rate
        learning_rate = tf.train.exponential_decay(
            learning_rate=FLAGS.initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=FLAGS.learning_rate_decay_factor,
            staircase=True)

        # Define the loss functions and get the total loss
        training_pred = model.grasp_net(training_images)
        training_loss = model.custom_loss_function(
            training_pred, training_theta_labels, training_class_labels)
        tf.losses.add_loss(training_loss)
        total_loss = tf.losses.get_total_loss()

        with tf.variable_scope('validation'):
            # Compute loss for validation
            validation_pred = model.grasp_net(validation_images, is_training=False)
            validation_loss_op = model.custom_loss_function(
                validation_pred, validation_theta_labels, validation_class_labels)
            # Compute number of correctness
            num_correctness_op = model.get_num_correctness(
                validation_pred, validation_theta_labels, validation_class_labels)

        # Set optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # Create_train_op to ensure that each time we ask for the loss,
        # the updates are run and the gradients being computed are applied too
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Where model logs are stored
        model_log_dir = os.path.join(FLAGS.log_dir, FLAGS.model_name)
        # Make sure to overwrite the folder
        if os.path.exists(model_log_dir):
            shutil.rmtree(model_log_dir)
        os.mkdir(model_log_dir)

        # Set summary
        tf.summary.image('validation/images', validation_images, max_outputs=FLAGS.batch_size)
        tf.summary.scalar('validation/loss: ', validation_loss_op)
        summary_op = tf.summary.merge_all()

        # Restore pre-trained model
        variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'extra_layers/fc1', 'validation'])
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(session):
            """A Saver function to later restore the model."""
            return saver.restore(session, FLAGS.pretrained_model_path)

        # Define a supervisor for running a managed session
        sv = tf.train.Supervisor(logdir=model_log_dir, summary_op=None, init_fn=restore_fn)

        # Run the managed session
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * FLAGS.num_epochs):
                # At the start of every epoch, show the vital information:
                if step % num_steps_per_epoch == 0:
                    print('------Epoch %d/%d-----' % (step/num_steps_per_epoch + 1, FLAGS.num_epochs))

                start_time = time.time()
                _, global_step_count = sess.run([train_op, global_step])
                time_elapsed = time.time() - start_time

                # Log the summaries every constant steps
                if global_step_count % FLAGS.logging_gap == 0:
                    loss, num_correctness, summaries = sess.run([validation_loss_op, num_correctness_op, summary_op])
                    accuracy = 1.0 * num_correctness / FLAGS.batch_size
                    print('global step %s: loss = %.4f, accuracy = %.4f (%d / %d) with (%.2f sec/step)'
                          % (global_step_count, loss,
                             accuracy,  num_correctness, FLAGS.validation_batch_size, time_elapsed))
                    training_loss = sess.run(total_loss)
                    print('training loss = %.4f' % training_loss)
                    sv.summary_computed(sess, summaries)


if __name__ == '__main__':
    train()
