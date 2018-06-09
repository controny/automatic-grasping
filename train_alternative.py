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
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('validation_batch_size', 500, 'batch size for validation')
flags.DEFINE_integer('num_epochs', 10, 'number of epochs')
flags.DEFINE_integer('max_steps', 40000, 'number of max training steps')
flags.DEFINE_integer('logging_gap', 50, 'logging gap')
flags.DEFINE_integer('decay_steps', 500, 'number of steps before decay')
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate')
flags.DEFINE_float('learning_rate_decay_factor', 0.95, 'learning rate decay factor')
flags.DEFINE_float('lmbda', 0.005, 'lambda parameter for regularization')

FLAGS = flags.FLAGS


def train():
    """Main training function to set out training."""

    with tf.Graph().as_default():
        training_images, training_class_labels, training_theta_labels, num_training_samples =\
            read_TFRecord.get_batch_data('Train', FLAGS.batch_size)
        validation_images, validation_class_labels, validation_theta_labels, num_validation_samples = \
            read_TFRecord.get_batch_data('Validation', FLAGS.validation_batch_size)
        #training_theta_labels=tf.Print(training_theta_labels,[training_theta_labels, training_theta_labels.shape],message='training_theta_labels:',summarize=18)
        
        num_steps_per_epoch = int(num_training_samples / (FLAGS.batch_size / 2))
        
        # Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()

        # Define exponentially decaying learning rate
        learning_rate = tf.train.exponential_decay(
            learning_rate=FLAGS.learning_rate,
            global_step=global_step,
            decay_steps=FLAGS.decay_steps,
            decay_rate=FLAGS.learning_rate_decay_factor,
            staircase=True)


        # Define the loss functions and get the total loss
        training_pred = model.grasp_net(training_images, lmbda=FLAGS.lmbda)
        training_loss, training_accuracy = model.create_loss(
            training_pred, training_theta_labels, training_class_labels)
        # training_loss = model.custom_loss_function(training_pred, training_theta_labels, training_class_labels)
        # num_correctness = model.get_num_correctness(training_pred, training_theta_labels, training_class_labels)
        # training_accuracy = 1.0 * tf.cast(num_correctness, tf.float32) / FLAGS.batch_size
        tf.losses.add_loss(training_loss)
        total_loss = tf.losses.get_total_loss()


        # Compute loss for validation
        validation_pred = model.grasp_net(validation_images, is_training=False)
        # Compute number of correctness
        validation_loss, validation_accuracy = model.create_loss(
            validation_pred, validation_theta_labels, validation_class_labels)
        # validation_loss = model.custom_loss_function(validation_pred, validation_theta_labels, validation_class_labels)
        # num_correctness = model.get_num_correctness(validation_pred, validation_theta_labels, validation_class_labels)
        # validation_accuracy = 1.0 * tf.cast(num_correctness, tf.float32) / FLAGS.validation_batch_size

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
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.image('validation/images', validation_images, max_outputs=FLAGS.validation_batch_size)
        tf.summary.scalar('training/loss: ', total_loss)
        tf.summary.scalar('training/accuracy: ', training_accuracy)
        tf.summary.scalar('validation/loss: ', validation_loss)
        tf.summary.scalar('validation/accuracy: ', validation_accuracy)
        summary_op = tf.summary.merge_all()

        # Restore VGG16 pre-trained model
        variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8'])

        # print(*variables_to_restore, sep='\n')
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(session):
            """A Saver function to later restore the model."""
            # Not to restore
            return None
            # Restore pre-trained model
            return saver.restore(session, FLAGS.pretrained_model_path)

        # Define a supervisor for running a managed session
        sv = tf.train.Supervisor(logdir=model_log_dir, summary_op=None, init_fn=restore_fn)

        # Run the managed session
        with sv.managed_session() as sess:
            print('------Max steps: %d-----' % (min(FLAGS.max_steps, num_steps_per_epoch * FLAGS.num_epochs)))
            for step in range(min(FLAGS.max_steps, num_steps_per_epoch * FLAGS.num_epochs)):
                # At the start of every epoch, show the vital information:
                if step % num_steps_per_epoch == 0:
                    print('------Epoch %d/%d-----' % (step/num_steps_per_epoch + 1, FLAGS.num_epochs))

                start_time = time.time()
                _, global_step_count, loss, accuracy = sess.run([train_op, global_step, total_loss, training_accuracy])
                print('global step %s: training loss = %.4f, accuracy = %.4f' % (global_step_count, loss, accuracy))
                time_elapsed = time.time() - start_time

                # Log the summaries every constant steps
                if (global_step_count + 1) % FLAGS.logging_gap == 0:
                    loss, accuracy, summaries = sess.run([validation_loss, validation_accuracy, summary_op])
                    print('global step %s: validation loss = %.4f, accuracy = %.4f with (%.2f sec/step)' % (global_step_count, loss, accuracy, time_elapsed))
                    sv.summary_computed(sess, summaries)


if __name__ == '__main__':
    train()
