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
theta_size = 18
label_size = 1

flags = tf.flags

# File paths
flags.DEFINE_string('log_dir', '../log/', 'log directory')
flags.DEFINE_string('model_name', 'model', 'model name')
flags.DEFINE_string('pretrained_model_path', '../pretrained_model/vgg_16.ckpt', 'pretrained mode path')

# Training parameters
flags.DEFINE_integer('batch_size', 50, 'batch size')
flags.DEFINE_integer('num_epochs', 30, 'number of epochs')
flags.DEFINE_integer('logging_gap', 50, 'logging gap')
flags.DEFINE_integer('num_epochs_before_decay', 10, 'number of epochs before decay')
flags.DEFINE_float('initial_learning_rate', 0.0001, 'initial learning rate')
flags.DEFINE_float('learning_rate_decay_factor', 0.7, 'learning rate decay factor')

FLAGS = flags.FLAGS


def train():
    """Main training function to set out training."""

    with tf.Graph().as_default():
        # Testing constants
        data_dir = '/home/shixun7/TFRecord/'
        images, class_labels, theta_labels = read_TFRecord.get_dataset(data_dir, 'Train_positive', 2, 2)
        print('images', images)
        print('class_labels', class_labels)
        print('theta_labels', theta_labels)
        tf.summary.image('images', images)

        # Create the model
        predictions = model.grasp_net(images)

        # Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()

        num_samples = 16000
        num_steps_per_epoch = int(num_samples / FLAGS.batch_size)
        decay_steps = int(FLAGS.num_epochs_before_decay * num_steps_per_epoch)

        # Define exponentially decaying learning rate
        learning_rate = tf.train.exponential_decay(
            learning_rate=FLAGS.initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=FLAGS.learning_rate_decay_factor,
            staircase=True)

        # Define the loss functions and get the total loss
        loss = model.custom_loss_function(predictions, theta_labels, class_labels)
        tf.losses.add_loss(loss)
        total_loss = tf.losses.get_total_loss()

        # Set optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # create_train_op ensures that each time we ask for the loss, the update_ops
        # are run and the gradients being computed are applied too
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Where model logs are stored
        model_log_dir = os.path.join(FLAGS.log_dir, FLAGS.model_name)
        # Make sure to overwrite the folder
        if os.path.exists(model_log_dir):
            shutil.rmtree(model_log_dir)
        os.mkdir(model_log_dir)

        # Set summary
        tf.summary.scalar('prediction: ', tf.reduce_mean(predictions))
        tf.summary.scalar('loss: ', total_loss)
        summary_op = tf.summary.merge_all()

        # Restore pre-trained model
        variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'extra_layers/fc1'])
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
                total_loss, global_step_count = sess.run([train_op, global_step])
                time_elapsed = time.time() - start_time

                # Log the summaries every constant steps
                if global_step_count % FLAGS.logging_gap == 0:
                    print('global step %s: loss: %.4f (%.2f sec/step)'
                          % (global_step_count, total_loss, time_elapsed))
                    summaries = sess.run(summary_op)
                    sv.summary_computed(sess, summaries)


if __name__ == '__main__':
    train()
