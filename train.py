import tensorflow as tf
import tensorflow.contrib.slim as slim
import model
import time

# File paths
log_dir = '../log/'
pretrained_model_path = '../pretrained_model/vgg_16.ckpt'

# Input setting
image_size = 224
channel_num = 3
label_size = 18

# Training parameters
batch_size = 100
training_steps = 100
log_step = 10
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
decay_steps = 50    # Better to be calculate dynamically


def train():
    """Main training function to set out training."""

    with tf.Graph().as_default():
        # Testing constants
        images = tf.ones([batch_size, image_size, image_size, channel_num])
        labels = tf.ones([batch_size, label_size])

        # Create the model
        predictions = model.grasp_net(images)

        # Create the global step for monitoring the learning_rate and training.
        global_step = tf.train.get_or_create_global_step()

        # Define exponentially decaying learning rate
        learning_rate = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        # Define the loss functions and get the total loss
        loss = model.custom_loss_function(predictions, labels)
        tf.losses.add_loss(loss)
        total_loss = tf.losses.get_total_loss()

        # Set optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # create_train_op ensures that each time we ask for the loss, the update_ops
        # are run and the gradients being computed are applied too
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Where checkpoints are stored
        if not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)

        # Set summary
        tf.summary.scalar('prediction: ', tf.reduce_mean(predictions))
        tf.summary.scalar('loss: ', total_loss)
        summary_op = tf.summary.merge_all()

        # Restore pre-trained model
        variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'extra_layers/fc1'])
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(session):
            """A Saver function to later restore the model."""
            return saver.restore(session, pretrained_model_path)

        # Define a supervisor for running a managed session
        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, init_fn=restore_fn)

        # Run the managed session
        with sv.managed_session() as sess:
            for step in range(training_steps):
                start_time = time.time()
                total_loss, global_step_count = sess.run([train_op, global_step])
                time_elapsed = time.time() - start_time
                print('global step %s: loss: %.4f (%.2f sec/step)'
                      % (global_step_count, total_loss, time_elapsed))

                # Log the summaries every constant steps
                if global_step_count % log_step == 0:
                    summaries = sess.run(summary_op)
                    sv.summary_computed(sess, summaries)


if __name__ == '__main__':
    train()
