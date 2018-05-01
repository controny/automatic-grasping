import tensorflow as tf
import tensorflow.contrib.slim as slim
import model

LOG_DIR = '../log/'
PRETRAINED_MODEL_PATH = '../pretrained_model/vgg_16.ckpt'

image_size = 224
channel_num = 3
label_size = 18
batch_size = 100


def add_summary_op(summary_ops, tensor, message):
    op = tf.summary.scalar(message, tensor)
    op = tf.Print(op, [tensor], message)
    summary_ops.append(op)


def train():
    """Main training function to set out training."""

    # Testing constants
    images = tf.ones([batch_size, image_size, image_size, channel_num])
    labels = tf.ones([batch_size, label_size])

    # Create the summary ops such that they also print out to std output:
    summary_ops = []

    # Create the model
    predictions = model.grasp_net(images, summary_ops)
    add_summary_op(summary_ops, tf.reduce_mean(predictions), 'prediction: ')

    # Define the loss functions and get the total loss
    loss = model.custom_loss_function(predictions, labels)
    tf.losses.add_loss(loss)
    total_loss = tf.losses.get_total_loss()
    add_summary_op(summary_ops, total_loss, 'loss')

    # Set optimizer
    learning_rate = 0.5
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # create_train_op ensures that each time we ask for the loss, the update_ops
    # are run and the gradients being computed are applied too
    train_op = tf.train.get_or_create_global_step()

    # Where checkpoints are stored
    if not tf.gfile.Exists(LOG_DIR):
        tf.gfile.MakeDirs(LOG_DIR)

    # Restore pre-trained model
    variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'extra_layers/fc1'])
    # print(*variables_to_restore, sep='\n')
    init_fn = slim.assign_from_checkpoint_fn(PRETRAINED_MODEL_PATH, variables_to_restore)

    # Kick off training
    slim.learning.train(
        train_op,
        LOG_DIR,
        init_fn=init_fn,
        summary_op=tf.summary.merge(summary_ops),
        number_of_steps=10,
        save_summaries_secs=1,
        save_interval_secs=1)


if __name__ == '__main__':
    train()
