import tensorflow as tf
from PIL import Image
import os
import io

slim = tf.contrib.slim

image_size = 227
image_channel = 3
batch_size = 64
num_classes = 18
num_readers = 2 #???
num_preprocessing_threads = 2 #???

def get_dataset(dataset_dir,
                set,  
                num_readers,
                num_preprocessing_threads,
                num_epochs = None,
                shuffle=True,
                is_training=True):
    dataset_dir_list = [os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir) if filename.startswith(set)]
    num_samples = 0
    for tfrecord_file in dataset_dir_list:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    print('num_samples', num_samples)

    reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        'image/theta/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(channels = 3),
    'class_label': slim.tfexample_decoder.Tensor('image/class/label'),
    'theta_label': slim.tfexample_decoder.Tensor('image/theta/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(
        data_sources = dataset_dir_list,
        decoder = decoder,
        reader = reader,
        num_samples = num_samples,
        items_to_descriptions=None 
        )

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers = num_readers,
        shuffle = shuffle,
        num_epochs = num_epochs,
        common_queue_capacity = 24 + 3 * batch_size,  # ???
        common_queue_min = 24)


    [image, class_label, theta_label] = provider.get(['image', 'class_label', 'theta_label'])
    #do a simple reshape to batch it up
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [image_size, image_size])
    image = tf.squeeze(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image -= [123.68, 116.779, 103.939]

    image = tf.Print(image, [image.shape], 'image: ')

    if is_training:
        images, class_labels, theta_labels = tf.train.batch([image, class_label, theta_label],
                                                            batch_size = batch_size,
                                                            num_threads = num_preprocessing_threads,
                                                            capacity = 5 * batch_size)
    else:
        images = tf.expand_dims(image,axis=0)
        class_labels = tf.expand_dims(class_label, axis=0)
        theta_labels = tf.expand_dims(theta_label, axis=0)
    return images, class_labels, theta_labels

flags = tf.app.flags
#State your dataset directory
flags.DEFINE_string('source_dir', "/home/shixun7/TFRecord/", 'String: Your TFRecord directory')
#flags.DEFINE_string('source_dir', "output\\", 'String: Your TFRecord directory')
FLAGS = flags.FLAGS

def main():
    with tf.Graph().as_default():
        images_p, class_labels_p, theta_labels_p = get_dataset(dataset_dir = FLAGS.source_dir,
                                                                set = 'Train_positive',
                                                                num_readers = num_readers,
                                                                num_preprocessing_threads = num_preprocessing_threads)
        tf.summary.image('positive images', images_p)
        images_n, class_labels_n, theta_labels_n = get_dataset(dataset_dir = FLAGS.source_dir,
                                                                set = 'Train_negative',
                                                                num_readers = num_readers,
                                                                num_preprocessing_threads = num_preprocessing_threads)
        images = tf.concat([images_p, images_n], axis=0)
        class_labels = tf.concat([class_labels_p, class_labels_n], axis=0)
        theta_labels = tf.concat([theta_labels_p, theta_labels_n], axis=0)

        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            log_dir = './dataset_test/'
            if not tf.gfile.Exists(log_dir):
                tf.gfile.MakeDirs(log_dir)
            writer = tf.summary.FileWriter(log_dir, sess.graph)
            writer.add_graph(sess.graph)

            tf.global_variables_initializer().run()
            summary = sess.run(summary_op)
            writer.add_summary(summary)
        writer.close()

if __name__ == '__main__':
    main()