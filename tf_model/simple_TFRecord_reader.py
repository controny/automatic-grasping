# coding=utf-8
import tensorflow as tf
import os
import matplotlib.pyplot as plt

flags = tf.app.flags
flags.DEFINE_string('source_dir', "/home/shixun7/TFRecord/", 'String: Your TFRecord directory')
FLAGS = flags.FLAGS


def show_image(dataset_dir, set):
    dataset_dir_list = [os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir)
                        if filename.startswith(set)]
    print('dataset dir list', dataset_dir_list)
    num_samples = 0
    for tfrecord_file in dataset_dir_list:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

            # read test
            example = tf.train.Example()
            example.ParseFromString(record)
            img_string = example.features.feature['image/encoded'].bytes_list.value[0]
            with tf.Session() as sess:
                image_placeholder = tf.placeholder(dtype=tf.string)
                decoded_img = tf.image.decode_jpeg(image_placeholder, channels=3)
                reconstructed_img = sess.run(decoded_img, feed_dict={image_placeholder: img_string})
                plt.imshow(reconstructed_img)
                plt.show()


if __name__ == '__main__':
    show_image(FLAGS.source_dir, 'Train_positive')
