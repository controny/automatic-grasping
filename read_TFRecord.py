import tensorflow as tf
from PIL import Image
import os
import io

slim = tf.contrib.slim

def get_dataset(dataset_dir, set):
    dataset_dir_list = [os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir) if filename.startswith(set)]
    
    num_samples = 0
    for tfrecord_file in dataset_dir_list:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
        'image/theta/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }

    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(channels = 3),
    'class_label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
    'theta_label': slim.tfexample_decoder.Tensor('image/theta/label', shape=[]),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(
        data_sources = dataset_dir_list,
        reader = reader,
        decoder = decoder,
        num_samples = num_samples, 
        items_to_descriptions = None)
    
    return dataset, num_samples

def load_batch(dataset, 
               num_epochs = None,
               image_size = 224,
               batch_size = 128,
               num_readers = 2,
               num_threads = 2,
               shuffle = True):
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
                                                              num_readers = num_readers,
                                                              shuffle = shuffle,
                                                              num_epochs = num_epochs,
                                                              common_queue_capacity=20 * batch_size,
                                                              common_queue_min=10 * batch_size
                                                              )

    [image, class_label, theta_label] = provider.get(['image', 'class_label', 'theta_label'])
    #do a simple reshape to batch it up
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [image_size, image_size])
    image = tf.squeeze(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image -= [123.68, 116.779, 103.939]

    images, class_labels, theta_labels = tf.train.batch([image, class_label, theta_label],
                                                        batch_size = batch_size,
                                                        num_threads = num_threads,
                                                        capacity = 5 * batch_size)


    return images, class_labels, theta_labels

def get_batch_data(datasetName = "Train", batch_size = 128, image_size = 224, dataset_dir = "/home/shixun7/vrepTFRecord_v2/", balanced = True):
    """
    get a batch of data from both label class(positive/negative)
    :param datasetName: Train，Validation or Test
    :param balanced: half from positive, another half from negative 
    :param batch_size: number of images in this batch
    :param dataset_dir: the dir of TFRecord dataset
    :return: images, class_labels, theta_labels, Train_n_num_samples
    if not balanced, num_samples is returned
    notice that returned value is Train_n_num_samples, negative set numbers, not doubled!
    """
    num_readers = 2
    num_preprocessing_threads = 2
    shuffle = True
    num_epochs = None
    theta_label_size = 18
    class_label_size = 1

    if balanced == True:
        dataset_p, Train_p_num_samples = get_dataset(dataset_dir = dataset_dir, set = datasetName + '_positive')
        dataset_n,Train_n_num_samples = get_dataset(dataset_dir = dataset_dir, set = datasetName + '_negative')

        images_p, class_labels_p, theta_labels_p = load_batch(dataset = dataset_p, 
                                                              image_size = image_size,
                                                              num_epochs = num_epochs,
                                                              batch_size = int(batch_size / 2),
                                                              num_readers = num_readers,
                                                              num_threads = num_preprocessing_threads,
                                                              shuffle = shuffle)
        images_n, class_labels_n, theta_labels_n = load_batch(dataset = dataset_n, 
                                                              image_size = image_size,
                                                              num_epochs = num_epochs,
                                                              batch_size = int(batch_size / 2),
                                                              num_readers = num_readers,
                                                              num_threads = num_preprocessing_threads,
                                                              shuffle = shuffle)

        images = tf.concat([images_p, images_n], axis=0)
        class_labels = tf.concat([class_labels_p, class_labels_n], axis=0)
        theta_labels = tf.concat([theta_labels_p, theta_labels_n], axis=0)
        # change shape
        class_labels = tf.reshape(class_labels, [-1, class_label_size])
        theta_labels = tf.one_hot(theta_labels, theta_label_size)
        return images, class_labels, theta_labels, Train_n_num_samples
    
    if balanced == False:
        dataset, num_samples = get_dataset(dataset_dir = dataset_dir, set = datasetName)

        images, class_labels, theta_labels = load_batch(dataset = dataset, 
                                                        image_size = image_size, 
                                                        num_epochs = num_epochs,
                                                        batch_size = batch_size,
                                                        num_readers = num_readers,
                                                        num_threads = num_preprocessing_threads,
                                                        shuffle = shuffle)
        # change shape
        class_labels = tf.reshape(class_labels, [-1, class_label_size])
        theta_labels = tf.one_hot(theta_labels, theta_label_size)
        return images, class_labels, theta_labels, num_samples

def  get_one_class_batch_data(datasetName = "Train", classLable = "positive", batch_size = 128, image_size = 224, dataset_dir = "/home/shixun7/vrepTFRecord_v2/"):
    """
    get a batch of data from only a class(positive/negative)
    :param datasetName: Train，Validation or Test
    :param classLable: positive or negative
    :param batch_size: number of images in this batch
    :param dataset_dir: the dir of TFRecord dtaset
    :return:  images, class_labels, theta_labels, num_samples
    """
    num_readers = 2
    num_preprocessing_threads = 2
    shuffle = True
    num_epochs = None
    theta_label_size = 18
    class_label_size = 1
    
    setName = datasetName + '_' + classLable

    dataset, num_samples = get_dataset(dataset_dir = dataset_dir, set = setName)
    images, class_labels, theta_labels = load_batch(dataset = dataset, 
                                                    image_size = image_size,
                                                    num_epochs = num_epochs,
                                                    batch_size = int(batch_size),
                                                    num_readers = num_readers,
                                                    num_threads = num_preprocessing_threads,
                                                    shuffle = shuffle)

    # change shape
    class_labels = tf.reshape(class_labels, [-1, class_label_size])
    theta_labels = tf.one_hot(theta_labels, theta_label_size)
    return images, class_labels, theta_labels, num_samples

flags = tf.app.flags
flags.DEFINE_string('source_dir', "/home/shixun7/vrepTFRecord/", 'String: Your TFRecord directory')
flags.DEFINE_string('set', "Train", 'String: Your TFRecord directory')
FLAGS = flags.FLAGS

def main():
    with tf.Graph().as_default():
        images, class_labels, theta_labels, num_samples = get_batch_data(#dataset_dir = FLAGS.source_dir,
                                                              datasetName = FLAGS.set, 
                                                              batch_size = 128, 
                                                              image_size = 224)
        tf.summary.image('Train images', images, max_outputs=128)
        summary_op = tf.summary.merge_all()
        with tf.Session() as sess:
            log_dir = './dataset_test/'
            if not tf.gfile.Exists(log_dir):
                tf.gfile.MakeDirs(log_dir)
            writer = tf.summary.FileWriter(log_dir, sess.graph)
            writer.add_graph(sess.graph)

            tf.global_variables_initializer().run()
            coord = tf.train.Coordinator() 
            tf.train.start_queue_runners(coord=coord)
            summary = sess.run(summary_op) 
            writer.add_summary(summary)
        writer.close()

if __name__ == '__main__':
    main()