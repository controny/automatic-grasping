import tensorflow as tf
from math import pi
import os
import sys
import create_TFRecord_utils_v2
import random
import numpy as np

flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('data_dir', "/home/shixun7/data/vrep_dataset/", 'String: Your dataset directory')
#flags.DEFINE_string('data_dir', "..\\..\\simpleData\\", 'String: Your dataset directory')

#Output filename for the naming the TFRecord file
#flags.DEFINE_string('tfrecord_filename', "..\\..\\vrepTFRecord\\", 'String: The output filename to name your TFRecord file')
flags.DEFINE_string('tfrecord_filename', "/home/shixun7/vrepTFRecord_v3/", 'String: The output filename to name your TFRecord file')

#the set of images, including Train, Validation, Test 
flags.DEFINE_string('set', "Train", 'the set of images, including Train, Validation, Test')

#image size
flags.DEFINE_integer('image_size', 224, 'size of image')

#image size
flags.DEFINE_integer('vali_proportion', 2, '0-9, the proportion of Validation in all Data')

FLAGS = flags.FLAGS

sets = ['Train', 'Test', 'Validation']
labels = ['positive','negative']
label_file = 'dataInfo.txt'
folder = 'Images'

def main():

    #==============================================================CHECKS==========================================================================
    #Check if there is a tfrecord_filename entered
   # if not FLAGS.tfrecord_filename:
        #raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    #Check if there is a dataset directory entered
    #if not FLAGS.dataset_dir:
     #   raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

    #==============================================================END OF CHECKS===================================================================

    data_dir = os.path.join(FLAGS.data_dir, FLAGS.set)
    for label in labels:
        writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.tfrecord_filename, FLAGS.set + '_'+label + '.tfrecord'))
        writerVali = tf.python_io.TFRecordWriter(os.path.join(FLAGS.tfrecord_filename, 'Validation' + '_'+label + '.tfrecord'))
        data_info = create_TFRecord_utils_v2.read_example_list(os.path.join(data_dir, label, label_file))
        
        array = np.arange(len(data_info))
        np.random.shuffle(array)
        valiNum = int(float(FLAGS.vali_proportion)/ 10.0 * len(data_info))
        i = 0
        for index in array[:valiNum]:
            image_name, theta = data_info[array[index]]    
            i += 1
            sys.stdout.write('\r>> Converting image %d/%d' % (i, len(data_info)))
            image_path = os.path.join(data_dir, label, folder, image_name)
            tf_example = create_TFRecord_utils_v2.dict_to_tf_example(image_path, FLAGS.image_size, label, float(theta))
            writerVali.write(tf_example.SerializeToString())

        for index in array[valiNum:]:
            image_name, theta = data_info[array[index]]    
            i += 1
            sys.stdout.write('\r>> Converting image %d/%d' % (i, len(data_info)))
            image_path = os.path.join(data_dir, label, folder, image_name)
            tf_example = create_TFRecord_utils_v2.dict_to_tf_example(image_path, FLAGS.image_size, label, float(theta))
            writer.write(tf_example.SerializeToString())
        
        writer.close()
        print('\nFinished converting the %s dataset! label: %s' % (FLAGS.set, label))

if __name__ == "__main__":
    main()
