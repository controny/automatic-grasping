import tensorflow as tf
from math import pi
from PIL import Image
import os
import io

slim = tf.contrib.slim
#===================================================  Dataset Utils  ===================================================

def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def read_example_list(path):
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()
    return [line.strip().split(',') for line in lines]

def convert_theta(theta):
    """
    Args:
        theta: a num from -pi/2 to pi/2

    Returns:
        Returns a num from 0 to 17 representing the angle
    """
    if theta > pi/2:
        theta = theta - pi
    if theta < -pi/2:
        theta = theta + pi
    theta = (theta + pi/2) / pi * 180
    diff = [abs(theta - 10*i) for i in range(18)]                             
    return diff.index(min(diff))

def dict_to_tf_example(path, size, label, theta):
    with tf.gfile.GFile(path, 'rb') as fid:
        encoded_jpg = fid.read()

        class_label = 0 
        if label == 'positive':
            class_label = 1
        
        theta_label = convert_theta(float(theta))
        
        return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(b'jpg'),
        'image/class/label': int64_feature(class_label),
        'image/theta/label': int64_feature(theta_label),
    }))       