# coding=utf-8
import model
import tensorflow as tf
import tensorflow.contrib.slim as slim
from flask import Flask, request
import os
import PIL.Image as Image
import numpy as np

app = Flask(__name__)

flags = tf.flags

flags.DEFINE_string('log_dir', '../log/', 'log directory')
flags.DEFINE_string('model_name', 'model', 'model name')
flags.DEFINE_string('base_model', '', 'base model')

FLAGS = flags.FLAGS

# Define input placeholder and output
x = tf.placeholder(tf.float32, [1, 224, 224, 3])
prediction = model.grasp_net(x, is_training=False, base_model=FLAGS.base_model)

# Where model logs are stored
model_log_dir = os.path.join(FLAGS.log_dir, FLAGS.model_name)
# Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(model_log_dir)
# Get all the variables to restore from the checkpoint file and create the saver function to restore
variables_to_restore = slim.get_variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
saver.restore(sess, checkpoint_file)


@app.route('/inference', methods=['POST'])
def upload_file():
    image = request.files['image']
    image = Image.open(image).resize([224, 224], Image.BILINEAR)
    image = np.expand_dims(np.array(image).astype(np.float), 0)
    # Preprocess image
    image -= [123.68, 116.779, 103.939]

    result = prediction.eval(feed_dict={x: image})
    print('result', result)
    result = np.argmax(result)

    return str(result * 10)


app.run(host='0.0.0.0', port=8080)
sess.close()
