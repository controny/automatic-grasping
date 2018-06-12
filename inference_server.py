# coding=utf-8
import model
import tensorflow as tf
import tensorflow.contrib.slim as slim
from flask import Flask, request
from werkzeug.utils import secure_filename
import os
import PIL.Image as Image
import numpy as np
from io import BytesIO

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
    image = Image.open(image)
    image = np.expand_dims(np.asarray(image), 0)

    result = prediction.eval(feed_dict={x: image})
    result = np.argmax(result)

    return result


app.run(host='0.0.0.0', port=8080)
sess.close()
