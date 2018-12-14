import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def _int64_feature (value): 
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

MNIST_DIR = "../../data/mnist"

mnist = input_data.read_data_sets(MNIST_DIR, one_hot=False)
images = mnist.train.images
labels = mnist.train.labels

num_examples, pixels = images.shape
print("MNIST num_examples: {}, pixels: {}".format(num_examples, pixels))

filename = "./mnist.tfrecords"
with tf.python_io.TFRecordWriter(filename) as writer:
    for idx in range(num_examples):
        image_raw = images[idx].tostring() # images[idx].dtype is float32
        example = tf.train.Example(
            features = tf.train.Features (
                feature = {
                    "pixels": _int64_feature(pixels),
                    "label": _int64_feature(labels[idx]),
                    "image_raw": _bytes_feature(image_raw),
                }
            )
        )
        writer.write(example.SerializeToString())
print("Done!!")
