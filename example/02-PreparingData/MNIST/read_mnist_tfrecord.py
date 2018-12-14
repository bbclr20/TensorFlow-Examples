import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import os

reader = tf.TFRecordReader()

DATA_DIR = "./data"
FILE_PATTERN = "mnist.tfrecords-*-*"
files = glob(os.path.join(DATA_DIR, FILE_PATTERN))
print(files)

filename_queue = tf.train.string_input_producer(files)

_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features = {
        "pixels": tf.FixedLenFeature([], tf.int64),
        "label": tf.FixedLenFeature([], tf.int64),
        "image_raw": tf.FixedLenFeature([], tf.string),
    }
)

pixels = tf.cast(features["pixels"], tf.int32)
label = tf.cast(features["label"], tf.int32)
image = tf.decode_raw(features["image_raw"], tf.float32)
image = tf.reshape(image, [28, 28])

batch_size = 25
capacity = 10 * batch_size
min_after_dequeue = 8 * batch_size

iamge_batch, label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size, 
    capacity=capacity, min_after_dequeue=min_after_dequeue)


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # read a batch
    images, labels = sess.run([iamge_batch, label_batch])

    coord.request_stop()
    coord.join(threads)
    
    for idx, (img, lab) in enumerate(zip(images, labels)):
        plt.figure("MNIST")
        plt.subplot(5, 5, idx+1)
        plt.title(lab)
        plt.imshow(img)
        plt.axis("off")
    plt.show()
    