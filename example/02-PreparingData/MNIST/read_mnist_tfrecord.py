import tensorflow as tf
import matplotlib.pyplot as plt

reader = tf.TFRecordReader()

TFRECORD_FILE = "./mnist.tfrecords"
filename_queue = tf.train.string_input_producer([TFRECORD_FILE])

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
labels = tf.cast(features["label"], tf.int32)
images = tf.decode_raw(features["image_raw"], tf.float32)
images = tf.reshape(images, [-1, 28, 28])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # read a record
    pixel, label, image = sess.run([pixels, labels, images])

    coord.request_stop()
    coord.join(threads)
    
    plt.figure("MNIST")
    plt.title(label)
    plt.imshow(image[0])
    plt.axis("off")
    plt.show()
