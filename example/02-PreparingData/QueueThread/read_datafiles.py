import tensorflow as tf
import os
from glob import glob

data_dir = "./shards"
data_pattern = os.path.join(data_dir, "data.tfrecords-*-*")

files = glob(data_pattern)

filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features={
        "i":tf.FixedLenFeature([], tf.int64),
        "j":tf.FixedLenFeature([], tf.int64),
    }
)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # defined on in create_datafiles.py
    num_shards = 2
    instances_per_shard = 5

    n_data = num_shards * instances_per_shard
    for i in range(n_data):
        print(sess.run([features["i"], features["j"]]))
    
    coord.request_stop()
    coord.join(threads)
