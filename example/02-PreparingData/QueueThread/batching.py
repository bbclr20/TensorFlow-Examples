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

#
# create an example queue which returns batch examples
#

# defined in create_datafiles.py
num_shards = 2
instances_per_shard = 5

batch_size = num_shards
capacity = 3 * batch_size

example, label = features["i"], features["j"]
# example_batch, label_batch = tf.train.batch(
#     [example, label], batch_size=batch_size, capacity=capacity)

min_after_dequeue = 2 # typically, this is a large number
example_batch, label_batch = tf.train.shuffle_batch(
    [example, label], batch_size=batch_size, 
    capacity=capacity, min_after_dequeue=min_after_dequeue)


with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(instances_per_shard):
        # each batch return 5(num_shards) data
        # thus we have 2(instances_per_shard) steps
        eb, lb = sess.run([example_batch, label_batch])
        print("Example batch: {}, label batch: {}".format(eb, lb))
    
    coord.request_stop()
    coord.join(threads)
