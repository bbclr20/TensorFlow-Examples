import tensorflow as tf
import os

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

num_shards = 2           # num of output files
instances_per_shard = 5  # num of data in a file

data_dir = "./shards"
if tf.gfile.IsDirectory(data_dir):
    tf.gfile.DeleteRecursively(data_dir)
tf.gfile.MakeDirs(data_dir)

for i in range(num_shards):    
    filename = "data.tfrecords-{:05d}-{:05d}".format(i, num_shards) # i'th shard, total shards
    filename = os.path.join(data_dir, filename)
    print(filename)

    with tf.python_io.TFRecordWriter(filename) as writer:
        for j in range(instances_per_shard):
            example = tf.train.Example(
                features = tf.train.Features (
                    feature = {
                        "i": _int64_feature(i),
                        "j": _int64_feature(j),
                    }
                )
            )
            print("{}, {}".format(i, j))
            writer.write(example.SerializeToString())
