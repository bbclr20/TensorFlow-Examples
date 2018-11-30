import tensorflow as tf
import numpy as np

with tf.python_io.TFRecordWriter("FinalExam.tfrecord") as writer:
    for i in range(1,3):
        math = int(100 * np.random.random())
        physics = int(100 * np.random.random())
        chemistry = int(100 * np.random.random())

        example = tf.train.Example(
            features = tf.train.Features(
                feature={
                    "id": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[i])),
                    "math": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[math])),
                    "physics": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[physics])),
                    "chemistry": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[chemistry]))
                }
            )
        )
        writer.write(example.SerializeToString())
