import tensorflow as tf

filename_queue = tf.train.string_input_producer(["FinalExam.tfrecord"], num_epochs=3)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example, 
    features={
        "id": tf.FixedLenFeature([], tf.int64),
        "math": tf.FixedLenFeature([], tf.int64),
        "physics": tf.FixedLenFeature([], tf.int64),
        "chemistry": tf.FixedLenFeature([], tf.int64)
    }
)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        # 2 students 3 epochs
        for i in range(6): 
            example = sess.run(features)
            print(example)
    except Exception as e:
        print(e)
    finally:
        coord.request_stop()
        print("Finish reading")
    coord.join(threads)
