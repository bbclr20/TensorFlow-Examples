import tensorflow as tf

LABEL_BYTES = 1
IMAGE_SIZE = 32 # width and height
IMAGE_DEPTH = 3 # channels
IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEPTH
NUM_CLASSES = 10

def read_cifar10(data_file, batch_size):
    record_bytes = LABEL_BYTES + IMAGE_BYTES

    data_files = tf.gfile.Glob(data_file)
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)

    # read 1 + 32 * 32 * 3 bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)

    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])

    # slice label and image
    label = tf.cast(tf.slice(record, [0], [LABEL_BYTES], tf.int32))
    depth_major = tf.reshape(tf.slice(record, [LABEL_BYTES], [IMAGE_BYTES]),
                            [IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH])
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    # define the format of example queue
    example_queue = tf.RandomShuffleQueue(
                    capacity = 16 * batch_size,
                    min_after_dequeue = 8* batch_size,
                    dtypes=[tf.float32, tf.int32],
                    shapes=[[IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH], [1]])
    
    num_threads = 16
    example_enqueue_op = example_queue.enqueue([image, label])

    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
                              example_queue, [example_enqueue_op] * num_threads))

    # get training image and labels
    image, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size), [batch_size, 1])
    
    # encode the sparse matrix to one-hot encoding
    labels = tf.sparse_to_dense(
        tf.concat(values=[indices, labels], axis=1), [batch_size, NUM_CLASSES], 1.0, 0.0)

