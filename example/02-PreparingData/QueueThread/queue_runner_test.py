import tensorflow as tf

queue = tf.FIFOQueue(10, tf.float32)
enqueue_op = queue.enqueue([tf.random_normal([1])])

qr = tf.train.QueueRunner(queue, [enqueue_op] * 5) # 5 threads
tf.train.add_queue_runner(qr)

out = queue.dequeue()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(3):
        print(sess.run(out))
    
    coord.request_stop()
    coord.join(threads)
    