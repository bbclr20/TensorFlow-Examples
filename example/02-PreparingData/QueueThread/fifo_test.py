import tensorflow as tf

q = tf.FIFOQueue(2, tf.int32)
init = q.enqueue_many([[0, 10]])

x = q.dequeue()
y = x + 1

q_inc = q.enqueue([y])

with tf.Session() as sess:
    sess.run(init)

    for _ in range(5):
        val, _ = sess.run([y, q_inc])
        print(val)
