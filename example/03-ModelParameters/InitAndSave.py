import tensorflow as tf
import pickle
import numpy as np

# define variables in the graph
w1 = tf.Variable(tf.random_normal((4,4), mean=10, stddev=0.35), name="w1")
w2 = tf.get_variable("w2", shape=[4, 4],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.zeros((4,1)), name="b", dtype=tf.float32)

x = tf.placeholder(shape=(4,1), dtype=tf.float32, name="x")
y = tf.matmul(w1, x) + b

# create a saver
saver = tf.train.Saver({"w1": w1})

# create a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # display the value of variables
    print("w1: {}\n".format(sess.run(w1)))
    print("w2: {}\n".format(sess.run(w2)))
    print("b: {}\n".format(sess.run(b)))

    # feed value and run the operation
    feed_x = np.random.randn(4,1)
    print("x: {}\n".format(feed_x))
    print("y: {}\n".format(sess.run(y, feed_dict={x: feed_x})))
    

    # save the data with checkpoint and pickle
    # we will load and compare these data later 
    saver.save(sess, "./checkpoint/w1.cpkt")
    np_w1 = sess.run(w1)
    with open("./checkpoint/w1.pkl", "wb") as file:
        pickle.dump(np_w1, file)