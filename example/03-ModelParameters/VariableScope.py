import tensorflow as tf
import numpy as np

def simple_net(feed_x):
    w = tf.get_variable("w", shape=(4, 4), initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=(4, 1), initializer=tf.constant_initializer(1.0))
    y = w * feed_x + b
    return y

feed_x = np.random.randn(4,1)   
with tf.variable_scope("scope1"):
    y1 = simple_net(feed_x)

# with tf.variable_scope("scope1"): # Error: forget to reuse scope1 explicitly !!
#     y2 = simple_net(feed_x)

with tf.variable_scope("scope1", reuse=True):
    y3 = simple_net(feed_x)

# with tf.variable_scope("scope2", reuse=True): # Error: scope2 does not exist !!
#     y4 = simple_net(feed_x)

with tf.variable_scope("scope2", reuse=tf.AUTO_REUSE):  # reuse if variable already exists
    y5 = simple_net(feed_x)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("y1: {}\n".format(sess.run(y1)))
    print("y3: {}\n".format(sess.run(y3)))
    print("y5: {}\n".format(sess.run(y5)))
