import tensorflow as tf
import numpy as np
import pickle

w1 = tf.get_variable("w1", shape=[4, 4])
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./checkpoint/w1.cpkt")
    print("w1: {}".format(sess.run(w1)))

    with open("./checkpoint/w1.pkl", "rb") as file:
        np_w1 = pickle.load(file)

        # raise asssert if the they are not equal
        np.testing.assert_almost_equal(np_w1, sess.run(w1), 
            decimal=7, err_msg="Pickle file is not equal to the checkpoint!!")
