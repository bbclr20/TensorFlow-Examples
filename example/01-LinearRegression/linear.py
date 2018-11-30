import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

learning_rate = 0.01
max_train_steps = 1000
amplitude = 3
log_step = 50
data_range = 10

#
# generate training data
#
train_X = np.arange(0, data_range, 0.1).reshape(-1, 1)
fluctuation = amplitude * np.random.randn(
                            train_X.shape[0], train_X.shape[1])

a = 5
b = 10
train_Y = a * train_X + b + fluctuation

plt.figure("Linear fit")
plt.plot(train_X, train_Y, "ro")

#
# define graph
#
X = tf.placeholder(tf.float32, [None, 1])
w = tf.Variable(tf.random_normal([1,1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
Y = tf.matmul(X,w) + b

Y_ = tf.placeholder(tf.float32, [None, 1])
loss = tf.losses.mean_squared_error(Y_, Y)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(max_train_steps):
        sess.run(train_op, feed_dict={X: train_X, Y_: train_Y})

        if step%log_step == 0:
            l = sess.run(loss, feed_dict={X: train_X, Y_: train_Y})
            print("Step: {:3d}, loss:{:.3f}".format(step, l))
    weight, bias = sess.run([w, b])
    print("weight: {}, bias: {}".format(weight[0, 0], bias[0]))

plt.plot(train_X, weight * train_X + bias)
plt.show()


#
# solve the problem with least squares regression
# U * transpose(b, w) = train_Y
#
U = np.ones((train_X.shape[0], train_X.shape[1]))
U = np.concatenate((U, train_X), axis=1)
A = np.matmul(U.transpose(), U)

inverse_A = np.linalg.inv(A)
lsr_solution = np.matmul(inverse_A, np.matmul(U.transpose(), train_Y))
print("lsr solution w: {}, b: {}".format(lsr_solution[1,0], lsr_solution[0,0]))
