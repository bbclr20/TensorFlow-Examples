import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

batch_size = 500

def sample_data(N):
    return np.random.normal(3, 1.5, N)
    
def random_data(N):
    return np.random.random(N) * 10
    
def generator(z, h_dim):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        g1 = tf.layers.dense(z, h_dim, bias_initializer=tf.random_normal_initializer)
        g_out = tf.layers.dense(g1, batch_size, bias_initializer=tf.random_uniform_initializer)
        return g_out

def discriminator(X, h_dim):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        d1 = tf.layers.dense(X, h_dim, activation=tf.nn.tanh)
        d2 = tf.layers.dense(d1, h_dim, activation=tf.nn.tanh)
        d3 = tf.layers.dense(d2, h_dim, activation=tf.nn.tanh)
        d_prob = tf.nn.sigmoid(d3)
        d_prob = d_prob + tf.constant(1e-5)
        return d_prob

if __name__ == "__main__":
    z = tf.placeholder(tf.float32, [None, 10], name="z")
    X = tf.placeholder(tf.float32, [None, batch_size], name="X")

    G_sample = generator(z, 20)
    D_real = discriminator(X, 60)
    D_fake = discriminator(G_sample, 60)

    with tf.name_scope("loss"):
        D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1.0 - D_fake))
        G_loss = -tf.reduce_mean(tf.log(D_fake))

    vars = tf.trainable_variables()
    G_vars = [v for v in vars if "generator" in v.name]
    D_vars = [v for v in vars if "discriminator" in v.name]

    G_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(G_loss, var_list=G_vars)
    D_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_loss, var_list=D_vars)

    # train
    with tf.Session() as sess:    
        sess.run(tf.global_variables_initializer())
         
        d_losses = []
        g_losses = []
        for i in range(130):
            for _ in range(30):
                x_feed = sample_data(batch_size).reshape([-1, batch_size])
                z_feed = random_data(10).reshape([-1, 10])
                _, loss = sess.run([D_solver, D_loss], feed_dict={X: x_feed, z: z_feed})
                d_losses.append(loss)
            print("step: {}, D_loss_current: {}".format(i, loss)) 
            
            for _ in range(30):
                z_feed = random_data(10).reshape([-1, 10])
                _, loss = sess.run([G_solver, G_loss], feed_dict={z: z_feed})
                g_losses.append(loss)
            print("step: {}, G_loss_current: {}".format(i, loss)) 

        # test
        sns.set()
        x_feed = sample_data(batch_size).reshape([-1, batch_size])
        z_feed = random_data(10).reshape([-1, 10])
        
        z_trans = sess.run(G_sample, feed_dict={z: z_feed})
        z_trans = z_trans.reshape([-1,])
        plt.figure("Fitting result")
        plt.hist(z_trans, 50, density=True, label="GAN")
        xs = x_feed.reshape([-1,])
        plt.hist(xs, 50, density=True, alpha=0.5, label="Numpy")
        plt.legend()

        plt.figure("Loss")
        plt.plot(g_losses, label="g_losses")                    # ln2
        plt.plot(d_losses, "-ro", label="d_losses", mfc='none') # ln4
        plt.xlabel("step")
        plt.legend()
        plt.show()
