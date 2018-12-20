import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


STEPS = 16000
BATCH_SIZE = 100

DATA_DIR = "../../../data/mnist"
mnist = input_data.read_data_sets(DATA_DIR, one_hot=False)

def generator(z):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        g1 = tf.layers.dense(z, 128, activation=tf.nn.relu)
        g_prob = tf.layers.dense(g1, 784, activation=tf.nn.sigmoid)
        g_prob = g_prob + tf.constant(1e-5)
        return g_prob

def discriminator(X):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        d1 = tf.layers.dense(X, 128, activation=tf.nn.relu)
        d_logit = tf.layers.dense(d1, 1)
        d_prob = tf.nn.sigmoid(d_logit)
        d_prob = d_prob + tf.constant(1e-5)
        return d_prob

def sample_z(m, n):
    return np.random.uniform(-1, 1, size=[m, n])
        
if __name__ == "__main__":
    z_dim = 100
    z = tf.placeholder(tf.float32, [None, z_dim], name="z")
    X = tf.placeholder(tf.float32, [None, 784], name="X")
            
    G_sample = generator(z)

    D_real = discriminator(X)
    D_fake = discriminator(G_sample)

    with tf.name_scope("loss"):
        D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1.0 - D_fake))
        G_loss = -tf.reduce_mean(tf.log(D_fake))

    vars = tf.trainable_variables()
    G_vars = [v for v in vars if "generator" in v.name]
    D_vars = [v for v in vars if "discriminator" in v.name]

    G_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(G_loss, var_list=G_vars)
    D_solver = tf.train.AdamOptimizer(learning_rate=0.001).minimize(D_loss, var_list=D_vars)

    # summarize the training process
    tf.summary.scalar("D_loss", D_loss)
    tf.summary.scalar("G_loss", G_loss) 
    merged = tf.summary.merge_all()
    train_ops = [merged, D_solver, D_loss, G_solver, G_loss]
    
    # train
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("summary/train", sess.graph)
        
        sess.run(tf.global_variables_initializer())
        for step in range(STEPS):
            batch_xs, _ = mnist.train.next_batch(BATCH_SIZE)
            res = sess.run(train_ops, feed_dict={X: batch_xs, z: sample_z(BATCH_SIZE, z_dim)})
            
            summary = res[0]
            train_writer.add_summary(summary, step)
            D_loss_current = res[2]
            G_loss_current = res[4]
            
            if step%500 == 0:
                print("step: {}, D_loss_current: {}".format(step, D_loss_current)) 
                print("step: {}, G_loss_current: {}".format(step, G_loss_current)) 
        
        # generate and save the random images
        zs = sample_z(25, z_dim)
        imgs = sess.run(G_sample,feed_dict={z: zs})
        print("Random image max: {}, mins: {}".format(imgs.max(), imgs.min())) # Nan check
        imgs = imgs.reshape([25, 28, 28, 1])
        save_image = tf.summary.image("Generated", imgs, max_outputs=25)
        image_summary = sess.run(save_image)
        train_writer.add_summary(image_summary)


        import matplotlib.pyplot as plt
        imgs = imgs.reshape([25, 28, 28])
        plt.figure("Generated images")
        for i in range(25):
            plt.subplot(5,5,i+1)
            
            plt.imshow(imgs[i], cmap="gray")
            plt.axis("off")
        plt.show()
        train_writer.close()
