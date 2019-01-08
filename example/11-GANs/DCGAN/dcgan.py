import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


sns.set()

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../../../data/mnist", "Input data dir")
flags.DEFINE_string("model_dir", "checkpoint", "Model dir")
flags.DEFINE_string("model_name", "dcgan_mnist.ckpt", "Model name")
flags.DEFINE_integer("steps", 6000, "Training steps")
flags.DEFINE_integer("batch_size", 32, "Training batch size")
flags.DEFINE_boolean("train_model", True, 
                     "False to use previous model and demo vector arithmetic properties")
FLAGS = flags.FLAGS

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)

def generator(z):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        g1 = tf.layers.dense(z, 6 * 6 * 128, activation=tf.nn.relu)
        g1 = tf.reshape(g1, shape=[-1, 6, 6, 128])             # [batch, 6, 6, 128]
        g1 = tf.layers.batch_normalization(g1, momentum=0.8)      
        
        g2 = tf.layers.conv2d_transpose(g1, 64, 4, strides=2,
                                        activation=tf.nn.relu) # [batch, 14, 14, 64]
        g2 = tf.layers.batch_normalization(g2, momentum=0.8)
        
        g3 = tf.layers.conv2d_transpose(g2, 1, 2, strides=2,
                                        activation=tf.nn.sigmoid) # [batch, 28, 28, 1]
        g3 = g3 + tf.constant(1e-5)
        return g3

def discriminator(X):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        X = tf.layers.batch_normalization(X)
        d1 = tf.layers.conv2d(X, 128, 4, strides=2, padding="same",
                              activation=tf.nn.leaky_relu)  # [batch, 14, 14, 128]
        d1 = tf.layers.dropout(d1, 0.8)
        d1 = tf.layers.batch_normalization(d1, momentum=0.8)
        
        d2 = tf.layers.conv2d(d1, 64, 3, strides=2, padding="valid",
                              activation=tf.nn.leaky_relu) # [batch, 6, 6, 64] 
        d2 = tf.layers.dropout(d2, 0.8)
        d2 = tf.layers.batch_normalization(d2, momentum=0.8)
        
        flat = tf.contrib.layers.flatten(d2)
        d_prob = tf.layers.dense(flat, 1, activation=tf.nn.sigmoid)
        d_prob = d_prob + tf.constant(1e-5)
        return d_prob

def sample_z(m, n):
    return np.random.uniform(-1, 1, size=[m, n])
    # return np.random.normal(0, 1, size=[m, n])
        
def load_model_and_init(sess, saver):
    if not os.path.isdir(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)
    
    model_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
       
    if tf.train.checkpoint_exists(model_path):
        print("Loading pre-trained model...")
        saver.restore(sess, model_path)
    else:
        print("Initializing global variables")
        sess.run(tf.global_variables_initializer())

def main(_):
    z_dim = 100
    z = tf.placeholder(tf.float32, [None, z_dim], name="z")
    X = tf.placeholder(tf.float32, [None, 784], name="X")
            
    G_sample = generator(z)

    D_real = discriminator(tf.reshape(X, [-1, 28, 28, 1]))
    D_fake = discriminator(G_sample)

    with tf.name_scope("loss"):
        D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1.0 - D_fake))
        G_loss = -tf.reduce_mean(tf.log(D_fake))

    vars = tf.trainable_variables()
    G_vars = [v for v in vars if "generator" in v.name]
    D_vars = [v for v in vars if "discriminator" in v.name]

    G_solver = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(G_loss, var_list=G_vars)
    D_solver = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(D_loss, var_list=D_vars)

    # summarize the training process
    tf.summary.scalar("D_loss", D_loss)
    tf.summary.scalar("G_loss", G_loss) 
    merged = tf.summary.merge_all()
    train_ops = [merged, D_solver, D_loss, G_solver, G_loss]
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("summary/train", sess.graph)
        load_model_and_init(sess, saver)

        if FLAGS.train_model:
            G_losses = []
            D_losses = []
            for step in range(FLAGS.steps):
                batch_xs, _ = mnist.train.next_batch(FLAGS.batch_size)
                batch_zs = sample_z(FLAGS.batch_size, z_dim)
                res = sess.run(train_ops, feed_dict={X: batch_xs, z: batch_zs})
                
                summary = res[0]
                train_writer.add_summary(summary, step)
                D_loss_current = res[2]
                G_loss_current = res[4]

                G_losses.append(G_loss_current)
                D_losses.append(D_loss_current)

                if step%500 == 0:
                    print("step: {}, D_loss_current: {}".format(step, D_loss_current)) 
                    print("step: {}, G_loss_current: {}".format(step, G_loss_current)) 
            
            # generate and save the random images
            zs = sample_z(25, z_dim)
            imgs = sess.run(G_sample,feed_dict={z: zs})
            print("Random image max: {}, mins: {}".format(imgs.max(), imgs.min())) # Nan check
            save_image = tf.summary.image("Generated", imgs, max_outputs=25)
            image_summary = sess.run(save_image)
            train_writer.add_summary(image_summary)

            # save model
            model_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
            saver.save(sess, model_path)
            train_writer.close()

            # plt show result
            imgs = imgs.reshape([25, 28, 28])
            plt.figure("Generated images")
            for i in range(25):
                plt.subplot(5, 5, i+1)            
                plt.imshow(imgs[i], cmap="gray")
                plt.axis("off")

            plt.figure("Losses")
            plt.plot(G_losses, label="G_losses")
            plt.plot(D_losses, "r", label="D_losses")
            plt.legend()

            plt.show()
        else:
            zs = sample_z(4, z_dim)
            org, a, b, c = np.split(zs, 4)
           
            ratio = 0.25
            for i in range(9):
                plt.figure("Vector arithmetic examples")
                plt.subplot(3, 3, i+1)
                if i == 0:
                    z_feed = org + ratio * a + ratio * b + ratio * c                            
                elif i == 1:
                    z_feed = org - ratio * a + ratio * b + ratio * c            
                elif i == 2:
                    z_feed = org + ratio * a - ratio * b + ratio * c
                elif i == 3:
                    z_feed = org + ratio * a + ratio * b - ratio * c
                elif i == 4:
                    z_feed = org
                elif i == 5:
                    z_feed = org - ratio * a - ratio * b + ratio * c
                elif i == 6:
                    z_feed = org + ratio * a - ratio * b - ratio * c
                elif i == 7:
                    z_feed = org - ratio * a + ratio * b - ratio * c
                elif i == 8:
                    z_feed = org - ratio * a - ratio * b - ratio * c

                img = sess.run(G_sample,feed_dict={z: z_feed})
                img = img.reshape([28, 28])
                plt.imshow(img, cmap="gray")
                plt.axis("off")

            plt.show()

if __name__ == "__main__":
    tf.app.run()
