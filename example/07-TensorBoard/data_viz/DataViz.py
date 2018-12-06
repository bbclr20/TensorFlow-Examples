import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

LOG_DIR = "summary"
TO_EMBED_COUNT = 500
DATA_DIR = "../../../data/mnist"

mnist_sprites = "mnistdigits.png"
mnist_metadata = "metadata.tsv"

path_for_mnist_sprites = os.path.join(LOG_DIR, mnist_sprites)
path_for_mnist_metadata = os.path.join(LOG_DIR, mnist_metadata)

def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))  
    
    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage

def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch, 28*28) to matrix (batch, 28, 28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1 - mnist_digits

if __name__ == "__main__":
    mnist = input_data.read_data_sets(DATA_DIR, one_hot=False)
    batch_xs, batch_ys = mnist.train.next_batch(TO_EMBED_COUNT)

    if tf.gfile.IsDirectory(LOG_DIR):
        print("Delete old dir...")
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)

    # define the config and operation
    embedding_var = tf.Variable(batch_xs, name="Input")
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    embedding.metadata_path = mnist_metadata    # path_for_mnist_metadata
    embedding.sprite.image_path = mnist_sprites # path_for_mnist_sprites
    embedding.sprite.single_image_dim.extend([28,28])

    projector.visualize_embeddings(summary_writer, config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)
        
        graph_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        graph_writer.close()

    to_visualise = batch_xs
    to_visualise = vector_to_matrix_mnist(to_visualise)
    to_visualise = invert_grayscale(to_visualise)

    sprite_image = create_sprite_image(to_visualise)

    plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')

    with open(path_for_mnist_metadata,'w') as file:
        file.write("Index\tLabel\n")
        for index, label in enumerate(batch_ys):
            file.write("%d\t%d\n" % (index, label))
