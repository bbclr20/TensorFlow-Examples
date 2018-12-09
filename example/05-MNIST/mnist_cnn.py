import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../../data/mnist/", "Input data dir")
flags.DEFINE_string("model_dir", "./checkpoint/", "Diretory to save the model")
flags.DEFINE_string("model_file", "mnist_cnn.cpkt", "The filename of the model")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")
flags.DEFINE_float("keep_prob", 0.8, "Probability to keep the variable")
flags.DEFINE_integer("batch_size", 50, "Training batch size")
flags.DEFINE_integer("steps", 5000, "Training steps")
FLAGS = flags.FLAGS

def main(_):
    #
    # define the graph
    #
    with tf.name_scope("Input"):
        x = tf.placeholder(shape=[None, 784], dtype=tf.float32, name="input")
        y_ = tf.placeholder(shape=[None, 10], dtype=tf.float32, name="label")
        x_image = tf.reshape(x, [-1, 28, 28 ,1]) # N h w c
    
    with tf.name_scope("First_convolution_layer"):
        filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.01))
        bias1 = tf.Variable(tf.constant(0.1, shape=[32]))
        conv1 = tf.nn.conv2d(x_image, filter1, strides=[1, 1, 1, 1], padding="SAME")
        h_conv1 = tf.nn.relu(conv1 + bias1)
        max_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # max_pool1 output shape = [-1, 14, 14, 32]

    with tf.name_scope("Second_convolution_layer"):
        filter2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
        conv2 = tf.nn.conv2d(max_pool1, filter2, strides=[1, 1, 1, 1], padding="SAME")
        h_conv2 = tf.nn.relu(conv2 + bias2)
        max_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # max_pool2 output shape = [-1, 7, 7, 64]

    with tf.name_scope("Fully_connected_layer"):
        flat = tf.reshape(max_pool2, [-1, 7*7*64])
        keep_prob = tf.placeholder(tf.float32)
        full1_drop = tf.nn.dropout(flat, keep_prob=keep_prob)
        
        w_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
        b_fc1 = tf.Variable(tf.truncated_normal([1024]))
        h_fc1 = tf.nn.relu(tf.matmul(full1_drop, w_fc1) + b_fc1)

        w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
        b_fc2 = tf.Variable(tf.truncated_normal([10]))

        
        y = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)    

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
        tf.summary.scalar("cross_entropy", cross_entropy)
    
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_op = optimizer.minimize(cross_entropy)

    with tf.name_scope("Accuracy"):
        correct_mask = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    saver = tf.train.Saver()
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    #
    # try to load previous model and training
    #
    with tf.Session() as sess:
        model = os.path.join(FLAGS.model_dir, FLAGS.model_file)
        train_writer = tf.summary.FileWriter("summary/train/", sess.graph)
        validation_writer = tf.summary.FileWriter("summary/validation/")
        
        if not os.path.isdir(FLAGS.model_dir):
            os.mkdir(FLAGS.model_dir)

        if tf.train.checkpoint_exists(model):
            print("Loading pre-trained model...")
            saver.restore(sess, model)
        else:
            print("Initializing global variables")
            sess.run(tf.global_variables_initializer())
        
        merged = tf.summary.merge_all()
        for step in range(1, FLAGS.steps):
            batch_data, batch_label = mnist.train.next_batch(FLAGS.batch_size)
            summary, _ = sess.run([merged, train_op], 
                feed_dict={x: batch_data, y_: batch_label, keep_prob: FLAGS.keep_prob})    
            train_writer.add_summary(summary, step)
            
            # validation
            if step%10 == 0:
                batch_data, batch_label = mnist.validation.images, mnist.validation.labels
                summary, acc = sess.run([merged, accuracy], 
                    feed_dict={x: batch_data, y_: batch_label, keep_prob:1.0})
                validation_writer.add_summary(summary, step) 
                print("Steps: {}, Accuracy: {:.3f}".format(step, acc))
                if acc >=0.97:
                    break

        # test the training result
        batch_data, batch_label = mnist.test.images, mnist.test.labels
        acc = sess.run(accuracy, 
            feed_dict={x: batch_data, y_: batch_label, keep_prob:1.0})
        print("Test accuracy: {:.3f}".format(acc))
        
        saver.save(sess, model)
        train_writer.close()
        validation_writer.close()

if __name__ == "__main__":
    tf.app.run()