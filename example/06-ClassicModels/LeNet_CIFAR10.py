import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

classes = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
n_classes = len(classes)

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../../data/cifar10/", "Input data dir")
flags.DEFINE_string("model_dir", "./checkpoint/", "Diretory to save the model")
flags.DEFINE_string("model_file", "mnist_cnn.cpkt", "The filename of the model")
flags.DEFINE_integer("n_fc1", 384, "Size of first fully connected layer")
flags.DEFINE_integer("n_fc2", 192, "Size of second fully connected layer")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")
flags.DEFINE_float("keep_prob", 0.8, "Probability to keep the variable")
flags.DEFINE_integer("batch_size", 50, "Training batch size")
flags.DEFINE_integer("epochs", 50, "Training steps")
FLAGS = flags.FLAGS

def unpickle(filename):
    with open(filename, "rb") as file:
        dict = pickle.load(file, encoding="latin1")
        dict["data"] = (dict["data"].reshape(-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return dict["data"], dict["labels"]

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def onehot(labels):
    n_sample = len(labels)
    n_class = len(classes)
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels

def main(_):
    data1, label1 = unpickle(os.path.join(FLAGS.data_dir, "data_batch_1"))
    data2, label2 = unpickle(os.path.join(FLAGS.data_dir, "data_batch_2"))
    data3, label3 = unpickle(os.path.join(FLAGS.data_dir, "data_batch_3"))
    data4, label4 = unpickle(os.path.join(FLAGS.data_dir, "data_batch_4"))
    data5, label5 = unpickle(os.path.join(FLAGS.data_dir, "data_batch_5"))

    x_train = np.concatenate((data1, data2, data3, data4, data5), axis=0)
    x_train = normalize(x_train)
    y_train = np.concatenate((label1, label2, label3, label4, label5), axis=0)
    y_train = onehot(y_train)

    # image = x_train[11]
    # idx = np.argmax(y_train[11])
    # plt.figure(classes[idx])
    # plt.imshow(image)
    # plt.show()
    
    test_data, test_label = unpickle(os.path.join(FLAGS.data_dir, "test_batch"))
    x_test = test_data
    x_test = normalize(x_test)
    y_test = test_label
    y_test = onehot(y_test)
    
    #
    # define graph
    #
    with tf.name_scope("input"):
        x_image = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        y_ = tf.placeholder(shape=[None, n_classes], dtype=tf.float32)

    weights = {
        "w_conv1": tf.get_variable("w_conv1", shape=[5, 5, 3, 32], 
                                   initializer=tf.contrib.layers.xavier_initializer()),
        "w_conv2": tf.get_variable("w_conv2", shape=[5, 5, 32, 64], 
                                   initializer=tf.contrib.layers.xavier_initializer()),
        "w_fc1": tf.get_variable("w_fc1", shape=[8 * 8 * 64, FLAGS.n_fc1], 
                                 initializer=tf.contrib.layers.xavier_initializer()),
        "w_fc2": tf.get_variable("w_fc2", shape=[FLAGS.n_fc1, FLAGS.n_fc2], 
                                 initializer=tf.contrib.layers.xavier_initializer()),
        "w_fc3": tf.get_variable("w_fc3", shape=[FLAGS.n_fc2, n_classes], 
                                 initializer=tf.contrib.layers.xavier_initializer()),
    }

    biases = {
        "b_conv1": tf.get_variable("b_conv1", shape=[32], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.01)),
        "b_conv2": tf.get_variable("b_conv2", shape=[64], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.01)),
        "b_fc1": tf.get_variable("b_fc1", shape=[FLAGS.n_fc1], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.01)),
        "b_fc2": tf.get_variable("b_fc2", shape=[FLAGS.n_fc2], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.01)),
        "b_fc3": tf.get_variable("b_fc3", shape=[n_classes], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.01)),
    }

    with tf.name_scope("convolution_layer"):
        conv1 = tf.nn.conv2d(x_image, weights["w_conv1"], 
                             strides=[1, 1, 1, 1], padding="SAME") 
        conv1 = conv1 + biases["b_conv1"]
        conv1 = tf.nn.relu(conv1)
        pool1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], 
                               strides=[1, 2, 2, 1], padding="SAME")

        conv2 = tf.nn.conv2d(pool1, weights["w_conv2"], 
                             strides=[1, 1, 1, 1], padding="SAME") 
        conv2 = conv2 + biases["b_conv2"]
        conv2 = tf.nn.relu(conv2)
        pool2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    with tf.name_scope("fully_connected_layer"):
        flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
        keep_prob = tf.placeholder(tf.float32)
        flat_drop = tf.nn.dropout(flat, keep_prob=keep_prob)

        fc1 = tf.matmul(flat_drop, weights["w_fc1"]) + biases["b_fc1"]
        fc2 = tf.matmul(fc1, weights["w_fc2"]) + biases["b_fc2"]
        y = tf.matmul(fc2, weights["w_fc3"]) + biases["b_fc3"]

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
        tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_op = optimizer.minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_mask = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    saver = tf.train.Saver()

    #
    # training
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
        
        total_batch = int(x_train.shape[0] / FLAGS.batch_size)
        merged = tf.summary.merge_all()
        for epoch in range(FLAGS.epochs + 1):
            for batch in range(total_batch):
                batch_x = x_train[batch * FLAGS.batch_size: (batch+1) * FLAGS.batch_size, :] 
                batch_y = y_train[batch * FLAGS.batch_size: (batch+1) * FLAGS.batch_size, :]
                summary, _ = sess.run([merged, train_op], feed_dict={x_image: batch_x, y_: batch_y, keep_prob: 0.8})
                train_writer.add_summary(summary, epoch * total_batch + batch)

                if batch%50==0:
                    idxs = np.random.choice(x_test.shape[0], FLAGS.batch_size, replace=False)
                    batch_x = x_test[idxs]
                    batch_y = y_test[idxs]
                    summary, acc = sess.run([merged, accuracy], feed_dict={x_image: batch_x, y_: batch_y, keep_prob: 1})
                    validation_writer.add_summary(summary, epoch * total_batch + batch)
                    print("Epoch: {:3d}, batch:{:3d}, accuracy:{:.3f}".format(epoch+1, batch, acc))
        
        saver.save(sess, model)
        train_writer.close()
        validation_writer.close()


if __name__ == "__main__":
    tf.app.run()
