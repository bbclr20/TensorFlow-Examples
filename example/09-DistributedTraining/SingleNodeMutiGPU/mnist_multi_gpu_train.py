from datetime import datetime

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../../../data/mnist/", "Input data dir")
flags.DEFINE_float("learning_rate_base", 0.001, "Learning rate base")
flags.DEFINE_float("learning_rate_decay", 0.99, "Learning rate decay")
flags.DEFINE_float("l2", 1e-4, "L2 regulization rate")
flags.DEFINE_integer("batch_size", 10000, "Training batch size")
flags.DEFINE_integer("total_epoch", 10, "Training batch size")
flags.DEFINE_string("summary_dir", "./summary/", "Diretory to save the summary")
flags.DEFINE_string("model_dir", "./checkpoint/", "Diretory to save the model")
flags.DEFINE_string("model_name", "mnist_cnn.cpkt", "The filename of the model")
FLAGS = flags.FLAGS

def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == "GPU"]
    gpu_num = len(gpu_names)
    print("{} GPUs are detected: {}".format(gpu_num, gpu_names))
    return gpu_num


def model(X, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("conv1", reuse=reuse):
        L1 = tf.layers.conv2d(X, 64, [3, 3], reuse=reuse)
        L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
        L1 = tf.layers.dropout(L1, 0.7, True)

    with tf.variable_scope("conv2", reuse=reuse):
        L2 = tf.layers.conv2d(L1, 128, [3, 3], reuse=reuse)
        L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
        L2 = tf.layers.dropout(L2, 0.7, True)

    with tf.variable_scope("conv3", reuse=reuse):
        L3 = tf.layers.conv2d(L2, 128, [3, 3], reuse=reuse)
        L3 = tf.layers.max_pooling2d(L3, [2, 2], [2, 2])
        L3 = tf.layers.dropout(L3, 0.7, True)

    with tf.variable_scope("dense1", reuse=reuse):
        L4 = tf.contrib.layers.flatten(L3)
        L4 = tf.layers.dense(L4, 1024, activation=tf.nn.relu)
        L4 = tf.layers.dropout(L4, 0.5, True)
       
    with tf.variable_scope("dense2", reuse=reuse):
        L5 = tf.layers.dense(L4, 256, activation=tf.nn.relu)

    with tf.variable_scope("dense3", reuse=reuse):
        LF = tf.layers.dense(L5, 10, activation=None)

    return LF

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
    # need to change learning rates and batch size by number of GPU
    batch_size = FLAGS.batch_size
    learning_rate_base = FLAGS.learning_rate_base
    learning_rate_decay = FLAGS.learning_rate_decay
    decay_step = 600000/batch_size
    l2 = FLAGS.l2
    total_epoch = FLAGS.total_epoch
    data_dir = FLAGS.data_dir
    model_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
    train_summary_path = os.path.join(FLAGS.summary_dir, "train")

    gpu_num = check_available_gpus()
    
    X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    Y = tf.placeholder(shape=[None, 10], dtype=tf.float32)

    X_A = tf.split(X, int(gpu_num))
    Y_A = tf.split(Y, int(gpu_num))

    """
    Multi GPUs Usage
    Results on P40
     * Single GPU computation time: 0:00:22.252533
     * 2 GPU computation time: 0:00:12.632623
     * 4 GPU computation time: 0:00:11.083071
     * 8 GPU computation time: 0:00:11.990167
     
    Need to change batch size and learning rates
         for training more efficiently
    
    Reference: https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf
    """
    accuracies = [] 
    losses = []
    for gpu_id in range(int(gpu_num)):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                cost = tf.nn.softmax_cross_entropy_with_logits_v2(
                                logits=model(X_A[gpu_id]),
                                labels=Y_A[gpu_id])
                losses.append(cost)
                loss = tf.reduce_mean(tf.concat(losses, axis=0), name="loss")
                
                correct_mask = tf.equal(tf.argmax(model(X), 1), tf.argmax(Y, 1))
                correct_count = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
                accuracies.append(correct_count)
                accuracy = tf.reduce_mean(accuracies, axis=0, name="accuracy")
            
    vars = tf.trainable_variables() 
    lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * l2
    loss = loss + lossL2 
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    maintain_averages_op = ema.apply([v for v in vars])

    global_step = tf.get_variable("global_step", [], 
                                initializer=tf.constant_initializer(0), trainable=False)
    
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 
                                            decay_step, learning_rate_decay)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
                            loss, colocate_gradients_with_ops=True, global_step=global_step)  # Important!

    train_op = tf.group(maintain_averages_op, optimizer)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("lossL2", lossL2)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("learning_rate", learning_rate)
    for v in vars:
        tf.summary.histogram(v.name, v)
        if "conv1/conv2d/kernel" in v.name:
            vt = tf.transpose(v, perm=[3,0,1,2]) # N h w c
            tf.summary.image(v.name, vt, max_outputs=32)

    #
    # train
    #
    config=tf.ConfigProto(log_device_placement=False)
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(train_summary_path, sess.graph)
        
        saver = tf.train.Saver()
        load_model_and_init(sess, saver)

        mnist = input_data.read_data_sets(data_dir, one_hot=True)
        total_batch = int(mnist.train.num_examples/batch_size)
        print("total data: {}, total batch: {}, batch_size: {}".format(
              mnist.train.num_examples, total_batch, batch_size))

        merged = tf.summary.merge_all()
        start_time = datetime.now()
        for epoch in range(total_epoch):
            total_cost = 0
            for _ in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                batch_xs = batch_xs.reshape(-1, 28, 28, 1)
                
                summary, _, cost_val, step = sess.run([merged, train_op, loss, global_step],
                                                      feed_dict={X: batch_xs, Y: batch_ys})
                 
                train_writer.add_summary(summary, step)
                total_cost += cost_val

            print("Epoch:{}, cost: {:.4f}".format(epoch, total_cost))
        
        print("--- Training time : {0} seconds /w {1} GPUs ---".format(
              datetime.now() - start_time, gpu_num))

        # check test accuracy
        batch_xs, batch_ys = mnist.test.images, mnist.test.labels
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        acc = sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys})
        print("Mean accuracy: ", acc)

        acc_all = sess.run(accuracies, feed_dict={X: batch_xs, Y: batch_ys})
        print("Accuracy on different devices: {} (these value must be same!)".format(acc_all))

        saver.save(sess, model_path)
        train_writer.close()

if __name__ == "__main__":
    tf.app.run()
