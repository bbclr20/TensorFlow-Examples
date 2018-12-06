import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../../../data/mnist/", "Input data dir")
flags.DEFINE_string("model_dir", "./checkpoint/", "Diretory to save the model")
flags.DEFINE_string("model_file", "single_node.cpkt", "The filename of the model")
flags.DEFINE_float("learning_rate", 0.5, "Learning rate")
flags.DEFINE_integer("batch_size", 1000, "Training batch size")
flags.DEFINE_integer("steps", 150, "Training steps")
FLAGS = flags.FLAGS

def main(_):
    data_dir = FLAGS.data_dir
    model_dir = FLAGS.model_dir
    model_file = FLAGS.model_file
    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size
    steps = FLAGS.steps
    
    mnist = input_data.read_data_sets(data_dir, one_hot=True)
    
    # create a model
    with tf.name_scope("input"):
        x = tf.placeholder(shape=[None, 784], dtype=tf.float32, name="x")
        y_label = tf.placeholder(shape=[None, 10], dtype=tf.float32, name="y_label")
    
    with tf.name_scope("softmax_layer"):
        with tf.name_scope("weights"):
            w = tf.get_variable(shape=[784, 10], dtype=tf.float32, name="w", 
                                initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram("weights", w)
        with tf.name_scope("biases"):
            b = tf.Variable(tf.zeros([10]), name="b")
        with tf.name_scope("Wx_plus_b"):
            y_predict = tf.matmul(x, w) + b
    
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_label, logits=y_predict))
        tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_mask"):
            correct_mask = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
        with tf.name_scope("correct_mask"):
            accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

    saver = tf.train.Saver()

    # start training
    with tf.Session() as sess:
        model = os.path.join(model_dir, model_file)
        train_writer = tf.summary.FileWriter("summary/train/", sess.graph)
        validation_writer = tf.summary.FileWriter("summary/validation/")

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        if tf.train.checkpoint_exists(model):
            print("Loading pre-trained model...")
            saver.restore(sess, model)
        else:
            print("Initializing global variables")
            sess.run(tf.global_variables_initializer())
        
        merged = tf.summary.merge_all()
        for i in range(1, steps+1):
            batch_data, batch_label = mnist.train.next_batch(batch_size)
            summary, _, _ = sess.run([merged, accuracy, train_op], feed_dict={x: batch_data, y_label: batch_label})
            train_writer.add_summary(summary, i)
                
            if i%10 == 0:
                batch_data, batch_label = mnist.validation.images, mnist.validation.labels
                summary, acc = sess.run([merged, accuracy], feed_dict={x: batch_data, y_label: batch_label})
                validation_writer.add_summary(summary, i) 
                print("Steps: {}, Accuracy: {:.3f}".format(i, acc))
                if acc >= 0.9:
                    break

        # test the training result
        batch_data, batch_label = mnist.test.images, mnist.test.labels
        acc = sess.run(accuracy, feed_dict={x: batch_data, y_label: batch_label})
        print("Test accuracy: {:.3f}".format(acc))

        saver.save(sess, model)
        train_writer.close()
        validation_writer.close()
        
if __name__ == "__main__":
    tf.app.run()
