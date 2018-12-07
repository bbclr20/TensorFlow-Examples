import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_integer("training_iteration", 1000,
                            "number of training iterations.")
tf.app.flags.DEFINE_integer("model_version", 1, "version number of the model.")
tf.app.flags.DEFINE_string("data_dir", "../../data/mnist/", "Data directory.")
FLAGS = tf.app.flags.FLAGS

def main(_):
    #
    # Train model
    #
    print("Training model...")
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    print("Create session")

    serialized_tf_example = tf.placeholder(tf.string, name="tf_example")
    feature_configs = {"x": tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    
    x = tf.identity(tf_example["x"], name="x")  # use tf.identity() to assign name
    y_ = tf.placeholder("float", shape=[None, 10])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, w) + b, name="y")
    
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for step in range(FLAGS.training_iteration):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        if step % 50 == 0:
            acc = sess.run(accuracy, 
                feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print("training accuracy %g" % acc)
            if acc >= 0.9: break
    print("Done training!!")

    #
    # Export model
    # WARNING(break-tutorial-inline-code): The following code snippet is
    # in-lined in tutorials, please update tutorial documents accordingly
    # whenever code changes.
    #
    values, indices = tf.nn.top_k(y, 10)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(
        tf.constant([str(i) for i in range(10)]))
    prediction_classes = table.lookup(tf.to_int64(indices))

    export_path_base = "./"
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
    if tf.gfile.IsDirectory(export_path):
        tf.gfile.DeleteRecursively(export_path)  
    print("Exporting trained model to", export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    classification_inputs = tf.saved_model.utils.build_tensor_info(
        serialized_tf_example)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
        prediction_classes)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

    classification_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                    classification_inputs
            },
            outputs={
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                    classification_outputs_classes,
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                    classification_outputs_scores
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={"images": tensor_info_x},
            outputs={"scores": tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
    # builder.add_meta_graph_and_variables(
    #     sess, [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #         "predict_images":
    #             prediction_signature,
    #         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #             classification_signature,
    #     },
    #     legacy_init_op=legacy_init_op)

    main_op = tf.group(tf.tables_initializer(), name="main_op")
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "predict_images":
                prediction_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                classification_signature,
        },
        main_op=main_op)

    builder.save()
    sess.close()
    print("Done exporting!")

if __name__ == "__main__":
    tf.app.run()

docker_command = """
sudo docker run -p 8500:8500 \
--mount type=bind,source=$(pwd),target=/models/mnist \
-e MODEL_NAME=mnist -t tensorflow/serving
"""
   