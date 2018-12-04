import tensorflow as tf
    
flags = tf.app.flags

flags.DEFINE_string("filename", "xxx.tfrecord", "Input filename")
flags.DEFINE_integer("version", 1, "API version")
flags.DEFINE_string("data_dir", "tmp/data/", "Output data dir")

FLAGS = flags.FLAGS

def main(_):
    print("Filename: ", FLAGS.filename)
    print("API version: ", FLAGS.version)
    print("Output data dir: ", FLAGS.data_dir)

if __name__ == "__main__":
    tf.app.run()
