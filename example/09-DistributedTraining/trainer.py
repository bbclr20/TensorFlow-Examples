import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import xavier_initializer

from time import time

flags = tf.app.flags
flags.DEFINE_string("data_dir", "../../data/mnist",
                    "Directory for storing mnist data")

flags.DEFINE_string("train_dir", "./summary",
                    "Directory for storing checkpoint and summary files")

flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization")

flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")

flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")

flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training stePS to perform")

flags.DEFINE_integer("batch_size", 100, "Training batch size")

flags.DEFINE_float("learning_rate", 0.5, "Learning rate")

flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")

flags.DEFINE_string("ps_hosts","localhost:2222",
                    "Comma-separated list of hostname:port pairs")

flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")

flags.DEFINE_string("job_name", None,"job name: worker or PS")

flags.DEFINE_integer("num_gpus", 0,
                     "Number of gpu device")
FLAGS = flags.FLAGS

IMAGE_PIXELS = 28

def main(unused_argv):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index =="":
        raise ValueError("Must specify an explicit `task_index`")

    # define PS, worker and cluster
    is_chief = (FLAGS.task_index == 0)
    PS_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({
        "PS": PS_spec,
        "worker": worker_spec})

    # start PS service
    server = tf.train.Server(
      cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "PS":
        server.join()

    num_workers = len(worker_spec)
    if FLAGS.num_gpus > 0:
        gpu = (FLAGS.task_index % FLAGS.num_gpus)
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
    elif FLAGS.num_gpus == 0:
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
    
    with tf.device(
        tf.train.replica_device_setter(
            worker_device=worker_device,
            ps_device="/job:PS/cpu:0",
            cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)

        with tf.name_scope("Input"):
            x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
            y_ = tf.placeholder(tf.float32, [None, 10])
      
        with tf.name_scope("Fully_connected_layer"):
            w1 = tf.get_variable(shape=[IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units], name="w1",
                initializer=xavier_initializer())
            b1 = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="b1")
            full1 = tf.matmul(x, w1) + b1
            relu1 = tf.nn.relu(full1)
            w2 = tf.get_variable(shape=[FLAGS.hidden_units, 10], name="w2", 
                initializer=xavier_initializer())
            b2 = tf.Variable(tf.zeros([10]), name="b2")
            y = tf.matmul(relu1, w2) + b2
            
        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
            tf.summary.scalar("cross_entropy", cross_entropy)
    
        with tf.name_scope("accuracy"):
            with tf.name_scope("correct_mask"):
                correct_mask = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            with tf.name_scope("correct_mask"):
                accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
                tf.summary.scalar("accuracy", accuracy)

        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)   
            # create a SyncReplicasOptimizer if sync_replicas is True
            if FLAGS.sync_replicas:
                if FLAGS.replicas_to_aggregate is None: 
                    # use all workers by desault 
                    replicas_to_aggregate = num_workers 
                else:
                    replicas_to_aggregate = FLAGS.replicas_to_aggregate
            
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer,
                    replicas_to_aggregate=replicas_to_aggregate,
                    total_num_replicas=num_workers,
                    name="mnist_sync_replicas")
            train_op = optimizer.minimize(cross_entropy, global_step=global_step)
    
        if FLAGS.sync_replicas:
            local_init_op = optimizer.local_step_init_op
            if is_chief:
                local_init_op = optimizer.chief_init_op
            ready_for_local_init_op = optimizer.ready_for_local_init_op

            chief_queue_runner = optimizer.get_chief_queue_runner()
            sync_init_op = optimizer.get_init_tokens_op()
        
        init_op = tf.global_variables_initializer()
        
        if FLAGS.sync_replicas:
            # synchronous training
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=FLAGS.train_dir,
                init_op=init_op,
                local_init_op=local_init_op,
                ready_for_local_init_op=ready_for_local_init_op,
                recovery_wait_secs=1,
                global_step=global_step)
        else:
            # asynchronous training
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=FLAGS.train_dir,
                init_op=init_op,
                recovery_wait_secs=1,
                global_step=global_step)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,  # don't log the device
            device_filters=["/job:PS", "/job:worker/task:%d" % FLAGS.task_index]) 

        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)

        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)
        
        if FLAGS.sync_replicas and is_chief:
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        time_begin = time()
        local_step = 0
        while True:
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            _, step = sess.run([train_op, global_step], feed_dict={x: batch_xs, y_: batch_ys})
            local_step += 1
            
            if local_step%10 == 0:
                batch_data, batch_label = mnist.validation.images, mnist.validation.labels
                acc = sess.run(accuracy, feed_dict={x: batch_data, y_: batch_label})
                print("Worker: {:2d}, training step:{:4d}, global step: {:4d}, accuracy:{:.3f}".
                    format(FLAGS.task_index, local_step, step, acc))
            if step >= FLAGS.train_steps:
                break
        
        time_end = time()
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        val_xent = sess.run(cross_entropy, feed_dict=val_feed)
        print("After %d training step(s), validation cross entropy = %g" %
            (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
    tf.app.run()

shell_command1="""
# create 1 PS and 1 worker

python3.5 trainer.py \
--ps_hosts=localhost:2222 \
--worker_hosts=localhost:2223 \
--job_name=PS \
--task_index=0

python3.5 trainer.py \
--ps_hosts=localhost:2222 \
--worker_hosts=localhost:2223 \
--job_name=worker \
--task_index=0
"""
