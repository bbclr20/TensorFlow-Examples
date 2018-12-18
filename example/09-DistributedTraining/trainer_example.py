import tensorflow as tf
from tensorflow import flags

flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization.")

flags.DEFINE_string("ps_hosts", None,
                    "Comma-separated list of hostname:port pairs")
                    
flags.DEFINE_string("worker_hosts", None,
                    "Comma-separated list of hostname:port pairs")

flags.DEFINE_string("job_name", None, "job name: worker or PS")
FLAGS = flags.FLAGS

def main(_):
    PS_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({
        "PS": PS_spec,
        "worker": worker_spec})
        
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "PS":
        server.join()
    # 将任务编号为0的worker设置为chief worker
    is_chief = (FLAGS.task_index == 0)
        
if __name__ == "__main__":
    tf.app.run()

shell_command1 = """
# create 1 PS and 1 workers locally

python3.5 trainer_example.py --ps_hosts=localhost:2222 \
--worker_hosts=localhost:2223 \
--job_name=PS \
--task_index=0

python3.5 trainer_example.py --ps_hosts=localhost:2222 \
--worker_hosts=localhost:2223 \
--job_name=worker \
--task_index=0
"""

shell_command2 = """
# create 2 PS and 2 workers locally

python3.5 trainer_example.py \
--ps_hosts=localhost:2222,localhost:2223 \
--worker_hosts=localhost:2224,localhost:2225 \
--job_name=PS \
--task_index=0

python3.5 trainer_example.py \
--ps_hosts=localhost:2222,localhost:2223 \
--worker_hosts=localhost:2224,localhost:2225 \
--job_name=PS \
--task_index=1

python3.5 trainer_example.py \
--ps_hosts=localhost:2222,localhost:2223 \
--worker_hosts=localhost:2224,localhost:2225 \
--job_name=worker \
--task_index=0

python3.5 trainer_example.py \
--ps_hosts=localhost:2222,localhost:2223 \
--worker_hosts=localhost:2224,localhost:2225 \
--job_name=worker \
--task_index=1
"""