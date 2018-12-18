import tensorflow as tf
from tensorflow.python.client import device_lib

# use environment as constraint
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


def check_available_devices():
    """Dump available devices"""
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == "GPU"]
    gpu_num = len(gpu_names)

    cpu_names = [x.name for x in local_devices if x.device_type == "CPU"]
    cpu_num = len(cpu_names)

    print("{} GPUs are detected: {}".format(gpu_num, gpu_names))
    print("{} CPUs are detected: {}".format(cpu_num, cpu_names))

check_available_devices()


# define tensors and put tensors to diffrent devices 
a = tf.constant([1.1, 2.2, 3.3], name="a")
b = tf.constant([1.1, 2.2, 3.3], name="b")

with tf.device("/gpu:1"):
    c = a + b
    d = tf.Variable(0.0, dtype=tf.float32, name="d")
    # GPU does no support int format but allow_soft_placement will place it on CPU
    e = tf.Variable(0, name="e")

config = tf.ConfigProto(
    log_device_placement=True, # log device 
    allow_soft_placement=True  # automatically choose an existing device
)

with tf.Session(config=config) as sess:
    sess.run(c)
