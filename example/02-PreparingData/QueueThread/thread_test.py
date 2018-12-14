import tensorflow as tf
import numpy as np
import threading
import time

def foo(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:
            print("Stoping from id: {}".format(worker_id))
            coord.request_stop()
        else:
            print("Working on id: {}".format(worker_id))
        time.sleep(1)

coord = tf.train.Coordinator()
threads = [
    threading.Thread(target=foo, args=(coord, i,)) for i in range(5)
]

for th in threads:
    th.start()

coord.join(threads)
