"""
@author:Fcant
@description：RandomShuffleQueue
@date: 2019-02-02/0002 下午 22:37
"""

import tensorflow as tf

q = tf.RandomShuffleQueue(capacity=10, min_after_dequeue=2, dtypes="float")

sess = tf.Session()
for i in range(0, 10): #10次入队
    sess.run(q.enqueue(i))

for i in range(0, 8): #8次出队
    print(sess.run(q.dequeue()))

run_options = tf.RunOptions(timeout_in_ms=10000) #等待10秒
try:
    sess.run(q.dequeue(), options=run_options)
except tf.errors.DeadlineExceededError:
    print('out of range')