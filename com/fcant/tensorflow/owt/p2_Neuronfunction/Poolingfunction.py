"""
@author:Fcant
池化函数
"""

import tensorflow as tf
import numpy as np

"""
tf.nn.avg_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
tf.nn.max_pool_with_argmax(input, ksize, strides, padding, Targmax=None, name=None)
tf.nn.avg_pool3d(input, ksize, strides, padding, name=None)
tf.nn.max_pool3d(input, ksize, strides, padding, name=None)
tf.nn.fractional_avg_pool(value, pooling_ratio, pseudo_random=None, overlapping=None,
                          deterministic=None, seed=None, seed2=None, name=None)
tf.nn.fractional_max_pool(value, pooling_ratio, pseudo_random=None, overlapping=None,
                          deterministic=None, seed=None, seed2=None, name=None)
tf.nn.pool(input, window_shape, pooling_type, padding, dilation_rate=None, strides=None,
           name=None, data_format=None)
"""

# def avg_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
# 输入
#   value：一个四维的张量。数据维度是[batch, height, width, channels]
#   ksize：一个长度不小于4的整型数组。每一位上的值对应于输入数据张量中每一维的窗口对应值
#   strides：一个长度不小于4的整型数组。该参数指定滑动窗口在输入数据张量每一维上的步长
#   data_format：'NHWC'代表输入张量维度的顺序，N为个数，H为高度，W为宽度，C为通道数
#   （RGB三通道或者灰度单通道）
#   name：为这个操作取一个名字
# 输出：一个张量，数据类型和value相同

"""
input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 10), dtype=np.float32)
y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
output = tf.nn.avg_pool(value=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
"""

"""
input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 10), dtype=np.float32)
y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
output = tf.nn.max_pool(value=y, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
"""

"""
input_data = tf.Variable(np.random.rand(10, 6, 6, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 10), dtype=np.float32)
y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
output, argmax = tf.nn.max_pool_with_argmax(input = y, ksize=[1, 2, 2, 1],
                                            strides=[1, 1, 1, 1], padding='SAME')
"""
