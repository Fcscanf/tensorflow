"""
@author:Fcant
卷积函数
"""

import tensorflow as tf;
import numpy as np

# tf.nn.convolution(input, filter, padding, strides=None, dilation_rate=None, name=None,
#                   data_format=None)
# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None,
#              name=None)
# tf.nn.depthwise_conv2d(input, filter, strides, padding, rate=None, name=None, data_format=None)
# tf.nn.separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, rate=None,
#                        name=None, data_format=None)
# tf.nn.atrous_conv2d(value, filters, rate, padding, name=None)
# tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding='SAME',
#                        data_format='NHWC', name=None)
# tf.nn.conv1d(value, filter, stride, padding, use_cudnn_on_gpu=None, data_format=None,
#              name=None)
# tf.nn.conv3d(input, filter, strides, padding, name=None)
# tf.nn.conv3d_transpose(value, filter, output_shape, strides, padding='SAME', name=None)
#
# def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None,
#              name=None)

#输入：
#   input：一个Tensor。数据类型必须是float32或者float64
#   filter：一个Tensor。数据类型必须与input相同
#   strides：一个长度是4的一维整数类型数组，每一维度对应的是input中每一维的对应移动步数。
#   比如：strides[1]对应input[1]的步数
#   padding：一个字符串，取值为SAME或者VALID
#   padding='SAME':仅适用于全尺寸操作，即输入数据维度和输出数据维度相同
#   padding='VALID':适用于部分窗口，即输入数据维度和输出数据维度不同
#   use_cudnn_on_gpu：一个可选布尔值，默认情况下是True
#   name：（可选）为这个操作取一个名字
#输出：一个Tensor，数据类型与input相同

"""
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 2), dtype=np.float32)
y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
tf.shape(y)
"""

"""
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 5), dtype=np.float32)
y = tf.nn.depthwise_conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
tf.shape(y)
"""

# def separable_conv2d(input, depthwise_filter, pointwise_filter, strides, padding, rate=None,
#                      name=None,data_format=None)

#特殊参数
#   depthwise_filter：一个张量。数据维度是四维[filter_height, filter_width, in_channels, channel_multiplier]。
#   其中，in_channls的卷积深度是1
#   pointwise_filter：一个张量。数据维度是四维[1, 1, channel_multiplier * in_channels, out_channels]。
#   其中，pointwise_filter是在depthwise_filter卷积之后的混合卷积

"""
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
depthwise_filter = tf.Variable(np.random.rand(2, 2, 3, 5), dtype=np.float32)
pointwise_filter = tf.Variable(np.random.rand(1, 1, 15, 20), dtype=np.float32)
#   out_channels >= channel_multiplier * inchannels
y = tf.nn.separable_conv2d(input_data, depthwise_filter, pointwise_filter,
                           strides=[1, 1, 1, 1], padding='SAME')
tf.shape(y)
"""

"""
input_data = tf.Variable(np.random.rand(1, 5, 5, 1), dtype=np.float32)
filters = tf.Variable(np.random.rand(3, 3, 1, 1), dtype=np.float32)
y = tf.nn.atrous_conv2d(input_data, filters, 2, padding='SAME')
tf.shape(y)
"""

#def conv2d_transpose(value,filter,output_shape,strides,padding='SAME',data_format='NHMC',name=None):

#特殊参数：
#   output_shape：一维的张量，表示反卷积运算后输出的形状
#输出：和value一样维度的Tensor

"""
x = tf.random_normal(shape=[1, 3, 3, 1])
kernel = tf.random_normal(shape=[2, 2, 3, 1])
y = tf.nn.conv2d_transpose(x, kernel, output_shape=[1, 5, 5, 3], strides=[1, 2, 2, 1], padding="SAME")
"""


