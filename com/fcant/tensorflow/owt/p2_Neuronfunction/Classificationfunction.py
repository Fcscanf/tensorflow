"""
@author:Fcant
分类函数
"""

import tensorflow as tf
import numpy as np


"""
tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)
tf.nn.softmax(logits, dim=-1, name=None)
tf.nn.log_softmax(logits, dim=-1, name=None)
tf.nn.softmax_cross_entropy_with_logits(logits, labels, dim=-1, name=None)
tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)
"""

#def sigmoid_cross_entropy_with_logits(logits, targets, name=None)

#输入：logits:[batch_size, num_classes], targets:[batch_size, size].logits用最后一层的输入即可
#最后一层不需要进行sigmoid运算，此函数内部进行了sigmoid操作
#输出：loss [batch_size, num_classes]

#def softmax_cross_entropy_with_logits(logits, labels, dim=-1, name=None)

#输入：logits and labels均为[batch_size, num_classes]
#输出：loss:[batch_size],里面保存的是batch中每个样本的交叉熵

#def sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)

#logits是神经网络最后一层的结果
#输入：logits:[batch_size,num_classes] labels:[batch_size],必须在[0, num_classes]
#输出：loss [batch_size],里面保存的是batch中每个样本的交叉熵