"""
@author:Fcant
"""

import tensorflow as tf

#计算Wx_plus_b的均值和方差，其中axes=[0]表示想要标准化的维度
fc_mean, fc_var = tf.nn.moments(Wx_plus_b, axes=[0], )
scale = tf.Variable(tf.ones([out_size]))
shift = tf.Variable(tf.zeros([out_size]))
epsilon = 0.001
Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)

#也就是在做：
#Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
#Wx_plus_b = Wx_plus_b * scale + shift