"""
@author:Fcant
"""

import tensorflow as tf
import numpy as np

"""
生成及加载数据
"""
# 构造满足一元二次方程的函数
"""
为了使点更密集一点，构建300个点，分布在1到-1区间，直接采用np生成等差数列的方法，并将结果为300个点的一维数组转换为300*1的二维数组
"""
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
"""
加入一点噪声，使它与x_data的维度一致，并且拟合为均值为0，方差为0,05的正态分布
"""
noise = np.random.normal(0, 0.05, x_data.shape)
"""
y = x^2 - 0.5 + noise
"""
y_data = np.square(x_data) - 0.5 + noise
"""
定义x和y的占位符作为将要输入神经网络的变量
"""
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

#---------------------------------------------------------------------

"""
构建网络模型
构建一个隐藏层和一个输入层，作为神经网络中的层
输入参数应该有4个变量：输入数据、输入数据的维度、输出数据的维度和激活函数；
每一层经过向量化（y=weights*x + biases）的处理，并经过激活函数的非线性化处理后，最终得到输出数据
"""
#定义隐藏层和输出层
def add_layer(inputs, in_size, out_size, activation_funcation=None):
    #构建权重：in_size*out_size大小的矩阵
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    #构建偏置：1*out_size的矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    #矩阵相乘
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_funcation is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_funcation(Wx_plus_b)
    return outputs #得到输出的数据
"""
构建隐藏层，假设隐藏层有20个神经元
"""
h1 = add_layer(xs, 1, 20, activation_funcation=tf.nn.relu)
"""
构建输出层，假设输出层和属入层一样有一个数据源
"""
prediction = add_layer(h1, 20, 1, activation_funcation=None)

"""
构建损失函数：计算输出层的预测值和真实值间的误差，对二者的平方和求再取平均值，得到损失函数。
运用梯度下降法，以0.1的学习速率最小化损失
"""
#计算预测值和真实值之间的误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#---------------------------------------------------------------------

"""
训练模型
"""
#初始化所有变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#训练一千次
for i in range (1000):
    sess.run(train_step, feed_dict={xs : x_data, ys : y_data})
    #每50次打印出一次损失值
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs : x_data, ys : y_data}))