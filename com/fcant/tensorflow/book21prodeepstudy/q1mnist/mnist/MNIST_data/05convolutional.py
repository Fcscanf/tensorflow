"""
@author:Fcant
@description：两层卷积网络分类
@date: 2019-03-29/0029 下午 22:11
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取MNIST数据集，如果数据集不存在则自动下载
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x为训练图像的占位符，y_为训练图像标签的占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 将单张图片从784维向量重新还原为28*28的矩阵图片
# 由于使用的是卷积网络对图像进行分类，所以不能再使用784维的向量表示输入的x
# [-1, 28, 28, 1]中的-1表示形状第一维的大小是根据x自动确定的
x_image = tf.reshape(x, [-1, 28, 28, 1])


def weight_variable(shape):
    """

    :param shape:
    :return: 返回一个给定形状的变量并自动以截断正态分布初始化
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """

    :param shape:
    :return: 返回一个给定形状的变量，初始化时所有值是0.1，可分别用这两个函数创建卷积的核（kernel）与偏置（bias）
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


"""
一个卷积层的标配：
卷积
激活函数
池化
"""
# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 真正进行卷积计算的，卷积计算后选用Relu函数作为激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 调用max_pool_2x2进行一次池化操作
h_pool1 = max_pool_2x2(h_conv1)

# 第二次卷积层
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层，输出为1024维的向量
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 使用Dropout防止神经网络过拟合的一种手段，Keep_prob是一个占位符，训练时为0.5，测试时为1
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 再加入一层全连接，把1024维的向量转换成10维，对应10个类别
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# y_conv相当于Softmax模型中的Logit，当然可以使用Softmax函数将其转换为10个类别的概率，再定义交叉熵损失
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 不采用Softmax再计算交叉熵的方法
# 而是用tf.nn.sigmoid_cross_entropy_with_logits直接计算
cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y_conv))
# 同样定义train_step
train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

# 定义测试的准确率
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练————训练时会在验证集上计算模型的准确度并输出，方便监控训练的进度，也可以据此调整模型的参数
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 训练2000步
for i in range(2000):
    batch = mnist.train.next_batch(50)
    # 每一百步报告在验证集上的准确率
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 训练结束后报告在测试集上的准确率
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# 2019-03-29 23:01:05.112449: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions
# that this TensorFlow binary was not compiled to use: AVX2 step 0, training accuracy 0.16 step 100,
# training accuracy 0.08 step 200, training accuracy 0.08 step 300, training accuracy 0.08 step 400,
# training accuracy 0.06 step 500, training accuracy 0.04 step 600, training accuracy 0.08 step 700,
# training accuracy 0.06 step 800, training accuracy 0.12 step 900, training accuracy 0.1 step 1000,
# training accuracy 0.12 step 1100, training accuracy 0.1 step 1200, training accuracy 0.1 step 1300,
# training accuracy 0.08 step 1400, training accuracy 0.02 step 1500, training accuracy 0.06 step 1600,
# training accuracy 0.06 step 1700, training accuracy 0.02 step 1800, training accuracy 0.14 step 1900,
# training accuracy 0.12 2019-03-29 23:05:38.462469: W tensorflow/core/framework/allocator.cc:124] Allocation of
# 1003520000 exceeds 10% of system memory. test accuracy 0.086
