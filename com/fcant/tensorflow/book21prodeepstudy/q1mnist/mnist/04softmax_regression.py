"""
@author:Fcant
@description：使用Tensorflow定义了一个Softmax模型，实现了MNIST数据集的分类
@date: 2019-03-29/0029 下午 20:18
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取MNIST数据集，如果数据集不存在则自动下载
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 创建一个占位符（placeholder）x，代表待识别的图片
x = tf.placeholder(tf.float32, [None, 784])

# W是Softmax模型的参数，将一个784维的输入转换为一个10维的输出
# 在Tensorflow中，模型的参数用tf.Variable表示
W = tf.Variable(tf.zeros([784, 10]))

# b是又一个Softmax模型的参数，一般叫做“偏置顶”（bias）
b = tf.Variable(tf.zeros([10]))

# y表示模型的输出
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_是实际的图像标签，同样以占位符表示
y_ = tf.placeholder(tf.float32, [None, 10])

# 得到了两个重要的Tensor：y和y_
# y是模型的输出，y_是实际的图像标签，注意y_是独热表示的
# 下面根据y和y_构造损失

# 根据y和y_构造交叉熵损失
cross_entropy = \
    tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# 有了损失就可以用梯度下降法针对模型的参数（W和b）进行优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 创建一个Session，只有在Session中才能运行优化步骤train_step
sess = tf.InteractiveSession()
# 运行之前必须要初始化所有变量，分配内存
tf.global_variables_initializer().run()

# 进行1000步梯度下降
for _ in range(1000):
    # 在mnist.train中取100个训练数据
    # batch_xs是形状为（100， 784）的图像数据，batch_ys是形如（100， 10）的实际标签
    # batch_xs， batch_ys对应着两个占位符x和y_
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 在Session中运行train_step，运行时要传入占位符的值
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# 正确的预测结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算预测的准确率，他们都是Tensor
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 在Session中运行Tensor可以得到Tensor的值
# 这里是获取最终模型的准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# 0.9109
