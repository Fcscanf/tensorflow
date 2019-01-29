import tensorflow as tf

"""
创建图
"""
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])

"""
创建一个常量运算操作，产生一个1*2矩阵
"""
matrx1 = tf.constant([[3., 3.]])
"""
创建一个常量运算操作，产生一个2*1矩阵
"""
matrx2 = tf.constant([[2.], [2.]])

"""
变量-创建一个变量，初始化标量0
"""
state = tf.Variable(0, name="counter")

"""
创建一个常量张量
"""
#input1 = tf.constant(3.0)
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

"""
创建一个矩阵乘法，把matrix1和matrix2作为输入
"""
c = a * b
"""
返回值product代表矩阵乘法的结果
"""
product = tf.matmul(matrx1, matrx2)
output = tf.multiply(input1, input2)

"""
创建会话
"""
with tf.Session() as sess:
    #指在第二个GPU上运行
    #with tf.device("/gpu:1"):
    print(sess.run(c))
    print(sess.run([product]))
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
    #输出[array([14.], dtype=float32)]
    sess.close()
