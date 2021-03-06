"""
@author:Fcant
"""

import tensorflow as tf

"""
Tensorflow中的激活函数
"""
tf.nn.relu()
tf.nn.sigmoid()
tf.nn.tanh()
tf.nn.elu()
tf.nn.bias_add()
tf.nn.crelu()
tf.nn.relu6()
tf.nn.softplus()
tf.nn.softsign()
tf.nn.dropout()  # 防止过拟合，用来舍弃某些神经元

#----------------------------sigmoid函数-----------------------------------
"""
sigmoid函数-传统神经网络最常用的激活函数之一（另一个是tanh）
    优点：输出映射在（0,1）内，单调连续，非常适合作输出层，求导容易。
    缺点：因为软饱和性，一旦落入饱和区，f·(x)就会变得接近于0，很容易产生梯度消失。
    软饱和性（软饱和是指激活函数h(x)在取值趋于无穷大时，它的一阶导数趋于0。
    硬饱和是指当|x|>c时，其中c为常数，f·(x)=0。relu是一类左侧硬饱和和激活函数
    梯度消失：指在更新模型参数时采用链式求导法则反向求导，越往前梯度越小，最终结果是到达一定深度后
  梯度对模型的更新就没有任何贡献了。
"""
#使用示例
a = tf.constant([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
sess = tf.Session()
print(sess.run(tf.sigmoid(a)))

#----------------------------tanh-函数-----------------------------------
"""
tanh-函数
tanh函数也具有软饱和性，因为他的输出以0为中心，收敛速度比sigmoid要快，但是仍然无法解决梯度消失的问题。
"""

#----------------------------relu函数-----------------------------------
"""
relu函数-目前最受欢迎的激活函数，softplus可以看作是relu的平滑版本
    relu定义为f(x)=max(x,0)
    softplus定义为f(x)=log(1+exp(x))
    relu在x<0时硬饱和，由于x>0时导数为1，所以rule能够在x>0时保持梯度不衰减，从而缓解梯度消失的问题，
    还能够快速地收敛，并提供了神经网络的稀疏表达能力。但是，随着训练的进行，部分输入会落到硬饱和区，导致
    对应的权重无法更新，称为“神经元死亡”
"""
#使用示例
a = tf.constant([-1.0, 2.0])
with tf.Session() as sess:
    b = tf.nn.relu(a)
    print(sess.run(b))
"""
除了rule本身外，Tensorflow还定义了relu6，也就是定义在min(max(features, ), 6)的tf.nn.relu6(features
, name=None),以及crelu，也就是tf.nn.crelu(features, name=None)。
"""

#----------------------------dropout函数-----------------------------------
"""
dropout函数
    一个神经元将以概率keep_prob决定是否被抑制，如果被抑制，该神经元的输出就为0；如果不被抑制，那么该神经元的
输出值将被放大到原来的1/keep_drop倍
    在默认情况下，每个神经元是否被抑制是相互独立的。但是否被抑制也可以通过noise_shape来调节。当noise_shape
    [i]==shape(x)[i]时，x中的元素是相互独立的。如果shape(x)=[k,l,m,n],x中的维度的顺序分别为批、行、列和
    通道，如果noise_shape=[k,l,l,n]，那么每个批和通道都是相互独立的，但是每行每列的数据都是关联的，也就是说，
    要不都为0，要不都还是原来的值。
"""
#使用示例
a = tf.constant([[-1.0, 2.0, 3.0, 4.0]])
with tf.Session() as sess:
    b = tf.nn.dropout(a, 0.5, noise_shape=[1, 4])
    print(sess.run(b))
    b = tf.nn.dropout(a, 0.5, noise_shape=[1, 1])
    print(sess.run(b))

#----------------------------激活函数的选择-----------------------------------
"""
激活函数的选择，当输入特征相差明显时，用tanh的效果会很好，且在循环过程中会不断扩大特征效果并显示出来。
当特征相差不明显时，sigmoid效果比较好，同时，用sigmoid和tanh作为激活函数时，需要对输入进行规范化，
否则激活后的值全部进入平坦区，隐层的输出会全部趋同，丧失原有的特征表达。而relu会好很多，有时候不需要
输入规范化来避免上述情况。
因此，现在大部分的卷积神经网络都采用relu作为激活函数。在自然语言处理上，大概有85%-90%的神经网络会采用
relu，10%-15%的神经网络会采用tanh
"""