"""
@author:Fcant
@description：读取MNIST数据集并保存为图片
@date: 2019-03-28/0028 下午 22:22
"""

from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
from PIL import Image
import os

# 读取MNIST数据集，如果数据集不存在则自动下载
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 把原始图片保存在MNIST_data/raw/文件夹下。如果没有该文件夹则会自动创建
save_dir = 'MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 保存前20张图片
for i in range(20):
    # mnist.train.images[i, :]就表示第i张图片，序号从0开始
    image_array = mnist.train.images[i, :]
    # Tensorflow中的MNIST图片是一个784维的向量，我们重新把他还原为28*28维的图像
    image_array = image_array.reshape(28, 28)
    # 保存的文件命名格式
    filename = save_dir + 'mnist_train_%d.jpg' % i
    # 将image_array保存为图片
    # 先用scipy.misc.toimage转换为图像,再调用save直接保存
    Image.fromarray(image_array, 'RGB').save(filename)
    # scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

