"""
@author:Fcant
@description：图的存储与加载
@date: 2019-02-02/0002 下午 22:04
"""
import tensorflow as tf
from tensorflow.python.platform import gfile

v = tf.Variable(0, name='my_varable')
sess = tf.Session()
tf.train.write_graph(sess.graph_def, '/tmp/tfmodel', 'train.pbtext')

with tf.Session() as _sess:
    with gfile.FastGFile("/tmp/tfmodel/train.pbtext", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFormString(f.read())
        _sess.graph.as_default()
        tf.import_graph_def(graph_def, name='tfgraph')