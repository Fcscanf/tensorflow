"""
@author:Fcant
"""

import tensorflow as tf

"""
通过所给的名字创建或返回一个变量
"""
#v = tf.get_variable(name, shape, dtype, initializer)
"""
为变量指明命名空间
"""
#tf.variable_scope(<scope_name>)
t = tf.add(8, 9)
print(t)   #输出Tensor("Add:0", shape=(), dtype=int32)

# with tf.variable_scope("foo"):
#     v = tf.get_variable("v", [1])
#     v2 = tf.get_variable("v", [1])
# assert v.name == "foo/v:0"

with tf.variable_scope("foo") as scope:
    v = tf.get_variable("v", [1])
with tf.variable_scope("foo", reuse=True):
    #也可以写成
    # scope.reuse_vaiables()
    v1 = tf.get_variable("v", [1])
assert v1 == v

"""
获取变量作用域
"""
with tf.variable_scope("foo") as foo_scope:
    v = tf.get_variable("v", [1])
with tf.variable_scope(foo_scope):
    w = tf.get_variable("w", [1])


with tf.variable("foo") as foo_scope:
    assert foo_scope.name == "foo"
with tf.variable_scope("bar"):
    with tf.variable_scope("baz") as other_scope:
        assert other_scope.name == "bar/baz"
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name == "foo" #保持不变

"""
变量作用域的初始化
"""
with tf.variable_scope("foo", initializer=tf.constant_initializer(0.4)):
    v = tf.get_variable("v", [1])
    assert v.eval() == 0.4 #被作用域初始化
    w = tf.get_variable("w", [1], initializer=tf.constant_initializer(0.3))
    assert w.eval() == 0.3 #重写初始化的值
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.eval() == 0.4 #继承默认的初始化器
    with tf.variable_scope("baz", initializer=tf.constant_initializer(0.2)):
        v = tf.get_variable("v", [1])
        assert v.eval() == 0.2 #重写父作用域的初始化器的值

with tf.variable_scope("foo"):
    x = 1.0 + tf.get_variable("v", [1])
    assert x.op.name == "foo/add"

"""
name_scope操作示例
"""
with tf.variable_scope("foo"):
    with tf.name_scope("bar"):
        v = tf.get_variable("v", [1])
        b = tf.Variable(tf.zeros([1], name='b'))
        x = 1.0 + v
assert v.name == "foo/v:0"
assert b.name == "foo/bar/b:0"
assert x.op.name == "foo/bar/add"