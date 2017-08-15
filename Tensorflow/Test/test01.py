# -*- coding: utf-8 -*-
import tensorflow as tf
a = tf.constant([1.0, 2.0], name = "a")
b = tf.constant([2.0, 3.0], name = "b")
result = a + b
sess = tf.Session()
sess.run(result)

# 获取当前默认的计算图
print(a.graph is tf.get_default_graph())