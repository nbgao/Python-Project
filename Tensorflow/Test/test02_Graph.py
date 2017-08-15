# -*- coding: utf-8 -*-
import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    # 在计算图gl中定义变量"v"，并设置初始值为0
    v = tf.get_variable("v", shape=[1], initializer = tf.zeros_initializer())
    
g2 = tf.Graph()
with g2.as_default():
    # 在计算图gl中定义变量"v"，并设置初始值为1
    v = tf.get_variable("v", shape=[1], initializer = tf.ones_initializer())
    
# 在计算图g1中读取变量"v"的取值
with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse = True):
        # 在计算图g1中，变量"v"的取值应该为0，所以下面这行会输出[0.]
        print(sess.run(tf.get_variable("v")))
        
# 在计算图g2中读取变量"v"的取值
with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        # 在计算图g1中，变量"v"的取值应该为0，所以下面这行会输出[1.]
        print(sess.run(tf.get_variable("v")))
        
        
        
        
''' 加法计算跑在GPU上 '''
g = tf.Graph()
a = tf.constant([1.0, 2.0], name = "a")
b = tf.constant([2.0, 3.0], name = "b")

# 获取当前默认的计算图
print(a.graph is tf.get_default_graph())
# 制定计算运行的设备
with g.device('/gpu:0'):
    result = a + b
    