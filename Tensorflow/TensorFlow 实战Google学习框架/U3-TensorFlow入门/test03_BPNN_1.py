# -*- coding: utf-8 -*-
import tensorflow as tf

# 声明w1、w2两个变量 通过seed参数设定了随机种子
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 暂时将输入的特征向量定义为一个常量
x = tf.constant([[0.7, 0.9]])

# 通过前向传播算法获得神经网络的输出
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
# 运行初始化过程
sess.run(w1.initializer)
sess.run(w2.initializer)

print(sess.run(y))
sess.close()
