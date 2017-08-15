# -*- coding: utf-8 -*-
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))

x1 = tf.placeholder(tf.float32, shape=(1,2), name="input")
a1 = tf.matmul(x1, w1)
y1 = tf.matmul(a1, w2)

sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)

# 下一行报错
# print(sess.run(y))

print(sess.run(y1, feed_dict={x1: [[0.7, 0.9]]}))



x2 = tf.placeholder(tf.float32, shape=(3,2), name="input")
a2 = tf.matmul(x2, w1)
y2 = tf.matmul(a2, w2)

sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)

print(sess.run(y2, feed_dict={x2: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))