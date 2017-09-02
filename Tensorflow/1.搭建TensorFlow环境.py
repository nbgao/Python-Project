
# coding: utf-8

# # 1. Create graph

# In[1]:

import tensorflow as tf


# In[2]:

# 创建一个常量op，产生一个1*2矩阵，这个op被作为一个节点加到默认图中
# 构造器的返回值代表该常量op的返回值
matrix1 = tf.constant([[3.,3.]])
# 创建另外一个常量op，产生一个2*1矩阵
matrix2 = tf.constant([[2.], [2.]])


# In[3]:

# 创建一个矩阵乘法 matmul op
product = tf.matmul(matrix1, matrix2)


# # 2. Start graph on session

# In[4]:

# 启动默认图
sess = tf.Session()
# 调用sess的run()方法来执行矩阵乘法op
result = sess.run(product)
print(result)


# In[5]:

with tf.Session() as sess:
    result = sess.run([product])
    print(result)


# In[16]:

with tf.Session() as sess:
    with tf.device("/gpu:1"):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.], [2.]])
        product = tf.matmul(matrix1, matrix2)

product


# # 3. InteractiveSession

# In[6]:

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# 使用初始化器 initializer op 的run()方法初始化'x'
x.initializer.run()

# 增加一个减法 sub op，从'x'减去'a'
sub = tf.subtract(x, a)
print(sub.eval())


# # 4. Variable

# In[7]:

import tensorflow as tf

state = tf.Variable(0, name = "counter")

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后，变量必须先经过初始化(init) op
init_op = tf.global_variables_initializer()

# 启动图，运行op
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state))
    
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


# # 5. Fetch

# In[10]:

import tensorflow as tf

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)


# # 6. Feed

# In[12]:

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))


# # Base on VirtualEnv install TensorFlow

# cd D:/Project/Python/Tensorflow

# sudo apt-get update
# sudo apt-get install python-pip python-dev python-virtualenv

# ## 创建一个全新的virtualenv环境

# virtualenv --system-site-packages ~/tensorflow
# cd ~/tensorflow
# 
# source bin/activate
# pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

# ## 每次使用Tensorflow时，执行以下命令

# cd ~/tensorflow
# source bin/activate

# ## 每当使用完Tensorflow，需要关闭环境时

# deactivate

# In[ ]:



