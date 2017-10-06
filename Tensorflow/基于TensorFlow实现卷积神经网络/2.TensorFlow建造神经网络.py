
# coding: utf-8

# # 1. Build nerual network

# ## 1.1 Preparing

# In[1]:

import numpy as np
import tensorflow as tf


# ## 1.2 Train data

# In[2]:

x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise


# ### 定义结点准备接收数据

# In[3]:

xs = tf.placeholder(tf.float32, [None,1])
ys = tf.placeholder(tf.float32, [None,1])


# ### 添加隐藏层和输出层

# In[4]:

def add_layer(inputs, in_size, out_size, activation_function=None):
    # 定义参数 Weights, biases
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 定义拟合公式
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # 激励函数
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# In[5]:

# 添加隐藏层(hidden layer)
# 输入值是xs，隐藏层有10个神经元，使用ReLU激活函数
# 输入1维数据，经隐藏层处理输出10维数据
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)


# In[6]:

# 添加输出层(output layer)
# 输入值是隐藏层l1，预测层输出1个结果
# 从隐藏层输入得到10维数据，经输出层处理输出1维数据
prediction = add_layer(l1, 10, 1, activation_function=None)


# ### loss表达式(预测结果与真实结果的误差)

# In[8]:

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))


# ### 选择optimizer使loss达到最小

# In[9]:

# 最基本的Optimizer: Gradient Descent，学习率=0.1
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)


# ### 初始化定义的变量

# In[93]:

init = tf.global_variables_initializer()


# ### 在Session中启动计算图

# In[94]:

sess = tf.Session()
sess.run(init)


# ### 设置迭代次数

# In[95]:

get_ipython().run_cell_magic('time', '', "\ny = np.zeros(20)\n# 迭代1000次学习，sess。run optimizer\nfor i in range(1000):\n    # training train_step 和 loss 都是由 placeholder 定义的运算\n    # 所以这里要用feed传入参数，placeholder 和 feed_dict 是绑定使用的\n    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})\n    if i%50 == 0:\n        # 观察每一步的变化\n        y[i//50] = sess.run(loss, feed_dict={xs: x_data, ys: y_data})\n        print('%3d:  %f' % (i, sess.run(loss, feed_dict={xs: x_data, ys: y_data})))")


# In[96]:

import matplotlib.pyplot as plt
x = np.arange(20)
plt.plot(x,y)
plt.ylim(0,0.8)
plt.show()


# # 2. Improving nerual network

# ## Dropout

# In[98]:

def add_layer(inputs, in_size, out_size, activation_function=None):
    # 定义参数 Weights, biases
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 定义拟合公式
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # 隐含节点 dropout 率等于0.5的时候效果最好，即 keep_prob=0.5
    # 原因是0.5的时候dropout随机生成的网络结构最多
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=0.5)
    # 激励函数
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
    


# # 3. Visualization (Tensorboard)

# In[103]:

import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this later
    # 区别：大框架，定义层layer，里面有小部件
    with tf.name_scope('layer'):
        # 区别：小部件
        with tf.name_scope('weghts'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


# In[121]:

# 区别：大框架，里面有inputs x, y
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')
    
# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)


# 区别：定义框架 loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    
# 区别：定义框架 train
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
    
''' 
区别：sess.graph 把所有框架加载到一个文件中放到文件夹"logs/"里 
接着打开terminal，进入你存放的文件夹地址上一层，运行命令 tensorboard --logdir='logs/'
会返回一个地址，然后用浏览器打开这个地址，在 graph 标签栏下打开
'''
writer = tf.summary.FileWriter("logs/", sess.graph)
# important step
sess.run(tf.global_variables_initializer())    


# tersorboard --logdir='logs/'

# # 4. Save & Load

# cd D:Project/Python/Tensorflow/
# 
# mkdir my_net

# ## Save variables

# In[18]:

import numpy as np
import tensorflow as tf

# Save to file
# remember to define the same dtype and shaoe when restore
W = tf.Variable([[1,2,3],[3,4,5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 用saver将所有variable保存到定义的路径
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("Save to path:", save_path)


# ## Restore variables

# In[ ]:

# redefine the same shape and same stype for your variables
W = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='biases')

# not need init step
saver = tf.train.Saver()
# 用saver从路径中将save_net.ckpt保存的W和b restore进来
with tf.Session() as sess:
    saver.restore(sess, "/my_net/save_net.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))


# In[ ]:



