
# coding: utf-8

# # TensorFlow Hello World ！

# In[3]:

import tensorflow as tf
tf.__version__


# In[4]:

hello = tf.constant("Hello, TensorFlow!")

sess = tf.Session()

print(sess.run(hello))


# # Computational Graph

# In[5]:

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)


# In[6]:

print("node1:", node1, "node2:", node2)
print("node3:", node3)


# In[7]:

sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3)", sess.run(node3))


# In[8]:

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b:[2,4]}))


# # Everything is Tensor

# ## Tensors

# In[9]:

[1., 2., 3.]
[[1., 2., 3.], [4., 5., 6.]]
temp = [[[1., 2., 3.]], [[7., 8., 9.]]]
print("temp:", temp)
print("temp[0]:", temp[0])
print("temp[0][0]:", temp[0][0])


# In[10]:

t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(t)


# **———————————————————————————————————————**

# # Linear Regression

# ## Cost Function

# $ cost=\frac{1}{m}\sum_{i=1}^{m}(H(x^{(i)}-y^{(i)}))^2 $

# $ H(x)=Wx+b $

# $cost(W,b)=\frac{1}{m}\sum_{i=1}^{m}(H(x^{(i)}-y^{(i)}))^2$

# ## Simplified Hypothesis in TF

# $H(x)=Wx$  
# y_model = tf.mul(x, W)

# $cost(W)=\frac{1}{m}\sum_{i=1}^{m}(Wx^{(i)}-y^{(i)})^2$
# 
# cost = tf.square(Y - y_model)

# ## Goal: Minimize Cost

# $\min_{W,b} cost(W,b)$

# ## Formal Definition

# $cost(W)=\frac{1}{2m}\sum_{i=1}^{m}(Wx^{(i)}-y^{(i)})^2$

# $W:=W-\alpha \frac{\partial }{\partial W}cost(W) $

# ## Gradient Descent Algorithm in TF

# $W:=W-\alpha \frac{\partial }{\partial W}cost(W) $

# $W:=W-\alpha\frac{1}{m}\sum_{i=1}^{m}(Wx^{(i)}-y^{(i)})x^{(i)} $

# cost = tf.square(Y - y_model) <br>
# train_op = tf.train.GradientDescentOptimizer(0.01.minimize(cost)

# ## 1. Build graph using TF operations

# $ H(x)=Wx+b $

# In[11]:

X_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = X_train * W + b


# $cost(W,b)=\frac{1}{m}\sum_{i=1}^{m}(H(x^{(i)}-y^{(i)}))^2$

# In[12]:

cost = tf.reduce_mean(tf.square(hypothesis - y_train))


# ### GradientDescent

# In[13]:

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


# ## 2. Run/update graph and get results

# In[14]:

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[15]:

for step in range(201):
    sess.run(train)
    if step%20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))


# ### Placeholders

# In[16]:

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# In[17]:

for step in range(201):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X:[1,2,3], Y:[1,2,3]})
    if step%20 == 0:
        print(step, cost_val, W_val, b_val)


# **———————————————————————————————————————**

# # Logistic Regression (Binaray Classification)

# ## Sigmoid Function

# ### $ H(X)=sigmoid(XW)=\frac{1}{1+e^{-XW}}$

# ### $ cost(W)=-\frac{1}{m}\sum{ylog(H(x)) + (1-y)(log(1-H(x)))} $

# ### $W:=W-\alpha \frac{\partial }{\partial W}cost(W) $

# ## Logistic Classification in TF

# $ H(X)=sigmoid(XW)=\frac{1}{1+e^{-XW}}$

# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# $ cost(W)=-\frac{1}{m}\sum{ylog(H(x)) + (1-y)(log(1-H(x)))} $

# cost = -tf.reduce_mean(Y \* tf.log(hypothesis) + (1-Y) \* tf.log(1-hypothesis))

# ### Training Data

# In[18]:

x_data = [[1,2], [2,3], [3,1], [4,3], [5,3], [6,2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


# In[19]:

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


# ### Train the model

# In[20]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step%200 == 0:
            print(step, cost_val)
    
    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


# ## Classifying diabetes

# In[21]:

import numpy as np
xy = np.loadtxt('./Data/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
print(np.shape(xy))
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


# In[22]:

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    feed = {X: x_data, Y: y_data}
    for step in range(10001):
        sess.run(train, feed_dict=feed)
        if step%200 == 0:
            print(step, sess.run(cost, feed_dict=feed))
            
            
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict=feed)
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


# # Softmax Classification

# hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)

# ### Softmax Function

# ### $S(y_i)=\frac{e^{y_{i}}}{\sum_{j}{e^{y_{j}}}}$

# ### Cost Function: Cross entropy

# ### $L = \frac{1}{N}\sum_i{D(S(WX_i+b), L_i)}$

# cost = tf.reduce_mean(-tf.reduce_sum(Y \* tf.log(hypothesis), axis=1)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# In[50]:

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step%200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
    
    # Test & one-hot encoding
    hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9],[1, 3, 4, 3],[1, 1, 0, 1]]})
    print(all, sess.run(tf.arg_max(all, 1)))


# ## MNIST Dataset

# ### Reading data and set variables

# In[86]:

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))


# ### Softmax

# In[87]:

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


# ### Training epoch/batch

# In[91]:

# parameters
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    # Initialize Tensorflow Variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c/total_batch
        
        print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.9f}'.format(avg_cost))
        
    # Repeat results on test dataset
    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))


# ### Sample image show and prediction

# In[94]:

import matplotlib.pyplot as plt
import random

with tf.Session() as sess:
    # Get one and predict
    r = random.randint(0, mnist.test.num_examples-1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    #print("Precision:", sess.run(tf.argmax(hypothesis,1), feed_dict={X:mnist.test.images[r:r+1]}))
    
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()

