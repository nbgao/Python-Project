
# coding: utf-8

# ## XOR with Logistic Regression (Binary classification)

# ### XOR data set

# In[2]:

import numpy as np
import tensorflow as tf

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype=np.float32)


# ### $ H(X)=sigmoid(XW)=\frac{1}{1+e^{-XW}}$
# ### $ cost(W)=-\frac{1}{m}\sum{ylog(H(x)) + (1-y)(log(1-H(x)))} $

# In[8]:

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else false
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step%1000==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))
    
    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


# #### One logistic function unit cannot seperate XOR

# ## Nerual Net
# ### 2 layers

# In[12]:

W1 = tf.Variable(tf.random_normal([2,2], name='weight1'))
b1 = tf.Variable(tf.random_normal([2], name='bias1'))
layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.random_normal([2,1], name='weight2'))
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2)


# In[15]:

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else false
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step%1000==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run([W1,W2]))
    
    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


# ### 4 layers

# In[22]:

W1 = tf.Variable(tf.random_normal([2,10], name='weight1'))
b1 = tf.Variable(tf.random_normal([10], name='bias1'))
layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.random_normal([10,10], name='weight2'))
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1,W2)+b2)

W3 = tf.Variable(tf.random_normal([10,10], name='weight3'))
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2,W3)+b3)

W4 = tf.Variable(tf.random_normal([10,1], name='weight4'))
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypothesis = tf.sigmoid(tf.matmul(layer3,W4)+b4)


# In[23]:

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else false
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(1001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step%100==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
    
    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


# ### Formal defination
# ### $ W:=W-\alpha \frac{\partial }{\partial W} \frac{1}{2m} \sum _{i=1}^m{(Wx^{(i)}-y^{(i)})^2} $
# ### $ W:=W-\alpha \frac{1}{2m} \sum_{i=1}^m{2(Wx^{(i)}-y^{(i)})x^{(i)}} $
# ### $ W:=W-\alpha \frac{1}{m} \sum_{i=1}^m{(Wx^{(i)}-y^{(i)})x^{(i)}} $

#  ### Gradient descent algorithm
#  ### $ W:=W-\alpha \frac{1}{m} \sum_{i=1}^m{(Wx^{(i)}-y^{(i)})x^{(i)}} $
#  ### $ w = w - \alpha \frac{\partial{E}}{\partial{W}} $
#  #### train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# ### Go deep & wide

# In[36]:

# 9 hidden layers!
# 11 Weights
W1 = tf.Variable(tf.random_uniform([2,5], -1.0, 1.0), name='Weight1')
W2 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='Weight2')
W3 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='Weight3')
W4 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='Weight4')
W5 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='Weight5')
W6 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='Weight6')
W7 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='Weight7')
W8 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='Weight8')
W9 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='Weight9')
W10 = tf.Variable(tf.random_uniform([5,5], -1.0, 1.0), name='Weight10')
W11 = tf.Variable(tf.random_uniform([5,1], -1.0, 1.0), name='Weight11')

# 11 Bias
b1 = tf.Variable(tf.zeros([5], name='Bias1'))
b2 = tf.Variable(tf.zeros([5], name='Bias2'))
b3 = tf.Variable(tf.zeros([5], name='Bias3'))
b4 = tf.Variable(tf.zeros([5], name='Bias4'))
b5 = tf.Variable(tf.zeros([5], name='Bias5'))
b6 = tf.Variable(tf.zeros([5], name='Bias6'))
b7 = tf.Variable(tf.zeros([5], name='Bias7'))
b8 = tf.Variable(tf.zeros([5], name='Bias8'))
b9 = tf.Variable(tf.zeros([5], name='Bias9'))
b10 = tf.Variable(tf.zeros([5], name='Bias10'))
b11 = tf.Variable(tf.zeros([1], name='Bias11'))

# 11 Layers
with tf.name_scope('layer1') as scope:
    L1 = tf.sigmoid(tf.matmul(X,W1)+b1)
with tf.name_scope('layer2') as scope:
    L2 = tf.sigmoid(tf.matmul(L1,W2)+b2)
with tf.name_scope('layer3') as scope:
    L3 = tf.sigmoid(tf.matmul(L2,W3)+b3)
with tf.name_scope('layer4') as scope:
    L4 = tf.sigmoid(tf.matmul(L3,W4)+b4)
with tf.name_scope('layer5') as scope:
    L5 = tf.sigmoid(tf.matmul(L4,W5)+b5)
with tf.name_scope('layer6') as scope:
    L6 = tf.sigmoid(tf.matmul(L5,W6)+b6)
with tf.name_scope('layer7') as scope:
    L7 = tf.sigmoid(tf.matmul(L6,W7)+b7)
with tf.name_scope('layer8') as scope:
    L8 = tf.sigmoid(tf.matmul(L7,W8)+b8)
with tf.name_scope('layer9') as scope:
    L9 = tf.sigmoid(tf.matmul(L8,W9)+b9)
with tf.name_scope('layer10') as scope:
    L10 = tf.sigmoid(tf.matmul(L9,W10)+b10)
with tf.name_scope('last') as scope:
    hypothesis = tf.sigmoid(tf.matmul(L10,W11)+b11)


# In[38]:

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else false
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(401):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step%100==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
    
    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


# ##### Not work!

# ### ReLU: Rectified Linear Unit
# #### L1 = tf.sigmoid(tf.matmul(X,W1)+b1)
# #### L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

# In[41]:

# 11 Layers with ReLU
with tf.name_scope('layer1') as scope:
    L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
with tf.name_scope('layer2') as scope:
    L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
with tf.name_scope('layer3') as scope:
    L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)
with tf.name_scope('layer4') as scope:
    L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)
with tf.name_scope('layer5') as scope:
    L5 = tf.nn.relu(tf.matmul(L4,W5)+b5)
with tf.name_scope('layer6') as scope:
    L6 = tf.nn.relu(tf.matmul(L5,W6)+b6)
with tf.name_scope('layer7') as scope:
    L7 = tf.nn.relu(tf.matmul(L6,W7)+b7)
with tf.name_scope('layer8') as scope:
    L8 = tf.nn.relu(tf.matmul(L7,W8)+b8)
with tf.name_scope('layer9') as scope:
    L9 = tf.nn.relu(tf.matmul(L8,W9)+b9)
with tf.name_scope('layer10') as scope:
    L10 = tf.nn.relu(tf.matmul(L9,W10)+b10)
with tf.name_scope('last') as scope:
    hypothesis = tf.sigmoid(tf.matmul(L10,W11)+b11)


# In[51]:

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else false
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

writer = tf.summary.FileWriter("logs/", sess.graph)

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step%1000==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
    
    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


# ##### Works very well!

# ## Nerual Network tips

# ### 1. Initializing weights
# ** W = tf.Variable(tf.random_normal([1]), name='weight') **
# 
# ** W = tf.get_variable("W", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer()) **

# ### 2. Activation functions
# ** tf.sigmoid **
# 
# ** tf.tanh **
# 
# ** tf.nn.relu **

# ### 3. Regularization
# #### Solutions for overfitting
# * More training data
# * Reduce the number of features
# * Regularization

# ### $ L(W)=\frac{1}{N}\sum_{i=1}^N{L_i(f(x_i,W),y_i})+\lambda R(W) $
# ### $ L=\frac{1}{N}\sum_{i=1}^N \sum_{j\neq y_i} max[0,f(x_i;W)_j - f(x_i;W)_{y_i}+1] + \lambda R(W) $
# #### λ =  regularization strength (hyperparameter)

# ### In common use:
# #### L2 regularization   $ R(W)=\sum_k \sum_l W_{k,l}^2 $
# #### L1 regularization   $ R(W)=\sum_k \sum_l \left|W_{k,l}\right| $
# #### Elastic net (L1+L2)   $ R(W)=\sum_k \sum_l \beta W_{k,l}^2 + \left|W_{k.l}\right| $
# #### Max norm rgularization
# #### Dropout
# #### Fancier: Batch normalization, stochastic depth

# In[87]:

# dropout (keep_prob) rate 0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[784,512])
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[512,512])
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# parameters
training_epochs = 15
batch_size = 100

# train model
for epoch in range(training_epochs):
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs, Y:batch_ys, keep_prob:0.7}
        c, _ = sess.run([cost, otimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
        
# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1}))


# ### 4. Optimizers
# ** tf.train.GradientDescentOptimizer **
# 
# ** tf.train.AdadeltaOptimizer **
# 
# ** tf.train.AdagradOptimizer **
# 
# ** tf.train.AdagradDAOptimizer **
# 
# ** tf.train.MomentumOptimer **
# 
# ** tf.train.AdamOptimizer **
# 
# ** tf.train.FtrlOptimer **
# 
# ** tf.train.ProximalGradientDescentOptimizer **
# 
# ** tf.train.ProxiamlAdagradOptimizer **
# 
# ** tf.train.RMSPropOptimizer **

# In[ ]:

#define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# ## TensorBoard

# ### 1. From TF graph, decide which tensors you want to log
# w2_hist = tf.summary.histogram("weight2", w2) <br> cost_summ = tf.summary.scalar("cost", cost)
# 
# ### 2. Merge all summaries
# summary = tf.summary.merge_all()
# 
# ### 3. Create writer and add graph
# writer =tf.summary.FileWriter('./logs') <br>
# writer.add_graph(sess.graph)
# 
# ### 4. Run summary merge and add_summary
# s, _ = sess.run([summary, optimizer], feed_dict=feed_dict) <br>
# writer.add_summary(s, global_step=global_step)
# 
# ### 5. Launch TensorBoard
# tensorboard --logdir=./logs

# In[ ]:

# Scalar tensors
cost_summ = tf.summary.scalar("cost", cost)

# Histogram(muti-dimensional tensors)
W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1,W2) + b2)

W2_hist = tf.summary.histogram('weight2', W2)
b2_hist = tf.summary.histogram('bias2', b2)
hypothesis_hist = tf.summary.histogram('hypothesis', hypothesis)

# Add scope for better graph hierarchy
with tf.name_scope('layer1') as scope:
    W1 = tf.Variable(tf.random_normal([2,2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    hypothesis = tf.sigmoid(tf.matmul(X,W1) + b1)
    
    W1_hist = tf.summary.histogram('weight1', W1)
    b1_hist = tf.summary.histogram('bias1', b1)
    layer1 = tf.summary.histogram('layer1', layer1)
    
with tf.name_scope('layer2') as scope:
    W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1,W2) + b2)
    
    W2_hist = tf.summary.histogram('weight2', W2)
    b2_hist = tf.summary.histogram('bias2', b2)
    hypothesis_hist = tf.summary.histogram('hypothesis', hypothesis)


# ### 2/3. Merge summaries and create writer after creating session

# In[ ]:

# Summaray
summary = tf.summary.merge_all()

# Initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Crate summary writer
writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)


# ### 4. Run merged summary and write (add summary)

# In[ ]:

s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
writer.add_summary(s, global_step=global_step)
global_step += 1


# ### 5. Launch tensorboard
# #### Local

# In[ ]:

writer = tf.summary.FileWriter("./logs/xor_logs")


# $ tensorboard -logdir=./logs/xor_logs

# #### Remote server

# $ ssh -L local_port:127.0.0.1:remote_port username@server.com
# 
# $ tensorboard -logdir=./logs/xor_logs

# ### P1 矩阵相乘

# In[54]:

with tf.name_scope('graph') as scope:
    matrix1 = tf.constant([[3., 3.]],name ='matrix1')  #1 row by 2 column
    matrix2 = tf.constant([[2.],[2.]],name ='matrix2') # 2 row by 1 column
    product = tf.matmul(matrix1, matrix2, name='product')

sess = tf.Session()
writer = tf.summary.FileWriter("logs/matmul", sess.graph)
init = tf.global_variables_initializer()
sess.run(init)


# ### P2 线性拟合1

# In[80]:

# Prepare the original data
with tf.name_scope('data'):
    x_data = np.random.rand(100).astype(np.float32)
    y_data = 0.3 * x_data + 0.1

# Create parameters
with tf.name_scope('paramters'):
    with tf.name_scope('weights'):
        weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
        tf.summary.histogram('weight', weight)
    with tf.name_scope('biases'):
        bias = tf.Variable(tf.zeros([1]))
        tf.summary.histogram('bias', bias)

# Get y_prediction
with tf.name_scope('y_prediction'):
    y_prediction = weight * x_data + bias
    
# Compute the loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y_data - y_prediction))
    tf.summary.scalar('loss', loss)
    
# Create optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)

# Create train, minimize the loss
with tf.name_scope('train'):
    train = optimizer.minimize(loss)

# Create init
with tf.name_scope('init'):
    init = tf.global_variables_initializer()
    
# Create a session
sess = tf.Session()

# Merged
merged = tf.summary.merge_all()

# Initialize
writer = tf.summary.FileWriter("logs/lr1", sess.graph)
sess.run(init)

# Loop
for step in range(101):
    sess.run(train)
    # rs = sess.run(merged)
    # writer.add_summary(rs, step)
    if step%10 == 0:
        print(step, 'weight:', sess.run(weight), 'bias:', sess.run(bias))


# In[ ]:



