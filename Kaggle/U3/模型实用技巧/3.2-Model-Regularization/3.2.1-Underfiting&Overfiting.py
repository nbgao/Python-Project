
# coding: utf-8

# # 1. Overfiting

# ## 1.1 使用线性回归模型在比萨训练样本上进行拟合

# In[2]:

# 输入训练样本的特征以及目标值，分别存储在X_train与y_train之中
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]


# In[3]:

# 从sklearn.linear_model中导入LinearRegression
from sklearn.linear_model import LinearRegression
# 使用默认配置初始化线性回归模型
regressor = LinearRegression()
# 直接以比萨的直径作为特征训练模型
regressor.fit(X_train, y_train)


# In[4]:

import numpy as np
# 在x轴上从0至25均匀采样100个数据点
xx = np.linspace(0,26,100)
xx = xx.reshape(xx.shape[0], 1)
# 以上述100个数据点作为基准，预测回归直线
yy = regressor.predict(xx)


# In[10]:

# 对回归预测到的直线进行作图
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))

plt.scatter(X_train, y_train)
plt.plot(xx, yy, label='Degree=1')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.show()


# In[9]:

# 输出线性回归模型在训练样本上的R-squared值
print('The R-squared value of Linear Regression performing on the training data is\n', 
     regressor.score(X_train, y_train))


# ## 1.2 使用2次多项式回归模型在比萨训练样本上进行拟合

# In[23]:

# 从sklearn.preprocessing中导入多项式特征产生器
from sklearn.preprocessing import PolynomialFeatures

# 使用PolynomialFeatures(degree=2)映射出2次多项式特征，存储在变量X_train_poly2中
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)

# 以线性回归器为基础，初始化回归模型。尽管特征的维度有提升，但是模型基础仍然是线性模型
regressor_poly2 = LinearRegression()

# 对2次多项式回归模型进行训练
regressor_poly2.fit(X_train_poly2, y_train)

# 从新映射绘图用x轴采样数据
xx_poly2 = poly2.transform(xx)

# 使用2次多项式回归模型对应x轴采样数据进行回归预测
yy_poly2 = regressor_poly2.predict(xx_poly2)


# In[24]:

# 分别对训练数据点、线性回归直线、2次多项式回归曲线进行作图
plt.figure(figsize=(8,6))

plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend()

plt.show()


# In[21]:

# 输出2次多项式回归模型在训练样本上的R-squared值
print('The R-squared value of Polynomial Regressor (Degree=2) performing on the training data is', regressor_poly2.score(X_train_poly2, y_train))


# ## 1.3 使用3次多项式回归模型在比萨训练样本上进行拟合

# In[34]:

# 从sklearn.preprocessing中导入多项式特征产生器
from sklearn.preprocessing import PolynomialFeatures

# 使用PolynomialFeatures(degree=3)映射出3次多项式特征，存储在变量X_train_poly3中
poly3 = PolynomialFeatures(degree=3)
X_train_poly3 = poly3.fit_transform(X_train)

# 以线性回归器为基础，初始化回归模型。尽管特征的维度有提升，但是模型基础仍然是线性模型
regressor_poly3 = LinearRegression()

# 对3次多项式回归模型进行训练
regressor_poly3.fit(X_train_poly3, y_train)

# 从新映射绘图用x轴采样数据
xx_poly3 = poly3.transform(xx)

# 使用3次多项式回归模型对应x轴采样数据进行回归预测
yy_poly3 = regressor_poly3.predict(xx_poly3)


# In[38]:

# 分别对训练数据点、线性回归直线、2次多项式回归曲线进行作图
plt.figure(figsize=(8,6))

plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
plt3, = plt.plot(xx, yy_poly3, label='Degree=3')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend()

plt.show()


# In[43]:

# 输出3次多项式回归模型在训练样本上的R-squared值
print('The R-squared value of Polynomial Regressor (Degree=3) performing on the training data is', regressor_poly3.score(X_train_poly3, y_train))


# ## 1.4 使用4次多项式回归模型在比萨训练样本上进行拟合

# In[25]:

# 从sklearn.preprocessing导入多项式特征生成器
from sklearn.preprocessing import PolynomialFeatures

# 初始化4次多项式特征生成器
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)

# 使用默认配置初始化4次多项式回归器
regressor_poly4 = LinearRegression()
# 对4次多项式回归模型进行训练
regressor_poly4.fit(X_train_poly4, y_train)

# 从新映射绘图用x轴采样数据
xx_poly4 = poly4.transform(xx)
# 使用4次多项式回归模型对应x轴采样数据进行回归预测
yy_poly4 = regressor_poly4.predict(xx_poly4)


# In[41]:

# 分别对训练数据点、线性回归直线、2次多项式以及4次多项式回归曲线进行作图
plt.figure(figsize=(8,6))

plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
plt3, = plt.plot(xx, yy_poly3, label='Degree=3')
plt4, = plt.plot(xx, yy_poly4, label='Degree=4')

plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend()

plt.show()


# In[27]:

# 输出4次多项式回归模型在训练样本上的R-squared值
print('The R-squared value of Polynomial Regressor (Degree=4) performing on the training data is', 
     regressor_poly4.score(X_train_poly4, y_train))


# ## 1.5 评估3种回归模型在测试数据集上的性能表现

# In[30]:

# 准备测试数据
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

# 使用测试数据对线性回归模型的性能进行评估
regressor.score(X_test, y_test)


# In[31]:

# 使用测试数据对2次多项式回归模型的性能进行评估
X_test_poly2 = poly2.transform(X_test)
regressor_poly2.score(X_test_poly2, y_test)


# In[42]:

# 使用测试数据对3次多项式回归模型的性能进行评估
X_test_poly3 = poly3.transform(X_test)
regressor_poly3.score(X_test_poly3, y_test)


# In[32]:

# 使用测试数据对4次多项式回归模型的性能进行评估
X_test_poly4 = poly4.transform(X_test)
regressor_poly4.score(X_test_poly4, y_test)


# ### 4种模型性能对比表

# |特征多项式次数|训练集R-squared值|测试集R-squared值|
# |:--:|:--:|:--:|
# |Degree=1|0.9100|0.8097|
# |Degree=2|0.9816|0.8675|
# |Degree=3|0.9947|0.8357|
# |Degree=4|1.0000|0.8096|
