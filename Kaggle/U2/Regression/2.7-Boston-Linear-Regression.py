
# coding: utf-8

# In[1]:

from sklearn.datasets import load_boston


# In[3]:

# 从读取房价数据存储在boston变量中
boston = load_boston()
# 输出数据描述
print(boston.DESCR)


# ## 美国波士顿地区房价数据分割

# In[8]:

import numpy as np
# 导入数据分割器
from sklearn.cross_validation import train_test_split

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)

print("The max target value is", np.max(boston.target))
print("The min target value is", np.min(boston.target))
print("The average target value is", np.mean(boston.target))


# ## 训练与测试数据标准化处理

# In[20]:

# 从sklearn.preprocessing导入数据标准化模块
from sklearn.preprocessing import StandardScaler


# In[21]:

# 分别初始化对特征和目标值的标准化
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)


# ## 使用线性回归模型LinearRegression和SGDRegressor分别对美国波士顿地区房价进行预测

# ### LinearRegression

# In[22]:

from sklearn.linear_model import LinearRegression
# 使用默认配置初始化线性回归器LinearRegression
lr = LinearRegression()
# 使用训练数据进行参数估计
lr.fit(X_train, y_train)
# 对测试数据进行回归预测
lr_y_predict = lr.predict(X_test)


# ### SGDRegressor

# In[23]:

from sklearn.linear_model import SGDRegressor
# 使用默认配置初始化线性回归器SGDRegressor
sgdr = SGDRegressor()
# 使用训练数据进行参数估计
sgdr.fit(X_train, y_train)
# 对测试数据进行回归预测
sgdr_y_predict = sgdr.predict(X_test)


# ## 使用3种回归评价机制以及2种调用R-squard评价模块的方法对模型的回归性能做出评价

# ### LinearRegression

# In[24]:

# 使用LinearRegression模型自带的评估模块，并输出评估结果
print('The value of default measurement of LinearRegression is', lr.score(X_test, y_test))


# In[25]:

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 使用 r2_score 模块，并输出评估结果
print('The value of R-squared of LinearRegression is', r2_score(y_test, lr_y_predict))

# 使用 mean_squared_error 模块，并输出评估结果
print('The mean squared error of LinearRegression is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

# 使用 mean_absolute_error 模块，并输出评估结果
print('The mean absolute error of LinearRegression is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))


# ### SGDRegressor

# In[26]:

# 使用SGDRegressor模型自带的评估模块，并输出评估结果
print('The value of default measurement of SGDRegressor is', sgdr.score(X_test, y_test))


# In[30]:

# 使用 r2_score 模块，并输出评估结果
print('The value of R-squared of SGDRegressor is', r2_score(y_test, sgdr_y_predict))

# 使用 mean_squared_error 模块，并输出评估结果
print('The mean squared error of SGDRegressor is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))

# 使用 mean_absolute_error 模块，并输出评估结果
print('The mean absolute error of SGDRegressor is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))


# In[ ]:



