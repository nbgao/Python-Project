
# coding: utf-8

# ## 使用“线性回归器”中分割处理好的训练和测试数据

# In[16]:

from sklearn.datasets import load_boston
# 从读取房价数据存储在boston变量中
boston = load_boston()

import numpy as np
# 导入数据分割器
from sklearn.cross_validation import train_test_split

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)

# 从sklearn.preprocessing导入数据标准化模块
from sklearn.preprocessing import StandardScaler

# 分别初始化对特征和目标值的标准化
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)


# ## 使用3种不同核函数配置的支持向量机回归模型进行训练，并且分别对测试数据进行预测

# In[6]:

# 导入支持向量机(回归)模型
from sklearn.svm import SVR


# ### Linear SVR

# In[7]:

# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict = linear_svr.predict(X_test)


# ### Poly SVR

# In[9]:

# 使用多项式核函数配置的支持向量机进行回归预测，并且对测试样本进行预测
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)


# ### RBF SVR

# In[10]:

# 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)


# ## 对3种核函数配置下的支持向量机回归模型在相同测试集上进行性能评估

# ### Linear SVR

# In[13]:

# 使用 R-squared、MSE和MAE指标对3种配置的支持向量机(回归)模型在相同测试集上进行性能评估
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

# Linear SVR
print('R-squared value of linear SVR is', linear_svr.score(X_test, y_test))
print('The mean squared error of linear SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('The mean absolute error of linear SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))


# ### Poly SVR

# In[14]:

# Poly SVR
print('R-squared value of Poly SVR is', poly_svr.score(X_test, y_test))
print('The mean squared error of Poly SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print('The mean absolute error of Poly SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))


# ### RBF SVR

# In[15]:

# RBF SVR
print('R-squared value of RBF SVR is', rbf_svr.score(X_test, y_test))
print('The mean squared error of RBF SVR is', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print('The mean absolute error of RBF SVR is', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))


# In[ ]:



