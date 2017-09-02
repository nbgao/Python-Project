
# coding: utf-8

# In[11]:

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


# ## 使用2种不同配置的K近邻回归模型对美国波士顿房价数据进行回归预测

# In[3]:

# 导入KNeighborRegressor K近邻回归器
from sklearn.neighbors import KNeighborsRegressor


# ### Uniform-weighted KNR

# In[4]:

# 初始化K近邻回归器，并且调整配置，使得预测的方式为平均回归：weights='uniform'
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)


# ### Distance-weighted KNR

# In[6]:

# 初始化K近邻回归器，并且调整配置，使得预测的方式为根据距离加权回归：weights='distance'
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)


# ## 对2种不同配置的K近邻回归模型在美国波士顿房价数据上进行预测性能的评估

# ### Uniform-weighted KNR

# In[8]:

# 使用R-squared、MSE、MAR 三种指标对平均回归配置的K近邻模型在测试集上进行性能评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print('R-squared value of uniform-weighted KNeighborRegression:', uni_knr.score(X_test, y_test))
print('The mean squared error of uniform-weighted KNeighborRegression:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('The mean absolute error of uniform-weighted KNeighborRegression:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))


# ### Distance-weighted KNR

# In[9]:

# 使用R-squared、MSE、MAR 三种指标对根据距离加权回归配置的K近邻模型在测试集上进行性能评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print('R-squared value of distance-weighted KNeighborRegression:', dis_knr.score(X_test, y_test))
print('The mean squared error of distance-weighted KNeighborRegression:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print('The mean absolute error of distance-weighted KNeighborRegression:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))


# In[ ]:



