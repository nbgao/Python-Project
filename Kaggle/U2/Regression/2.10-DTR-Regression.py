
# coding: utf-8

# In[10]:

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


# ## 使用回归树对美国波士顿地区房价训练数据进行学习，并对测试数据进行预测

# In[11]:

# 导入DecisionTreeRegressor (DTR)
from sklearn.tree import DecisionTreeRegressor


# In[12]:

# 使用默认配置初始化DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
dtr_y_predict = dtr.predict(X_test)


# ## 对单一回归树模型在美国波士顿地区房价测试数据上的预测性能进行评估

# In[15]:

# 使用 R-squared、MSE、MAE 指标对默认配置的回归树在测试集上进行性能评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print('R-squared value of DecisionTreeRegressor:', dtr.score(X_test, y_test))
print('The mean squared error of DecisionTreeRegressor:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print('The mean absolute error of DecisionTreeRegressor:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))


# In[ ]:



