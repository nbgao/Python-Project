
# coding: utf-8

# In[2]:

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


# ## 使用3种集成回归模型对美国波士顿地区房价训练数据进行学习，并对测试数据进行预测

# ### RandomForestRegressor (RFR)

# In[11]:

# 使用RandomForestRegressor训练模型，并对测试数据作出预测
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)


# ### ExtraTreesRegressor (ETR)

# In[12]:

# 使用ExtraTreesRegressor训练模型，并对测试数据作出预测
from sklearn.ensemble import ExtraTreesRegressor

etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)
etr_y_predict = etr.predict(X_test)


# ### GradientBoostingRegressor (GBR)

# In[13]:

# 使用GradientBoostingRegressor训练模型，并对测试数据作出预测
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_y_predict = gbr.predict(X_test)


# ## 对3种集成回归模型在美国波士顿房价测试数据上的回归预测性能进行评估

# ### RandomForestRegressor (RFR)

# In[14]:

# 使用 R-squared、MSE、MAE 指标对默认配置的随机回归森林在测试集上进行性能评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print('R-squared value of RandomForestRegressor:', rfr.score(X_test, y_test))
print('The mean squared error of RandomForestRegressor:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print('The mean absolute error of RandomForestRegressor:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))


# ### ExtraTreesRegressor (ETR)

# In[51]:

# 使用 R-squared、MSE、MAE 指标对默认配置的极端回归森林在测试集上进行性能评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print('R-squared value of ExtraForestRegressor:', etr.score(X_test, y_test))
print('The mean squared error of ExtraForestRegressor:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('The mean absolute error of ExtraForestRegressor:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))

# 利用训练好的极端回归森林模型，输出每种特征对预测目标的贡献度
sorted(zip(etr.feature_importances_, boston.feature_names))


# ### GradientBoostingRegressor (GBR)

# In[52]:

# 使用 R-squared、MSE、MAE 指标对默认配置的梯度提升回归树在测试集上进行性能评估
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print('R-squared value of GradientBoostingRegressor:', gbr.score(X_test, y_test))
print('The mean squared error of GradientBoostingRegressor:', mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
print('The mean absolute error of GradientBoostingRegressor:', mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))


# ## 多种回归模型预测性能比较表

# |__Rank__|__Regressors__|__R-squared__|__MSE__|__MAE__|
# |:-------:|:-------------:|:-----------:|:-----:|:------:|
# |1|GradientBoostingRegressor|0.8434|12.14|2.27|
# |2|RandomForestRegressor|0.8182|14.10|2.42|
# |3|ExtraTreesRegressor|0.7954|15.86|2.52|
# |4|SVM Regressor (RBF Kernel)|0.7564|18.89|2.61|
# |5|KNN Regressor (Distance-weighted)|0.7198|21.73|2.81|
# |6|DecisionTreeRegressor|0.7020|23.10|3.03|
# |7|KNN Regressor (Uniform-weighted)|0.6903|24.01|2.97|
# |8|LinearRegression|0.6763|25.10|3.53|
# |9|SGDRegressor|0.6599|26.38|3.55|
# |10|SVM Regressor (Linear Kernel)|0.6517|27.76|3.57|
# |11|SVM Regressor (Poly Kernel)|0.4045|46.18|3.75|

# In[ ]:



