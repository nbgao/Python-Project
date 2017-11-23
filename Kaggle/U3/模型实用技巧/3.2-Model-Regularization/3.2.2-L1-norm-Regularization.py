
# coding: utf-8

# # 2. L1 Norm Regularization

# In[24]:

# 输入训练样本的特征以及目标值，分别存储在X_train与y_train之中
X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

# 从sklearn.linear_model中导入LinearRegression
from sklearn.linear_model import LinearRegression
# 使用默认配置初始化线性回归模型
regressor = LinearRegression()
# 直接以比萨的直径作为特征训练模型
regressor.fit(X_train, y_train)

# 从sklearn.preprocessing导入多项式特征生成器
from sklearn.preprocessing import PolynomialFeatures

# 初始化4次多项式特征生成器
poly4 = PolynomialFeatures(degree=4)
X_train_poly4 = poly4.fit_transform(X_train)

# 使用默认配置初始化4次多项式回归器
regressor_poly4 = LinearRegression()
# 对4次多项式回归模型进行训练
regressor_poly4.fit(X_train_poly4, y_train)

# 准备测试数据
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]

X_test_poly4 = poly4.transform(X_test)
len(X_test_poly4)


# ## Lasso模型在4次多项式特征上的拟合表现

# In[26]:

from sklearn.linear_model import Lasso
# 使用默认配置初始化Lasso
lasso_poly4 = Lasso()
# 使用Lasso对4此多项式特征进行拟合
lasso_poly4.fit(X_train_poly4, y_train)

# 对Lasso模型在测试样本上的回归性能进行评估
print(lasso_poly4.score(X_test_poly4, y_test))


# In[27]:

# 输出Lasso模型的参数列表
print(lasso_poly4.coef_)


# In[28]:

# 回顾普通4次多项式回归模型过拟合之后的性能
print(regressor_poly4.score(X_test_poly4, y_test))


# In[29]:

# 回顾普通4次多项式回归模型的参数列表
print(regressor_poly4.coef_)


# # 3. L2 Norm Regularization

# ## Ridge模型在4次多项式特征上的拟合表现

# In[30]:

# 输出普通4次多项式回归模型的参数列表
print(regressor_poly4.coef_)


# In[34]:

# 输出上述这些参数的平方和，验证参数之间的巨大差异
import numpy as np
print(np.sum(regressor_poly4.coef_ ** 2))


# In[35]:

# 从sklearn.linear_model导入Ridge
from sklearn.linear_model import Ridge
# 使用默认配置初始化Ridge
ridge_poly4 = Ridge()

# 使用Ridge模型对4次多项式特征进行拟合
ridge_poly4.fit(X_train_poly4, y_train)

# 输出Ridge模型在测试样本上的回归性能
print(ridge_poly4.score(X_test_poly4, y_test))


# In[36]:

# 输出Ridge模型的参数列表，观察参数差异
print(ridge_poly4.coef_)


# In[37]:

# 计算Ridge模型拟合后参数的平方和
print(np.sum(ridge_poly4.coef_ ** 2))


# In[ ]:



