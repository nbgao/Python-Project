
# coding: utf-8

# # 1. Numpy数组的基本操作

# ## 1.1 重设形状

# numpy.reshape(a, newshape)

# In[1]:

import numpy as np

np.arange(10).reshape((5, 2))


# ## 1.2 数组展开

# numpy.ravel(a, order='C')

# In[3]:

a = np.arange(10).reshape((2, 5))
a


# In[4]:

np.ravel(a)


# In[5]:

np.ravel(a, order='F')


# ## 2.3 轴移动

# numpy.moveaxis(a, source, destination)

# In[8]:

a = np.ones((1,2,3))
a


# In[9]:

np.moveaxis(a, 0, -1)


# In[11]:

a.shape


# In[12]:

np.moveaxis(a, 0, -1).shape


# ## 1.4 轴交换

# numpy.swapaxes(a, axis1, axis2)

# In[13]:

a = np.ones((1,4,3))

np.swapaxes(a, 0 ,2)


# In[14]:

a.shape


# In[16]:

np.swapaxes(a, 0, 2).shape


# ## 2.5 数组转置

# numpy.transpose(a, axes=None)

# In[20]:

a = np.arange(4).reshape(2, 2)
a


# In[21]:

np.transpose(a)


# In[22]:

a.T


# ## 1.6 维度改变

# numpy.atleast_1d() <br>
# numpy.atleast_2d() <br>
# numpy.atleast_3d() <br>

# In[25]:

np.atleast_1d([1])


# In[26]:

np.atleast_2d([1])


# In[27]:

np.atleast_3d({1})


# ## 2.7 类型转变

# In[29]:

a = np.arange(4).reshape(2,2)
a


# In[30]:

np.asmatrix(a)


# ## 2.8 数组连接

# numpy.concatenate((a1, a2, $\cdots$ ) , axis=0)

# In[32]:

a = np.array([[1,2], [3,4], [5,6]])
b = np.array([[7,8], [9,10]])
c = np.array([[11,12]])

np.concatenate((a,b,c), axis=0)


# In[33]:

a = np.array([[1,2], [3,4], [5,6]])
b = np.array([[7,8,9]])

np.concatenate((a, b.T), axis=1)


# ## 2.9 数组重叠

# In[34]:

a = np.array([1,2,3])
b = np.array([4,5,6])
np.stack((a,b))


# In[35]:

np.stack((a,b), axis=-1)


# ## 2.10 拆分

# In[36]:

a = np.arange(10)
a


# In[37]:

np.split(a, 5)


# In[38]:

b = np.arange(10).reshape(2,5)
b


# In[39]:

np.split(b, 2)


# ## 2.11 删除

# In[40]:

a = np.arange(12).reshape(3,4)
a


# In[41]:

np.delete(a, 2, 1)


# In[42]:

np.delete(a, 2, 0)


# ## 2.12 数组插入

# In[43]:

a = np.arange(12).reshape(3,4)
b = np.arange(4)

np.insert(a, 2, b, 0)


# In[44]:

a


# In[45]:

b


# ## 2.13 附加

# In[46]:

a = np.arange(6).reshape(2,3)
b = np.arange(3)

np.append(a, b)


# ## 2.14 重设尺寸

# In[50]:

a = np.arange(10)
a.reshape(2,5)


# In[51]:

a


# In[53]:

a.resize(2,5)
a


# ## 2.15 翻转数组

# In[56]:

a = np.arange(16).reshape(4,4)

np.fliplr(a)
a


# In[57]:

np.flipud(a)
a


# # 2. Numpy 随机抽样

# ## 2.1 numpy.random.rand

# In[58]:

np.random.rand(2,5)


# ## 2.2 numpy.random.randn

# In[59]:

np.random.randn(1, 10)


# ## 2.3 numpy.random.randint

# ** randint(low, high, size, dtype) ** <br>
# ** 区间: [low, high) **

# In[60]:

np.random.randint(2, 5, 10)


# ## 2.4 numpy.random.random_integers

# ** random_integers(low, high, size) ** <br>
# ** 区间: [low, high] **

# In[61]:

np.random.random_integers(2, 5, 10)


# ## 2.5 numpy.random.random_sample

# ** 区间: [0,1) 随机浮点数**

# In[62]:

np.random.random_sample([10])


# ## 2.6 numpy.random.choice

# ** choice(a, size, replace, p) **

# In[63]:

np.random.choice(10, 5)


# ## 2.7 概率分布密度

# 1. numpy.random.beta(a，b，size)：从 Beta 分布中生成随机数。
# - numpy.random.binomial(n, p, size)：从二项分布中生成随机数。
# - numpy.random.chisquare(df，size)：从卡方分布中生成随机数。
# - numpy.random.dirichlet(alpha，size)：从 Dirichlet 分布中生成随机数。
# - numpy.random.exponential(scale，size)：从指数分布中生成随机数。
# - numpy.random.f(dfnum，dfden，size)：从 F 分布中生成随机数。
# - numpy.random.gamma(shape，scale，size)：从 Gamma 分布中生成随机数。
# - numpy.random.geometric(p，size)：从几何分布中生成随机数。
# - numpy.random.gumbel(loc，scale，size)：从 Gumbel 分布中生成随机数。
# - numpy.random.hypergeometric(ngood, nbad, nsample, size)：从超几何分布中生成随机数。
# - numpy.random.laplace(loc，scale，size)：从拉普拉斯双指数分布中生成随机数。
# - numpy.random.logistic(loc，scale，size)：从逻辑分布中生成随机数。
# - numpy.random.lognormal(mean，sigma，size)：从对数正态分布中生成随机数。
# - numpy.random.logseries(p，size)：从对数系列分布中生成随机数。
# - numpy.random.multinomial(n，pvals，size)：从多项分布中生成随机数。
# - numpy.random.multivariate_normal(mean, cov, size)：从多变量正态分布绘制随机样本。
# - numpy.random.negative_binomial(n, p, size)：从负二项分布中生成随机数。
# - numpy.random.noncentral_chisquare(df，nonc，size)：从非中心卡方分布中生成随机数。
# - numpy.random.noncentral_f(dfnum, dfden, nonc, size)：从非中心 F 分布中抽取样本。
# - numpy.random.normal(loc，scale，size)：从正态分布绘制随机样本。
# - numpy.random.pareto(a，size)：从具有指定形状的 Pareto II 或 Lomax 分布中生成随机数。
# - numpy.random.poisson(lam，size)：从泊松分布中生成随机数。
# - numpy.random.power(a，size)：从具有正指数 a-1 的功率分布中在 0，1 中生成随机数。
# - numpy.random.rayleigh(scale，size)：从瑞利分布中生成随机数。
# - numpy.random.standard_cauchy(size)：从标准 Cauchy 分布中生成随机数。
# - numpy.random.standard_exponential(size)：从标准指数分布中生成随机数。
# - numpy.random.standard_gamma(shape，size)：从标准 Gamma 分布中生成随机数。
# - numpy.random.standard_normal(size)：从标准正态分布中生成随机数。
# - numpy.random.standard_t(df，size)：从具有 df 自由度的标准学生 t 分布中生成随机数。
# - numpy.random.triangular(left，mode，right，size)：从三角分布中生成随机数。
# - numpy.random.uniform(low，high，size)：从均匀分布中生成随机数。
# - numpy.random.vonmises(mu，kappa，size)：从 von Mises 分布中生成随机数。
# - numpy.random.wald(mean，scale，size)：从 Wald 或反高斯分布中生成随机数。
# - numpy.random.weibull(a，size)：从威布尔分布中生成随机数。
# - numpy.random.zipf(a，size)：从 Zipf 分布中生成随机数。
