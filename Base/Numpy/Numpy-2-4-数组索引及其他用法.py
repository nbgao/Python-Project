
# coding: utf-8

# # 1. Numpy数组索引和切片

# ## 1.1 数组索引

# In[1]:

import numpy as np

a = np.arange(10)
a


# In[2]:

a[[1, 2, 3]]


# In[3]:

a = np.arange(20).reshape(4, 5)
a


# In[4]:

a[1, 2]


# In[5]:

a = [[0,1,2,3,4], [5,6,7,8,9], [10,11,12,13,14], [15,16,17,18,19]]
a


# In[6]:

a[1,2]


# In[7]:

a[1][2]


# In[8]:

a = np.arange(20).reshape(4, 5)
a


# In[9]:

a[[1,2], [3,4]]


# In[10]:

a = np.arange(30).reshape(2,5,3)
a


# In[11]:

a[[0,1], [1,2], [1,2]]


# ## 1.2 数组切片

# In[13]:

a = np.arange(10)
a


# In[14]:

a[:5]


# In[15]:

a[5:10]


# In[16]:

a[0:10:2]


# In[17]:

a = np.arange(20).reshape(4,5)
a


# In[18]:

a[0:3, 2:4]


# In[19]:

a[:, ::2]


# ## 1.3 索引与切片区别

# In[20]:

a = np.arange(10)
a


# In[23]:

a[1] = 100


# In[24]:

a


# # 2. 排序、搜索、计数

# ## 2.1 排序

# numpy.sort(a, axis=-1, kind='quicksort', order=None)

# In[25]:

a = np.random.rand(20).reshape(4,5)
a


# In[26]:

np.sort(a)


# 1. numpy.lexsort(keys ,axis)：使用多个键进行间接排序。
# - numpy.argsort(a ,axis,kind,order)：沿给定轴执行间接排序。
# - numpy.msort(a)：沿第 1 个轴排序。
# - numpy.sort_complex(a)：针对复数排序。

# ## 2.2 搜索和计数

# 1. argmax(a ,axis,out)：返回数组中指定轴的最大值的索引。
# - nanargmax(a ,axis)：返回数组中指定轴的最大值的索引,忽略 NaN。
# - argmin(a ,axis,out)：返回数组中指定轴的最小值的索引。
# - nanargmin(a ,axis)：返回数组中指定轴的最小值的索引,忽略 NaN。
# - argwhere(a)：返回数组中非 0 元素的索引,按元素分组。
# - nonzero(a)：返回数组中非 0 元素的索引。
# - flatnonzero(a)：返回数组中非 0 元素的索引,并铺平。
# - where(条件,x,y)：根据指定条件,从指定行、列返回元素。
# - searchsorted(a,v ,side,sorter)：查找要插入元素以维持顺序的索引。
# - extract(condition,arr)：返回满足某些条件的数组的元素。
# - count_nonzero(a)：计算数组中非 0 元素的数量。

# In[28]:

a = np.random.randint(0, 10, 20)
a


# In[29]:

np.argmax(a)


# In[30]:

np.nanargmax(a)


# In[31]:

np.argmin(a)


# In[32]:

np.argwhere(a)


# In[33]:

np.nonzero(a)


# In[34]:

np.flatnonzero(a)


# In[35]:

np.count_nonzero(a)

