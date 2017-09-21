
# coding: utf-8

# # 1. Numpy

# ## 1.1 从列表或元组转换

# numpy.array(object, dtype=None, copy=True, order=None, subok=False, ndim=0)

# In[4]:

import numpy as np

np.array([[[1,2,3], [1,2,3], [1,2,3]], [[1,2,3], [1,2,3], [1,2,3]], [[1,2,3], [1,2,3], [1,2,3]]])


# In[5]:

np.array([(1,2), (3,4), (5,6)])


# ## 1.2 arange 方法创建

# numpy.arange(start, stop, step, dtype=None)

# In[7]:

np.arange(3, 7, 0.5, dtype=float)


# ## 1.3 linspace 方法创建

# numpy.linspace(start, stop, step, num=50, endpoint=True, retstep=False, dtype=None)

# In[8]:

np.linspace(0, 10, 10, endpoint=True)


# In[9]:

np.linspace(0, 10, 10, endpoint=False)


# In[10]:

np.linspace(0, 10, 10)


# ## 1.4 ones 方法创建

# numpy.ones(shape, dtype=None, order='C')

# In[11]:

np.ones((2,3))


# ## 1.5 zeros 方法创建

# numpy.zeros(shape, dtpye=None, order='C')

# In[12]:

np.zeros((3,4))


# ## 1.6 eye 方法创建

# In[14]:

np.eye(3)


# In[13]:

np.eye(5, 4, 1)


# In[15]:

np.eye(4, 6, -1)


# ## 1.7 从已知数据创建

# np.fromfunction(lambda a, b: a+b, (5,4))

# # 2. ndarray 数组属性

# In[16]:

a = np.array([[[1,2,3], [1,2,3], [1,2,3]], [[1,2,3], [1,2,3], [1,2,3]], [[1,2,3], [1,2,3], [1,2,3]]])
a


# ## 2.1 ndarray.T

# In[17]:

a.T


# ## 2.2 ndarray.dtype

# In[18]:

a.dtype


# ## 2.3 ndarray.imag

# In[19]:

a.imag


# ## 2.4 ndarray.real

# In[20]:

a.real


# ## 2.5 ndarray.size

# In[21]:

a.size


# ## 2.6 ndarray.itemsize

# In[22]:

a.itemsize


# ## 2.7 ndaray.nbytes

# In[24]:

a.nbytes


# ## 2.8 ndarray.ndim

# In[23]:

a.ndim


# ## 2.9 ndarray.shape

# In[25]:

a.shape


# ## 2.10 ndarray.strides

# In[26]:

a.strides


# In[ ]:



