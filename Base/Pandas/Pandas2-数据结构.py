
# coding: utf-8

# # 1. 一维数据 Series

# ## 1.1 Dictionary -> Series

# In[3]:

import pandas as pd

d = {'a': 10, 'b': 20, 'c': 30}
print(d)


# In[4]:

s = pd.Series(d, index=['b', 'c', 'd', 'a'])
print(s)


# ## 1.2 ndarray -> Series

# In[8]:

import numpy as np

data = np.random.randn(5)    # 一维随机数
index = ['a', 'b', 'c', 'd', 'e']    # 指定索引

s = pd.Series(data, index)
print(s)


# In[9]:

s['a']


# In[10]:

# 默认数字索引
s = pd.Series(data)
print(s)


# In[11]:

s[4]


# In[12]:

2*s


# In[13]:

s - s


# # 2. 二维数据 DataFrame

# ## 2.1 Series dictionary -> DataFrame

# In[14]:

# 带Series的字典
d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']), 'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print(df)


# ## 2.2 ndarrays/lists dictionary -> DataFrame

# In[15]:

d = {'one': [1,2,3,4], 'two': [4,3,2,1]}

df1 = pd.DataFrame(d)    # 未制定索引
df2 = pd.DataFrame(d, index=['a','b','c','d'])    # 指定索引

print(df1)
print(df2)


# ## 2.3 Dictionary list -> DataFrame

# In[17]:

# 带字典的列表
d = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c':20}]

df = pd.DataFrame(d)
print(df)


# ## 2.4 DataFrame.from method

# In[23]:

d = [('A', [1,2,3]), ('B', [4,5,6])]
c = ['one', 'two', 'three']

df = pd.DataFrame.from_items(d, orient='index', columns=c)
print(df)


# ## 2.5 Columns select, add, delete

# In[24]:

print(df['one'])


# In[25]:

df.pop('one')
print(df)


# In[28]:

df.insert(2, 'four', [10,20])
print(df)


# # 3. 三维数据 Panel

# ## 3.1 Panel data

# - 截面数据
# - 时间序列数据
# - 面板数据

# ## 3.2 Panel structrue

# In[30]:

wp = pd.Panel(np.random.randn(2,5,4), items=['Item1', 'Item2'], major_axis=pd.date_range('1/1/2000', periods=5), minor_axis=['A','B','C','D'])
print(wp)


# In[31]:

print(wp['Item1'])


# In[32]:

print(wp['Item2'])


# In[33]:

print(wp.to_frame())


# In[ ]:



