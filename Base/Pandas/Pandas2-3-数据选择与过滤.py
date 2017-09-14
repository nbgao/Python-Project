
# coding: utf-8

# # 1. 基于索引数字选择

# In[1]:

import pandas as pd

df = pd.read_csv("./Data/los_census.csv")
print(df.head())


# In[3]:

print(df.iloc[:3])


# In[4]:

print(df.iloc[5])


# In[7]:

print(df.iloc[[1,3,5]])


# In[8]:

print(df.iloc[:, 1:4])


# # 2. 基于标签名称选择

# In[9]:

import numpy as np

df = pd.DataFrame(np.random.randn(6,5), index=list('abcdef'), columns=list('ABCDE'))
print(df)


# In[10]:

print(df.loc['a': 'c'])


# In[11]:

print(df.loc[['a', 'c', 'd']])


# In[12]:

print(df.loc[:, 'B':'D'])


# In[13]:

print(df.loc[['a','c'], 'C':])


# # 3. 数据随机取样

# In[14]:

s = pd.Series([0,1,2,3,4,5,6,7,8,9])
print(s.sample)


# In[15]:

print(s.sample(n=5))


# In[18]:

df = pd.DataFrame(np.random.randn(6,5), index=list('abcdef'), columns=list('ABCDE'))
print(df)
print(df.sample(n=3))


# In[19]:

df.sample(n=3, axis=1)


# # 4. 条件语句选择

# In[23]:

s = pd.Series(range(-5,5))

print(s)
print(s[(s<-2) | (s>1)])


# In[24]:

df = pd.DataFrame(np.random.randn(6,5), index=list('abcdef'), columns=list('ABCDE'))

print(df)
print(df[(df['B']>0) | (df['D']<0)])


# # 5. where()方法选择

# In[25]:

df = pd.DataFrame(np.random.randn(6,5), index=list('abcdef'), columns=list('ABCDE'))

print(df)
print(df.where(df<0))


# In[26]:

print(df.where(df<0,-df))


# # 6. query()方法选择

# In[27]:

df = pd.DataFrame(np.random.rand(10, 5), columns=list('abcde'))

print(df)
print(df.query('(a<b) & (b<c)'))


# In[28]:

print(df[(df.a < df.b) & (df.b < df.c)])


# In[ ]:



