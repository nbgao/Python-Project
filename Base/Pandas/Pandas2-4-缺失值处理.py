
# coding: utf-8

# # 1. 认识缺失值

# In[1]:

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(5,5), index=list('cafed'), columns=list('ABCDE'))
print(df)


# In[2]:

print(df.reindex(list('abcde')))


# # 2. 检测缺失值

# In[6]:

df2 = df.reindex(list('abcde'))

df2.isnull()
df2.notnull()


# In[9]:

df2.insert(value=pd.Timestamp('2017-10-1'), loc=0, column='T')

df2.loc[['a', 'c', 'e'], ['T']] = np.nan


# In[10]:

df2.loc[['a','b','c'],['T']] = np.nan


# In[15]:

df2.isnull()
df2.notnull()


# # 3. 填充和清除缺失值

# ## 3.1 填充缺失值fillna()

# In[17]:

df = pd.DataFrame(np.random.rand(9,5), columns=list('ABCDE'))

df.insert(value=pd.Timestamp('2017-10-1'), loc=0, column='Time')

df.iloc[[1,3,5,7], [0,2,4]] = np.nan

df.iloc[[2,4,6,8], [1,3,5]] = np.nan


# In[18]:

df.fillna(0)


# In[19]:

df.fillna(method='pad')


# In[20]:

df.fillna(method='bfill')


# In[22]:

df.iloc[[3,5],[1,3,5]] = np.nan
df


# In[23]:

df.fillna(method='pad')


# In[24]:

df.fillna(method='pad', limit=1)


# In[25]:

df.fillna(df.mean()['C':'E'])


# ## 3.2 清除缺失值dropna()

# In[26]:

df.dropna()


# In[28]:

df.dropna(axis=0)


# In[27]:

df.dropna(axis=1)


# # 4. 插值interpolate()

# In[30]:

df = pd.DataFrame({'A': [1.1, 2.2, np.nan, 4.5, 5.7, 6.9], 'B': [.21, np.nan, np.nan, 3.1, 11.7, 13.2]})
df


# In[31]:

df.interpolate()


# In[48]:

import matplotlib.pyplot as plt

y = df.interpolate()

plt.figure(figsize=(8,6))
plt.plot(y)
plt.legend(['A', 'B'])
plt.show()


# In[57]:

y = df.interpolate('quadratic')
plt.figure(figsize=(8,6))
plt.plot(y)
plt.show()


# In[58]:

y = df.interpolate('pchip')
plt.figure(figsize=(8,6))
plt.plot(y)
plt.show()


# In[52]:

y = df.interpolate('akima')
plt.figure(figsize=(8,6))
plt.plot(y)
plt.show()


# In[ ]:



