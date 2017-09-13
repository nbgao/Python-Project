
# coding: utf-8

# # 1. 数据读取与存储

# ## read_csv

# In[3]:

import pandas as pd

df = pd.read_csv("./Data/los_census.csv")
print(df)


# In[4]:

df = pd.read_table("./Data/los_census.txt")
print(df)


# ## read_table

# In[6]:

df = pd.read_table("./Data/los_census.txt", sep=',')
print(df)


# In[9]:

df = pd.read_csv("./Data/los_census.csv", header=1)
print(df)


# In[11]:

# 自定义索引名称
df = pd.read_csv("./Data/los_census.csv", names=['A','B','C','D','E','F','G'])
print(df)


# ## to_json

# In[10]:

df.to_json("./Data/1.json")


# # 2. Head & Tail

# In[12]:

df = pd.read_csv("./Data/los_census.csv")

print(df.head())
print(df.head(7))


# In[13]:

print(df.tail())
print(df.tail(7))


# # 3. Statistic

# ## 3.1 describe()

# In[14]:

df.describe()


# ## 3.2 idxmin() & idxmax()

# In[15]:

print(df.idxmin())
print(df.idxmax())


# ## 3.3 count()

# In[16]:

df.count()


# ## 3.4 value_counts()

# In[19]:

import numpy as np

s = pd.Series(np.random.randint(0, 9, size=100))

print(s)
print(s.value_counts)


# # 4. Math Function

# ## 4.1 sum()

# In[20]:

df.sum()


# ## 4.2 mean()

# In[21]:

df.mean()


# ## 4.3 median()

# In[22]:

df.median()


# ## 4.4 贝塞尔校正样本标准偏差 std()

# In[23]:

df.std()


# ## 4.5 无偏差 var()

# In[24]:

df.var()


# ## 4.6 平均值标准误差 sem()

# In[25]:

df.sem()


# ## 4.7 偏度 skew()

# In[27]:

df.skew()


# ## 4.8 峰度 kurt()

# In[28]:

df.kurt()


# # 5. 标签对齐

# In[26]:

s = pd.Series(data=[1,2,3,4,5], index=['a','b','c','d','e'])

print(s)
print(s.reindex(['e', 'b', 'f', 'd']))


# In[29]:

df = pd.DataFrame(data={'one':[1,2,3], 'two':[4,5,6], 'three':[7,8,9]}, index=['a','b','c'])

print(df)


# In[30]:

print(df.reindex(index=['b','c','a'], columns=['three', 'two', 'one']))


# In[31]:

print(s.reindex(df.index))


# # 6. Sort

# In[32]:

df = pd.DataFrame(data={'one':[1,2,3], 'two':[4,5,6], 'three':[7,8,9], 'four':[10,11,12]}, index=['a','b','c'])

print(df)


# In[34]:

print(df.sort_index())


# In[35]:

print(df.sort_index(ascending=False))


# In[37]:

df = pd.DataFrame(data={'one': [1, 2, 3, 7], 'two': [4, 5, 6, 9], 'three': [7, 8, 9, 2], 'four': [10, 11, 12, 5]}, index=['a', 'c', 'b','d'])

print(df)


# In[38]:

print(df.sort_values(by='three'))


# In[39]:

print(df[['one','two','three','four']].sort_values(by=['one','two']))

