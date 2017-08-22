
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # 1. Series

# In[2]:

s = pd.Series([1,3,5,np.nan,6,8])
s


# # 2. DataFrame

# In[5]:

dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns = list('ABCD'))
df


# In[7]:

df2 = pd.DataFrame({'A':1,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3]*4, dtype='int32'),
                    'E': pd.Categorical(["test","train","test","train"]),
                    'F': 'foo'})
df2


# In[9]:

df2.dtypes


# # 2. Data watch

# In[10]:

df.head()


# In[11]:

df.tail(3)


# In[12]:

df.index


# In[13]:

df.columns


# In[14]:

df.values


# In[15]:

df.describe()


# In[16]:

df.T


# In[17]:

df.sort_index(axis=1, ascending=False)


# In[19]:

df.sort_index(axis=0, ascending=False)


# In[30]:

df.sort_index(by='B')


# # 4. Choose & Split

# In[31]:

df['A']


# In[32]:

df[0:3]


# In[33]:

df['20130102':'20130104']


# ## Choose by labels

# In[34]:

df.loc[dates[0]]


# In[35]:

df.loc[:,['A','B']]


# In[36]:

df.loc[dates[0], 'A']


# In[38]:

df.at[dates[0], 'A']


# ## Choose by position

# In[39]:

df.iloc[3]


# In[40]:

df.iloc[3:5, 0:2]


# In[41]:

df.iloc[[1,2,3],[0,2]]


# In[42]:

df.iloc[1,1]


# In[43]:

df.iat[1,1]


# ## Boolean index

# In[49]:

df[df.B > 0]


# In[50]:

df[df > 0]


# In[52]:

df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
df2


# In[54]:

df2[df2['E'].isin(['two', 'four'])]


# # 5. Assignment

# In[55]:

s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
s1


# In[58]:

df['F'] = s1
df.at[dates[0], 'A'] = 0
df.iat[0, 1] = 0
df.loc[:, 'D'] = np.array([5]*len(df))
df


# In[59]:

df2 = df.copy()
df2[df2 > 0] = -df2
df2


# # 6. NA items

# In[60]:

df1 = df.reindex(index = dates[0:4], columns = list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
df1


# In[61]:

df1.dropna(how = 'any')


# In[62]:

df1.fillna(value = 5)


# In[63]:

pd.isnull(df1)


# # 6. Numerical calculas

# ## Statistic

# In[64]:

df.mean()


# In[65]:

df.mean(1)


# In[67]:

s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
s


# In[68]:

df.sub(s, axis = 'index')


# ## Functions application

# In[69]:

df.apply(np.cumsum)


# In[70]:

df.apply(lambda x: x.max()-x.min())


# ## Histogram

# In[72]:

s = pd.Series(np.random.randint(0, 7, size=10))
s


# ## Character process

# In[73]:

s = pd.Series(['A', 'B', 'C', 'Aaba', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()


# # 8. Merge

# ## Concat

# In[74]:

df = pd.DataFrame(np.random.randn(10, 4))
df


# In[76]:

pieces = [df[:3], df[3:7], df[7:] ]
pd.concat(pieces)


# ## Join

# In[82]:

left = pd.DataFrame({'key': ['foo','foo'], 'lval':[1,2]})
right = pd.DataFrame({'key':['foo','foo'], 'rval':[4,5]})

left


# In[83]:

right


# In[85]:

pd.merge(left, right, on = 'key')


# ## Append

# In[87]:

df = pd.DataFrame(np.random.randn(8,4), columns = ['A', 'B', 'C', 'D'])
df


# In[88]:

s = df.iloc[3]
df.append(s, ignore_index = True)


# # 9. Groupby

# In[89]:

df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
                          'foo', 'bar', 'foo', 'foo'],
                   'B' : ['one', 'one', 'two', 'three',
                          'two', 'two', 'one', 'three'],
                   'C' : np.random.randn(8),
                   'D' : np.random.randn(8)})
df


# In[90]:

df.groupby('A').sum()


# In[91]:

df.groupby(['A','B']).sum()


# ## Pivot table

# In[92]:

df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
                   'B' : ['A', 'B', 'C'] * 4,
                   'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D' : np.random.randn(12),
                   'E' : np.random.randn(12)})
df


# In[93]:

pd.pivot_table(df, values = 'D', index = ['A', 'B'], columns = ['C'])


# # 10. Time Series

# In[94]:

rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min', how='sum')


# In[97]:

rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts


# In[98]:

ts_utc = ts.tz_localize('UTC')
ts_utc


# In[99]:

ts_utc.tz_convert('US/Eastern')


# In[101]:

rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts


# In[103]:

ps = ts.to_period()
ps


# In[104]:

ps.to_timestamp()


# In[105]:

prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
ts  =pd.Series(np.random.randn(len(prng)), prng)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
ts.head()


# # 11. Category

# In[107]:

df  = pd.DataFrame({"id": [1,2,3,4,5,6], "raw_grade":['a','b','b','a','a','e']})
df["grade"] = df["raw_grade"].astype("category")
df["grade"]


# In[109]:

df["grade"].cat.categories = ["very good", "good", "very bad"]
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])
df["grade"]


# In[113]:

df.sort_values("grade")


# In[114]:

df.groupby("grade").size()


# # 12. Draw picture

# In[120]:

ts = pd.Series(np.random.randn(1000), index = pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()


# In[128]:

df = pd.DataFrame(np.random.randn(1000,4), index=ts.index, columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure()
df.plot()
plt.legend(loc = 'best')
plt.show()


# # 13. Data I/O

# ## CSV

# In[130]:

df.to_csv('Data/foo.csv')


# In[131]:

pd.read_csv('Data/foo.csv')


# ## Excel

# In[133]:

df.to_excel('Data/foo.xlsx', sheet_name='Sheet1')


# In[134]:

pd.read_excel('Data/foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])


# In[ ]:



