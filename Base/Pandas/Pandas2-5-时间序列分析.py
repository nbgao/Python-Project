
# coding: utf-8

# # 1. 时间戳 Timestamp

# In[1]:

import pandas as pd

pd.Timestamp('2017-10-01')


# In[8]:

pd.Timestamp('2017-09-19 20:45:30')


# In[9]:

pd.Timestamp('19/9/2017 20:45:45')


# # 2. 时间索引 DatetimeIndex

# **默认参数** <br>
# pandas.date_range(start=None, end=None, periods=None, freq='D', tz=None, normalize=False, name=None, closed=None, **kwargs)

# In[13]:

rng1 = pd.date_range('19/09/2017', periods=24, freq='H')
rng1


# In[14]:

rng2 = pd.date_range('19/09/2017', periods=10, freq='D')
rng2


# In[16]:

rng3 = pd.date_range('19/09/2017', periods=20, freq='1H20min')
rng3


# # 3. 时间转换 to_datetime

# **默认参数** <br>
# pandas.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, box=True, format=None, exact=True, unit=None, infer_datetime_format=False, orgin='unix')

# ## 3.1 输入标量

# In[17]:

pd.to_datetime('19/9/2017 10:00', dayfirst=True)


# ## 3.2 输入列表

# In[18]:

pd.to_datetime(['19/9/2017 20:59', '20/9/2017 21:00', '21/9/2017 12:00'])


# ## 3.3 输入Series

# In[19]:

pd.to_datetime(pd.Series(['Oct 11, 2017', '2017-11-11', '23/12/2017']), dayfirst=True)


# ## 3.4 输入DataFrame

# In[24]:

pd.to_datetime(pd.DataFrame({'year': [2017, 2018], 'month': [9, 10], 'day': [1, 2], 'hour': [11, 12]}))


# ## 3.5 errors=

# In[26]:

pd.to_datetime(['2017/10/1', 'abc'], errors='ignore')


# In[27]:

pd.to_datetime(['2017/10/1', 'abc'], errors='coerce')


# # 4. 时间序列检索

# In[32]:

import numpy as np

ts = pd.DataFrame(np.random.randn(100000,1), columns=['Value'], index=pd.date_range('20170101', periods=100000, freq='T'))
ts


# In[34]:

ts['2017-09-19 21:00:59': '2017-09-19 21:30:00']


# ## 5. 时间序列计算

# In[37]:

from pandas.tseries import offsets

dt = pd.Timestamp('2017-9-19 21:18:00')

dt + offsets.DateOffset(months=1, days=2, hour=3)


# ## 6. 其他方法

# ## 6.1 移动 Shifting

# In[39]:

ts = pd.DataFrame(np.random.randn(7,2), columns=['Value1', 'Value2'], index=pd.date_range('20170101', periods=7, freq='T'))
ts


# In[40]:

ts.shift(3)


# In[41]:

ts.shift(-3)


# In[42]:

ts.tshift(3)


# ## 6.2 重采样 Resample

# In[46]:

# 生成一个时间序列数据集
ts = pd.DataFrame(np.random.randn(50,1), columns=['Value'], index=pd.date_range('2017-01', periods=50, freq='D'))
ts


# In[47]:

ts.resample('H').ffill()


# In[48]:

ts.resample('5D').sum()


# In[ ]:



