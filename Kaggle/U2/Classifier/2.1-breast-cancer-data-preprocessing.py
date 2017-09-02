
# coding: utf-8

# ## 良/恶性乳腺癌肿瘤数据预处理

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

# 创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']


# In[3]:

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names = column_names)


# In[4]:

# 将?替换为标准缺失值表示
data = data.replace(to_replace = '?', value = np.nan)


# In[5]:

# 丢弃带有缺失值的数据
data = data.dropna(how = 'any')


# In[6]:

# 输出data的数据量和维度
data.shape


# In[7]:

data.head()


# ## 准备良/恶性乳腺癌肿瘤训练、测试数据

# In[8]:

from sklearn.cross_validation import train_test_split


# In[9]:

X_train, X_test, y_train, y_test = train_test_split(data[column_names[1:10]], data[column_names[10]], test_size = 0.25, random_state = 33)


# In[10]:

# 查验数据的数量和类别分布
y_train.value_counts()


# In[13]:

y_test.value_counts()


# ## 使用线性分类模型从事良/恶性肿瘤数据预测任务

# In[14]:

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


# In[16]:

# 标准化数据，保证每个维度的特征数据方差为1，均值为0
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[17]:

# 初始化LogisticRegression和SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()


# In[18]:

# 调用LogisticRegression中的fit函数训练模型参数
lr.fit(X_train, y_train)


# In[27]:

# 使用训练好的模型lr对X_test进行预测
lr_y_predict = lr.predict(X_test)


# In[20]:

# 调用SGDClassfier中的fit函数训练模型参数
sgdc.fit(X_train, y_train)


# In[26]:

# 使用训练好的模型sgdc对X_test进行预测
sgdc_y_predict = sgdc.predict(X_test)


# ## 使用线性分类模型从事良/恶性肿瘤预测任务的性能分析

# In[22]:

from sklearn.metrics import classification_report


# In[24]:

# 使用LogisticRegreesion模型自带的评分函数score获得模型的准确性结果
print('Accuracy of LR Classifier:', lr.score(X_test, y_test))


# In[28]:

# 利用classification_report模块获得LogisticRegression其他3个指标的结果
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))


# In[ ]:



