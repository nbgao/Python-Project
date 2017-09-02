
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

# 利用pandas的read_csv模块直接从互联网手机泰坦尼克号乘客数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')


# In[3]:

# 人工选取 pclass age sex 作为判别乘客是否能够生还的特征
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']


# In[4]:

# 补全缺失值
X['age'].fillna(X['age'].mean(), inplace=True)


# In[5]:

# 对原始数据进行分割，25%的乘客数据用于测试
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


# In[6]:

# 对类别型特征进行转化，成为特征向量
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record`'))


# ### DecisionTreeClassifier (DTC)

# In[7]:

# 使用单一决策树模型进行模型训练以及预测分析
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_y_predict = dtc.predict(X_test)


# ### RandomForestClassier (RFC)

# In[8]:

# 使用随机森林分类器进行集成模型的训练及预测分析
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)


# ### GradientBoostingClassifier (GBC)

# In[9]:

# 使用梯度提升决策树进行集成模型的训懒以及预测分析
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_y_predict = gbc.predict(X_test)


# ## 集成模型对泰坦尼克号乘客是否生还的预测性能

# In[10]:

# 从sklearn.metrices 导入classification_report
from sklearn.metrics import classification_report


# ### DecisionTreeClassifier (DTC)

# In[11]:

# 单一决策树分类性能
print('The accuracy of decision tree is', dtc.score(X_test, y_test))
print(classification_report(dtc_y_predict, y_test))


# ### RandomForestClassier (RFC)

# In[12]:

# 随机森林分类器分类性能
print('The accuracy of random forest classifier is', rfc.score(X_test, y_test))
print(classification_report(rfc_y_predict, y_test))


# ### GradientBoostingClassifier (GBC)

# In[13]:

# 梯度提升决策树分类性能
print('The accuracy of gradient tree boosting is', gbc.score(X_test, y_test))
print(classification_report(gbc_y_predict, y_test))


# In[ ]:



