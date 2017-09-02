
# coding: utf-8

# ## 泰坦尼克号乘客数据查验

# In[5]:

import pandas as pd
# 利用pandas的read_csv模块直接从互联网手机泰坦尼克号乘客数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')


# In[6]:

# 观察亲几行数据可以发现：数据种类各异，数值型、类别型，甚至还有缺失数据
titanic.head()


# In[7]:

titanic.shape


# In[8]:

# 使用pandas，数据都转入pandas独有的dataframe格式，直接用info()查看数据的统计信息
titanic.info()


# ## 使用决策树模型预测泰坦尼克号乘客的生还情况

# In[9]:

# 选取 pclass, age, sex 作为决定幸免与否的特征
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']


# In[10]:

# 对当前的特征进行探查
X.info()


# ### 数据预处理
# ** 1. age 数据列只有633个，需要补全 **  
# ** 2. sex 与 pclass 两个数据列的值都是类别型的，需要转化为数值特征，用0/1代替 **

# In[14]:

# 缺失值用平均值替代
X['age'].fillna(X['age'].mean())

# 对补完的数据重新探查
X.info()


# In[15]:

# 数据分割
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


# In[17]:

# 使用sklearn.feature_extraction中的特征转换器
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)


# In[20]:

# 将类别型的特征单独剥离出来，独成一列特征，数值型的保持不变
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)


# In[21]:

# 同样需要对测试数据的特征进行转换
X_test = vec.transform(X_test.to_dict(orient='record'))
print(X_test)


# In[22]:

# 从sklearn.tree中导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
# 使用分割得到的数据进行模型学习
dtc.fit(X_train, y_train)
# 用训练好的决策树模型对测试特征数据进行预测
y_predict = dtc.predict(X_test)


# ## 决策树模型对泰坦尼克号乘客是否生还的预测

# In[23]:

# 从sklearn.metrics导入classification_report
from sklearn.metrics import classification_report
# 输出预测准确性
print('The accuracy of Decision Tree is', dtc.score(X_test, y_test))


# In[24]:

# 输出更加详细的分类性能
print(classification_report(y_predict, y_test, target_names = ['died', 'survived']))


# In[ ]:



