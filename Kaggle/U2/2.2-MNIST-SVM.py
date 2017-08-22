
# coding: utf-8

# ## 手写体数据读取

# In[1]:

from sklearn.datasets import load_digits
digits = load_digits()


# In[2]:

digits.data.shape


# ## 手写体数据分割

# In[3]:

from sklearn.cross_validation import train_test_split


# In[4]:

# 随即所选取75%的数据作为训练样本，其余25%的数据作为测试样本
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.25, random_state = 33)


# In[5]:

y_train.shape


# In[6]:

y_test.shape


# ## 使用支持向量机(分类)对手写体数字图像进行识别

# In[8]:

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[9]:

# 对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[10]:

# 初始化线性假设的支持向量机分类器LinearSVC
lsvc = LinearSVC()


# In[12]:

# 进行模型训练
lsvc.fit(X_train, y_train)


# In[13]:

# 利用训练好的模型对测试样本的数字类别进行预测
y_predict = lsvc.predict(X_test)


# ## 支持向量机(分类)模型对手写体数字识别能力的评估

# In[14]:

# 使用模型自带的评估函数进行准确性测评
print('The Accuracy of Linear SVC is', lsvc.score(X_test, y_test))


# In[15]:

# 混淆矩阵模块更详细的分析报告
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names = digits.target_names.astype(str)))


# In[ ]:



