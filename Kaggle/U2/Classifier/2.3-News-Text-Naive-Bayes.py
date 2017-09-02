
# coding: utf-8

# ## 读取20类新闻文本的数据细节

# In[2]:

from sklearn.datasets import fetch_20newsgroups


# In[3]:

# fetch_20newsgroups需要即时从互联网下载数据
news = fetch_20newsgroups(subset='all')


# In[16]:

# 查验数据规模和细节
print(len(news.data))
print(news.data[0])
print(news.data[1])
print(news.data[2])
print(news.data[3])


# ## 20类新闻文本数据分割

# In[5]:

# 导入train_test_split
from sklearn.cross_validation import train_test_split


# In[6]:

# 随即采样25%的数据样本作为测试集
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)


# ## 使用朴素贝叶斯分类器对新闻文本数据进行类别预测

# In[7]:

# 从sklearn.feature_extraction.text里导入用于文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer


# In[8]:

vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)


# In[9]:

# 从sklearn.naive_bayes里导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB


# In[10]:

# 从使用默认配置初始化贝叶斯模型
mnb = MultinomialNB()
# 利用训练数据对模型参数进行估计
mnb.fit(X_train, y_train)
# 对测试样本进行类别预测，结果存储在y_predict中
y_predict = mnb.predict(X_test)


# In[11]:

len(y_predict)


# ## 对朴素贝叶斯分类器在新闻文本数据上的表现性能进行评估

# In[19]:

# 使用模型自带的评估函数进行准确性测评
print('The accuracy of Naive Bayes Classifier is', mnb.score(X_test, y_test))


# In[20]:

# 从sklearn.metrics里导入classification_report用于详细的分类性能报告
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names=news.target_names))


# In[ ]:



