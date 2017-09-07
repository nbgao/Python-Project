
# coding: utf-8

# # 1. Feature extraction 特征抽取

# ## DictVectorizer对使用字典存储的数据进行特征抽取与向量化

# In[1]:

# 定义一组字典列表，用来表示多个数据样本(每个字典代表一个数据样本)
measurements = [{'city': 'Dubai', 'temperature': 33.}, {'city': 'London', 'temperature': 12.}, {'city': 'San Fransisco', 'temperature': 18.}]

# 从sklearn.feature_extraction导入DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# 初始化DictVectorizer特征提取器
vec = DictVectorizer()
# 输出转化之后的特征矩阵
print(vec.fit_transform(measurements).toarray())
# 输出各个维度的特征含义
print(vec.get_feature_names())


# ## 使用CountVectorizer并且不去掉停用词的条件下，对文本特征进行量化的朴素贝叶斯分类性能测试

# In[2]:

# 从sklearn.datasets里导入20类新闻文本数据抓取器
from sklearn.datasets import fetch_20newsgroups

# 从互联网上即时下载新闻样本，subset='all'参数代表下载全部近2万条文本存储在变量news中
news = fetch_20newsgroups(subset='all')

# 从sklearn.cross_validation导入train_test_split模块用于分割数据集
from sklearn.cross_validation import train_test_split

# 对news中的数据进行分割，75%的文本用作训练集，25%的文本用作测试集
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

# 从sklearn.feature_extraction.text里导入CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# 采用默认的配置对CountVectorizer进行初始化(默认配置不去除英文停用词)，并且赋值给变量count_vec
count_vec = CountVectorizer()

# 只使用词频统计的方式将原始训练和测试文本转化为特征向量
X_count_train = count_vec.fit_transform(X_train)
X_count_test = count_vec.transform(X_test)


# In[6]:

# 从sklearn.naive_bayes里导入朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB

# 使用默认的配置对分类器进行初始化
mnb_count = MultinomialNB()

# 使用朴素贝叶斯分类器，对CountVectorizer(不去除停用词)后的训练样本进行参数学习
mnb_count.fit(X_count_train, y_train)

# 输出模型准确性结果
print('The accuracy of classifying 20newsgroups using Naive Bayes (CountVectorizer without filtering stopwords):',
      mnb_count.score(X_count_test, y_test))


# In[7]:

#  将分类预测的结果存储在变量y_count_predict中
y_count_predict = mnb_count.predict(X_count_test)

# 从sklearn.metrics导入classification_report
from sklearn.metrics import classification_report

# 输出更加详细的其他评价分类性能的指标
print(classification_report(y_test, y_count_predict, target_names=news.target_names))


# ## 使用TfidfVectorizer并且不去掉停用词的条件下，对文本特征进行量化的朴素贝叶斯分类性能测试

# In[9]:

# 从sklearn.feature_extraction.text里导入TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 采用默认的配置对TfidfVectorizer进行初始化(默认配置不去除英文应用词)，并且赋值给变量tfidf_vec
tfidf_vec = TfidfVectorizer()

# 使用tfidf的方式，将原始训练和测试文本转化为特征向量
X_tfidf_train = tfidf_vec.fit_transform(X_train)
X_tfidf_test = tfidf_vec.transform(X_test)

# 依然使用默认配置的朴素贝叶斯分类器，在相同的训练和测试数据上，对新的特征量化方式进行性能评估
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tfidf_train, y_train)

print('The accuracy of classifying 20newsgroups with Naive Bayes (TfidfVectorizer without filtering stopwords):',
      mnb_tfidf.score(X_tfidf_test, y_test))


# In[10]:

y_tfidf_predict = mnb_tfidf.predict(X_tfidf_test)
print(classification_report(y_test, y_tfidf_predict, target_names=news.target_names))


# ## 分别使用CountVectorizer与TfidfVectorizer，并且去掉停用词的条件下，对文本特征进行量化的朴素贝叶斯分类性能测试

# In[12]:

# 分别使用停用词过滤配置初始化CountVectorizer和TfidfVectorizer
count_filter_vec = CountVectorizer(analyzer='word', stop_words='english')
tfidf_filter_vec = TfidfVectorizer(analyzer='word', stop_words='english')

# 使用带有停用词过滤的CountVectorizer对训练和测试文本分别进行量化处理
X_count_filter_train = count_filter_vec.fit_transform(X_train)
X_count_filter_test = count_filter_vec.transform(X_test)

# 使用带有停用词过滤的TfidfVectorizer对训练和测试文本分别进行量化处理
X_tfidf_filter_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test = tfidf_filter_vec.transform(X_test)


# In[17]:

# 初始化默认配置的朴素贝叶斯分类器，并对CountVectorizer后的数据进行预测与准确性评估
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, y_train)

print('The accuracy of classifying 20newsgroups using Naive Bayes (CountVectorizer by filter stopwords):',
      mnb_count_filter.score(X_count_filter_test, y_test))
y_count_filter_predict = mnb_count_filter.predict(X_count_filter_test)


# In[18]:

# 初始化默认配置的朴素贝叶斯分类器，并对TfidfVectorizer后的数据进行预测与准确性评估
mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train, y_train)

print('The accuracy of classifying 20newsgroups using Naive Bayes (TfidfVectorizer by filter stopwords):',
      mnb_tfidf_filter.score(X_tfidf_filter_test, y_test))
y_tfidf_filter_predict = mnb_tfidf_filter.predict(X_tfidf_filter_test)


# In[19]:

# 对上述两个模型进行更加详细的性能评估
from sklearn.metrics import classification_report

print(classification_report(y_test, y_count_filter_predict, target_names=news.target_names))

print(classification_report(y_test, y_tfidf_filter_predict, target_names=news.target_names))


# In[ ]:



