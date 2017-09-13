
# coding: utf-8

# # 1. Grid Search

# ## 使用单线程对文本分类的朴素贝叶斯模型的超参数模型执行组合网格搜索

# In[1]:

# 从sklearn.datasets中导入20类新闻文本抓取器
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# 使用新闻抓取器从互联网上下载所有数据
news = fetch_20newsgroups(subset='all')


# In[2]:

# 从sklearn.cross_validation导入train_test_split用来分割数据
from sklearn.cross_validation import train_test_split

# 对前3000条新闻文本进行数据分割，25%的文本用于测试
X_train, X_test, y_train, y_test = train_test_split(news.data[:3000], news.target[:3000], test_size=0.25, random_state=33)


# In[13]:

# 导入支持向量机(分类)模型
from sklearn.svm import SVC
# 导入TfidfVectorizer文本抽取器
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入Pipeline
from sklearn.pipeline import Pipeline

# 使用Pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])

# 需要试验2个的超参数的个数分别是4、3，svc_gamma的参数共有10^-2,10^-1,...
# 一共有12种超参数组合，12个不同参数下的模型
parameters = {'svc__gamma': np.logspace(-2,1,4), 'svc__C': np.logspace(-1,1,3)}


# 从sklearn.grid_search中导入网格搜索模块GridSearchCV
from sklearn.grid_search import GridSearchCV

# 将12组参数组合以及初始化的Pipeline包括3折交叉验证的要求全部告知GridSearchCV
# 注意设定refit=True
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)

# 执行单线程网格搜索
get_ipython().magic('time _ = gs.fit(X_train, y_train)')
gs.best_params_, gs.best_score_

# 输出最佳模型在测试集上的准确性
print(gs.score(X_test, y_test))


# ## 2. 并行搜索

# In[14]:

# 将12组参数组合以及初始化的Pipeline包括3折交叉验证的要求全部告知GridSearchCV
# 注意设定refit=True
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3, n_jobs=-1)

# 执行单线程网格搜索
get_ipython().magic('time _ = gs.fit(X_train, y_train)')
gs.best_params_, gs.best_score_

# 输出最佳模型在测试集上的准确性
print(gs.score(X_test, y_test))

