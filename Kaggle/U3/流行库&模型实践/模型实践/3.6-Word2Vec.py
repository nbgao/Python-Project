
# coding: utf-8

# # 2.词向量(Word2vec)技术

# ## 用20类新闻文本(20newsgroups)进行词向量训练

# In[1]:

# 从sklearn.datasets导入20类新闻文本抓取器
from sklearn.datasets import fetch_20newsgroups

# 通过互联网即时下载数据
news = fetch_20newsgroups(subset='all')
X, y = news.data, news.target


# In[2]:

# 从bs4导入Beautifulsoup
from bs4 import BeautifulSoup
# 导入nltk和re工具包
import nltk, re


# In[3]:

# 定义一个函数名为news_to_sentences将每条新闻中的句子逐一剥离出来，并返回一个句子的列表
def news_to_sentences(news):
    news_text = BeautifulSoup(news).get_text()
    tokenizer = nltk.data.load('nltk:tokenizer/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(news_text)
    sentences = []
    for sent in raw_sentences:
        sentences.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())
    return sentences


# In[5]:

# 将长篇新闻文本中的句子剥离出来用于训练
sentences = []
for x in X:
    sentences += news_to_sentences(x)


# In[9]:

# 从gensim.models里导入word2vec
from gensim.models import word2vec

# 配置词向量的维度
num_features = 300
# 保证被考虑的词汇的维度
min_word_count = 20
# 设定并行化训练使用CPU计算核心的数量，多核可用
num_workers = 2
# 定义训练词向量的上下文窗口大小
context = 5
downsampling = 1e-3


# In[10]:

# 训练词向量模型
model = word2vec.Word2Vec(sentences,
                          workers=num_workers,
                          size=num_features,
                          min_count = min_word_count,
                          window=context, sample=downsampling)


# In[11]:

# 这个设定代表当前训练好的词向量模型为最终版，也可以加快模型的训练速度
model.init_sims(replace=True)


# In[12]:

# 利用训练好的模型，寻找训练文本中与morning最相关的10个词汇
model.most_similar('morning')

