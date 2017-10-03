
# coding: utf-8

# # 1. 自然语言处理包(NLTK)

# ## 使用词带法(Bag of Words)对示例文本进行特征向量化

# In[1]:

# 将2个句子以字符串的数据类型分别存储在变量sent1和sent2中
sent1 = 'The cat is walking in the bedroom.'
sent2 = 'A dog was running across the kitchen.'


# In[2]:

# 从sklearn.feature_extraction.text中导入CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()

sentences = [sent1, sent2]

# 输出特征向量化后的表示
print(count_vec.fit_transform(sentences).toarray())


# In[3]:

print(count_vec.get_feature_names())


# ## 使用NLTK对示例文本进行语言学分析

# In[4]:

# 导入nltk
import nltk


# In[6]:

# 对句子进行词汇分割和正规化
tokens_1 = nltk.word_tokenize(sent1)
print(tokens_1)


# In[7]:

tokens_2 = nltk.word_tokenize(sent2)
print(tokens_2)


# In[8]:

# 整理两句的词表，并且按照ASCII的排序输出
vocab_1 = sorted(set(tokens_1))
print(vocab_1)


# In[9]:

vocab_2 = sorted(set(tokens_2))
print(vocab_2)


# In[10]:

# 初始化stemmer寻找各个词汇最原始的词根
stemmer = nltk.stem.PorterStemmer()
stem_1 = [stemmer.stem(t) for t in tokens_1]
print(stem_1)


# In[11]:

stem_2 = [stemmer.stem(t) for t in tokens_2]
print(stem_2)


# In[13]:

# 初始化词标注器，对每个词汇进行标注
pos_tag_1 = nltk.tag.pos_tag(tokens_1)
print(pos_tag_1)


# In[14]:

pos_tag_2 = nltk.tag.pos_tag(tokens_2)
print(pos_tag_2)

