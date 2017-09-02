
# coding: utf-8

# ## 读取Iris数据集细节资料

# In[1]:

# 从sklearn.datasets导入iris数据集
from sklearn.datasets import load_iris


# In[2]:

iris = load_iris()

iris.data.shape


# In[4]:

print(iris.DESCR)


# ## 对Iris数据集进行分割

# In[5]:

from sklearn.cross_validation import train_test_split
# 使用train_test_split，利用随即种子random_state采样25%的数据作为测试集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)


# ## 使用K近邻分类器对鸢尾花(Iris)数据进行类别预测

# In[6]:

# 从sklearn.preprocessing里选择导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 从sklearn.neighbors里导入KNeighborsClassifier，即K近邻分类器
from sklearn.neighbors import KNeighborsClassifier


# In[7]:

# 对训练和测试的特征数据进行标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[11]:

# 使用K近邻分类器对测试数据进行类别预测，预测结果储存在变量y_predict中
knc = KNeighborsClassifier()
knc = knc.fit(X_train, y_train)
y_predict = knc.predict(X_test)


# ## 对K近邻分类器在鸢尾花(Iris)数据上的预测性能进行评估

# In[12]:

# 使用模型自带的评估函数进行准确性评测
print('The accuracy of K-Nearest Neighbor Classifier is', knc.score(X_test, y_test))


# In[13]:

# 使用sklearn.metrics里的classification_report模块对预测结果做更加详细的分析
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predict, target_names = iris.target_names))


# In[ ]:



