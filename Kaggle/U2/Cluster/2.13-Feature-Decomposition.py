
# coding: utf-8

# ## 线性相关矩阵秩计算样例

# In[2]:

import numpy as np

# 定义2阶线性相关矩阵
M1 = np.array([[1,2], [2,4]])
# 定义2阶非线性相关矩阵
M2 = np.array([[3,5], [4,1]])

# 计算2*2线性相关矩阵的秩
M1_rank = np.linalg.matrix_rank(M1, tol=None)
print('The rank of matrix M1 is', M1_rank)

# 计算2*2非线性相关矩阵的秩
M2_rank = np.linalg.matrix_rank(M2, tol=None)
print('The rank of matrix M2 is', M2_rank)


# ## 显示手写体数字图片经PCA压缩后的二维空间分布

# In[5]:

# 导入pandas用于数据读取和处理
import pandas as pd

# 使用pandas分别读取训练数据与测试数据
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

# 从训练与测试数据集上都分离出64维度的像素特征与1维度的数字目标
X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]


# In[15]:

# 从sklearn.decomposition导入PCA
from sklearn.decomposition import PCA

# 初始化一个可以将高维度特征向量(64维)压缩至20个维度的PCA
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)

# 显示10类手写体数字图片经PCA压缩后的2维空间分布
import matplotlib.pyplot as plt

def plot_pca_scatter():
    colors = ['black', 'cyan', 'purple', 'lime', 'green', 'red', 'yellow', 'blue', 'orange', 'pink' ]
    
    plt.figure(figsize=(12,10))
    for i in range(len(colors)):
        px = X_pca[:,0][y_digits.as_matrix() == i]
        py = X_pca[:,1][y_digits.as_matrix() == i]
        plt.scatter(px, py, c=colors[i])
        
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

plot_pca_scatter()


# ## 使用原始像素特征和经PCA压缩重建的低维特征，在相同配置的支持向量机(分类)模型上分别进行图像识别

# In[16]:

# 对训练数据、测试数据进行特征向量(图像像素)与分类目标的分割
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]


# In[35]:

# 导入基于线性核的支持向量机分类器
from sklearn.svm import LinearSVC

# 使用默认配置初始化LinearSVC，对原始64维像素特征的训练数据进行建模，并在测试数据上作出预测
svc = LinearSVC()
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)

# 使用PCA将64维的图像压缩到20个维度
estimator = PCA(n_components=20)


# 利用训练特征决定(fit)20个正交维度的方向，并转化(transform)原训练特征
pca_X_train = estimator.fit_transform(X_train)
# 测试特征也按照上述的20个正交维度方向进行转化
pca_X_test = estimator.transform(X_test)

# 使用默认配置初始化LinearSVC，对压缩过后的20维特征的训练数据进行建模，并在测试数据上作出预测
pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, y_train)
pca_y_predict = pca_svc.predict(pca_X_test)


# ## 原始像素特征与PCA压缩重建的低维特征，在相同配置的支持向量机(分类)模型上识别性能的差异

# In[36]:

# 从sklearn.metrics导入classification_report用于更加细致的分类性能分析
from sklearn.metrics import classification_report

# 对使用原始图像高维像素特征训练的支持向量机分类器的性能作出评估
print(svc.score(X_test, y_test))
print(classification_report(y_test, y_predict, target_names=np.arange(10).astype(str)))

# 对使用PCA压缩重建的低维图像特征训练的支持向量机分类器的性能作出评估
print(pca_svc.score(pca_X_test, y_test))
print(classification_report(y_test, pca_y_predict, target_names=np.arange(10).astype(str)))


# ## 压缩k维特征的训练数据迭代曲线

# In[65]:

x = np.arange(5,64)
y = np.zeros(len(x))

count = 0
for k in range(5,64):
    estimator = PCA(n_components=k)

    # 利用训练特征决定(fit)k个正交维度的方向，并转化(transform)原训练特征
    pca_X_train = estimator.fit_transform(X_train)
    # 测试特征也按照上述的k个正交维度方向进行转化
    pca_X_test = estimator.transform(X_test)

    # 使用默认配置初始化LinearSVC，对压缩过后的20维特征的训练数据进行建模，并在测试数据上作出预测
    pca_svc = LinearSVC()
    pca_svc.fit(pca_X_train, y_train)
    pca_y_predict = pca_svc.predict(pca_X_test)
    
    y[count] = pca_svc.score(pca_X_test, y_test)
    count += 1
    
plt.figure(figsize=(12,8))
plt.plot(x, y, 'o-')
plt.xlim((0,70))
plt.grid()
plt.show()


# In[ ]:



