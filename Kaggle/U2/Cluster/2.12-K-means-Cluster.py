
# coding: utf-8

# ## K-means聚类算法在手写数字图像数据上的使用示例

# ### 导入数据集并分离训练集和测试集

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 使用pandas分别读取训练数据与测试数据
digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

# 从训练与测试数据集上都分离出64维度的像素特征与1维度的数字目标
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]


# ### KMeans聚类模型训练

# In[3]:

# 从sklearn.cluster中导入KMeans模型
from sklearn.cluster import KMeans

# 初始化KMeans模型，并设置聚类中心数量为10
kmeans = KMeans(n_clusters=10)
kmeans.fit(X_train)

# 逐条判断每个测试图像所属的聚类中心
y_predict = kmeans.predict(X_test)


# ## 使用ARI进行K-means聚类性能评估

# In[4]:

# 从sklearn导入度量函数库metrics
from sklearn import metrics

# 使用ARI进行KMeans聚类性能评估
print(metrics.adjusted_rand_score(y_test, y_predict))


# ## 利用轮廓系数评价不同类簇数量的K-means聚类实例

# In[72]:

# 从sklearn.metrics导入silhouette_score用于计算轮廓系数
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 初始化原始数据点
x1 = np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2 = np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
X = np.array((x1, x2)).T

# 在1号子图做出原始数据点阵的分布
plt.figure(figsize=(8,8))
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Instances')
plt.scatter(x1,x2)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']

clusters = [2, 3, 4, 5, 6, 7, 8, 9]
subplot_counter = 0
sc_scores = []

plt.figure(figsize=(12,16))
for k in clusters:
    subplot_counter += 1
    plt.subplot(4, 2, subplot_counter)
    kmeans_model = KMeans(n_clusters=k).fit(X)
    
    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color=colors[l-1], marker=markers[l-1], ls='None')
    
    plt.xlim([0,10])
    plt.ylim([0,10])
    sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)
    
    # 绘制轮廓系数与不同类簇数量的直观显示图
    plt.title('K=%s, silhouette coefficient=%.03f' % (k, sc_score))
plt.tight_layout()
    
# 绘制轮廓系数与不同类簇数量的关系曲线
plt.figure(figsize=(12,9))
plt.plot(clusters, sc_scores, 'o-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')
plt.xlim((1.5,9.5))
plt.ylim((0,1))
plt.grid()
plt.show()


# ## “肘部”系数观察法

# In[101]:

from scipy.spatial.distance import cdist

# 使用均匀分布函数随机3个簇，每个簇周围10个数据样本
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3.0, 4.0, (2, 10))

# 绘制30个数据样本的分布图像
X = np.hstack((cluster1, cluster2, cluster3)).T
plt.figure(figsize=(8,8))
plt.scatter(X[:,0], X[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# In[102]:

# 测试9种不同聚类中心数量下，每种情况聚类的质量，并作图
K = range(1, 10)
meandistortions = []

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1))/X.shape[0])
    
plt.figure(figsize=(12,8))
plt.plot(K, meandistortions, 'o-', lw=2)
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.grid()
plt.show()

