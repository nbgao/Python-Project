# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

data = pd.read_csv('cluster_data.csv', header=0)
data.head()

x = data['x']
y = data['y']
plt.scatter(x,y)
plt.show()

model = k_means(data, n_clusters=3)
print(model)

# 聚类中心数组
cluster_centers = model[0]
# 聚类标签数组 
cluster_labels = model[1]
plt.scatter(x, y, c=cluster_labels)

# 标记聚类中心(五角星)
for center in cluster_centers:
    plt.scatter(center[0], center[1], marker="p", edgecolors="red")

plt.show()



## K值选择与聚类评估
# 横坐标数组
index = []
# 纵坐标数组
inertia = []

# K从1-10聚类
for i in range(1,10):
    model = k_means(data, n_clusters=i)
    index.append(i)
    inertia.append(model[2])
    
plt.plot(index, inertia, "-o")
plt.show()
# K=3时，由肘部系数可知，畸变程度最大

# 导入轮廓系数计算模块
from sklearn.metrics import silhouette_score
# 横坐标
index2 = []
# 轮廓系数列表
silhouette = []

# K从2-10聚类
for i in range(2,9):
    model = k_means(data, n_clusters=i)
    index2.append(i)
    silhouette.append(silhouette_score(data, model[1]))
    
print(silhouette)

plt.plot(index2, silhouette, '-o')
plt.show()
# K=3时轮廓系数越接近于1，聚类效果越好

