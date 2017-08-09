# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster

# 对聚类方法依次命名
cluster_names = ['KMeans', 'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift', 'SpectralClustering', 'AgglomerativeCluster', 'Bierch', 'DBSCAN']

# 确定聚类方法相应参数
cluster_estimators = [
        cluster.KMeans(n_clusters=3),
        cluster.MiniBatchKMeans(n_clusters=3),
        cluster.AffinityPropagation(),
        cluster.MeanShift(),
        cluster.SpectralClustering(n_clusters=3),
        cluster.AgglomerativeClustering(n_clusters=3),
        cluster.Birch(n_clusters=3),
        cluster.DBSCAN()
        ]

# 读取数据集csv文件
data = pd.read_csv('data_blobs.csv', header=0)
X = data[['x', 'y']]
Y = data['class']

# 记录子图数
plot_num = 1

# 不同的聚类方法依次运行
for name, algorithm in zip(cluster_names, cluster_estimators):
    algorithm.fit(X) # 聚类
    
    # 判断方法中是否由labels_参数，并执行不同的命令
    if hasattr(algorithm, 'labels_'):
        algorithm.labels_.astype(np.int)
    else:
        algorithm.predict(X)
    
    # 绘制子图
    plt.subplot(2, len(cluster_estimators)/2, plot_num)
    plt.scatter(data['x'], data['y'], c=algorithm.labels_)
    
    # 判断方法中是否有cluster_centers_参数，并执行不同的命令
    if hasattr(algorithm, 'cluster_centers_'):
        centers = algorithm.cluster_centers_
        plt.scatter(centers[:,0], centers[:,1], marker="p", edgecolors="red")
        
    # 绘制图标题
    plt.title(name)
    plot_num += 1
    
plt.show()
    