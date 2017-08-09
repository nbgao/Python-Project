# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
new_data = PCA(n_components=1).fit_transform(data)

# 输出原数据
print(data)
# 输出降为后数据
print(new_data)


# 载入数据集
from sklearn import datasets
digits_data = datasets.load_digits()

# 绘制数据集前5个手写数字灰度图
for index, image in enumerate(digits_data.images[:5]):
    plt.subplot(2, 5, index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    
plt.show()


X = digits_data.data
y = digits_data.target

# PCA将数据将为2维
estimator = decomposition.PCA(n_components=2)
reduce_data = estimator.fit_transform(X)

# 建立K-Means并输入数据
model = KMeans(n_clusters=10)
model.fit(reduce_data)

'''
# 计算聚类过程中的决策边界
x_min, x_max = reduce_data[:, 0].min() - 1, reduce_data[:, 0].max() + 1
y_min, y_max = reduce_data[:, 1].min() - 1, reduce_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min, y_max, .05))

result = model.predict(np.c_[xx.ravel(), yy.ravel()])

# 将决策边界绘制绘制出来
result = result.reshape(xx.shape)

plt.contourf(xx, yy, result, cmap=plt.cm.Greys)
plt.scatter(reduce_data[:, 0], reduce_data[:, 1], c=y, s=15)

# 绘制聚类中心点
center = model.cluster_centers_
plt.scatter(center[:, 0], center[:, 1], marker='p', linewidths=2, color='b', edgecolors='w', zorder=20)

# 图像参数设置
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.show()
'''