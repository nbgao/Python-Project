# coding: utf-8

# 导入数据

import pandas as pd
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv(url, names = names)


# 直方图矩阵 Histograms
data.hist()
plt.show()

data.hist(xlabelsize=7, ylabelsize=7, figsize=(8,6))
plt.show()


# 密度图矩阵 Density Plots
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, fontsize=8, figsize=(8,6))
plt.show()


# 箱线图矩阵 Box & Whisker Plots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, fontsize=8, figsize=(8,6))
plt.show()


# 相关系数矩阵 Correlation Matrix Plot
import numpy as np
# 计算变量之间的相关系数矩阵
correlation = data.corr()
# plot correlation matrix
# 调用figure创建一个绘图对象
fig = plt.figure()
ax = fig.add_subplot(111)
# 绘制热力图，从0到1
cax = ax.matshow(correlation, vmin=0, vmax=1)
# 将matshow生成热力图设置为颜色渐变条
fig.colorbar(cax)
# 生成0-9，步长为1
ticks = np.arange(0,9,1)
# 生成刻度
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# 散点图矩阵 Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(data, figsize=(10,10), diagonal='kde')
plt.show()

