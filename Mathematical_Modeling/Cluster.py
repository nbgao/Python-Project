# -*- coding: utf-8 -*-
import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import k_means

input_file1 = "../测试数据集/东海.xlsx"
input_file2 = "../测试数据集/杭州湾.xlsx"
input_file3 = "../测试数据集/南海.xlsx"
input_file4 = "../测试数据集/台湾海峡.xlsx"

output_file1 = '东海 K-Means Cluster.xlsx'
output_file2 = '杭州湾 K-Means Cluster.xls'
output_file3 = '南海 K-Means Cluster.xls'
output_file4 = '台湾海峡 K-Means Cluster.xls'


table1 = xlrd.open_workbook("../测试数据集/杭州湾.xlsx")
sheet1 = table1.sheets()[0]

nrows = sheet1.nrows
ncols = sheet1.ncols

''' 东海海表温度 '''
data1 = pd.read_excel(input_file1)
data1_year = data1.ix[:,0]
data1_year.plot()
plt.show()

data1_loc = data1.ix[0,:]
data1_loc.plot()
plt.show()

data1_T = data1.T
# KMeans聚类
model_1 = k_means(data1_T, n_clusters=6)

result1 = model_1[1]
result1_df = pd.DataFrame(result1)


result1_df.to_excel(output_file1, sheet_name='sheet1', index=False)
result1_df.to_excel('东海 K=6.xls', sheet_name='sheet1', index=False)


# 聚类结果
model_1[1]
# 聚类效果值
model_1[2]

'''
from xlwt import Workbook

table1_wt = Workbook()
sheet1_wt = table1_wt.add_sheet('K-Means Cluster 1')

for i in range(0,len(result1)-1):
    sheet1_wt.write(0,i,result1[i])
    
    
table1_wt.save('东海 K-Means Cluster.xls')
'''


''' K值选择与聚类评估 '''
# 横坐标数组
index = []
# 纵坐标数组
inertia = []

# K从1-10聚类
for i in range(1,10):
    model = k_means(data1_T, n_clusters=i)
    index.append(i)
    inertia.append(model[2])
    
plt.plot(index, inertia, "-o")
plt.show()
# K=3时，由肘部系数可知，畸变程度最大

''' 导入轮廓系数计算模块 '''
from sklearn.metrics import silhouette_score
# 横坐标
index2 = []
# 轮廓系数列表
silhouette = []

# K从2-10聚类
for i in range(2,9):
    model = k_means(data1_T, n_clusters=i)
    index2.append(i)
    silhouette.append(silhouette_score(data1_T, model[1]))
    
print(silhouette)

plt.plot(index2, silhouette, '-o')
plt.show()
# K=3时轮廓系数越接近于1，聚类效果越好



''' 其他聚类模型 '''
from sklearn import cluster
# MiniBatchKMeans
M2 = cluster.MiniBatchKMeans(n_clusters=3)
m2 = M2.fit(data1_T)

M3 = cluster.AffinityPropagation(),
M4 = cluster.MeanShift(),
M5 = cluster.SpectralClustering(n_clusters=3),
M6 = cluster.AgglomerativeClustering(n_clusters=3),
M7 = cluster.Birch(n_clusters=3),
M8 = cluster.DBSCAN()




''' 杭州湾海表温度 '''
data2 = pd.read_excel(input_file2)
data2_year = data2.ix[:,0]
data2_year.plot()
plt.show()

data2_loc = data2.ix[0,:]
data2_loc.plot()
plt.show()

data2_T = data2.T
# KMeans聚类
model_2 = k_means(data2_T, n_clusters=2)

result2 = model_2[1]
result2_df = pd.DataFrame(result2)

result2_df.to_excel(output_file2, sheet_name='sheet1', index=False)

''' K值选择与聚类评估 '''
index = []
inertia = []

for i in range(1,10):
    model = k_means(data2_T, n_clusters=i)
    index.append(i)
    inertia.append(model[2])
    
plt.plot(index, inertia, "-o")
plt.show()


''' 导入轮廓系数计算模块 '''
from sklearn.metrics import silhouette_score
index2 = []
silhouette = []

for i in range(2,9):
    model = k_means(data2_T, n_clusters=i)
    index2.append(i)
    silhouette.append(silhouette_score(data2_T, model[1]))
    
print(silhouette)

plt.plot(index2, silhouette, '-o')
plt.show()




''' 南海海表温度 '''
data3 = pd.read_excel(input_file3)
data3_year = data3.ix[:,0]
data3_year.plot()
plt.show()

data3_loc = data3.ix[0,:]
data3_loc.plot()
plt.show()

data3_T = data3.T
# KMeans聚类
model_3 = k_means(data3_T, n_clusters=3)

result3 = model_3[1]
result3_df = pd.DataFrame(result3)


result3_df.to_excel(output_file3, sheet_name='sheet1', index=False)

''' K值选择与聚类评估 '''
index = []
inertia = []


for i in range(1,10):
    model = k_means(data3_T, n_clusters=i)
    index.append(i)
    inertia.append(model[2])
    
plt.plot(index, inertia, "-o")
plt.show()

''' 导入轮廓系数计算模块 '''
from sklearn.metrics import silhouette_score
index2 = []
silhouette = []


for i in range(2,9):
    model = k_means(data3_T, n_clusters=i)
    index2.append(i)
    silhouette.append(silhouette_score(data3_T, model[1]))
    
print(silhouette)

plt.plot(index2, silhouette, '-o')
plt.show()





''' 台湾海峡海表温度 '''
data4 = pd.read_excel(input_file4)
data4_year = data4.ix[:,0]
data4_year.plot()
plt.show()

data4_loc = data4.ix[0,:]
data4_loc.plot()
plt.show()


data4_T = data4.T
# KMeans聚类
model_4 = k_means(data4_T, n_clusters=2)

result4 = model_4[1]
result4_df = pd.DataFrame(result4)


result4_df.to_excel(output_file4, sheet_name='sheet1', index=False)

''' K值选择与聚类评估 '''
index = []
inertia = []


for i in range(1,10):
    model = k_means(data4_T, n_clusters=i)
    index.append(i)
    inertia.append(model[2])
    
plt.plot(index, inertia, "-o")
plt.show()

''' 导入轮廓系数计算模块 '''
from sklearn.metrics import silhouette_score
index2 = []
silhouette = []


for i in range(2,9):
    model = k_means(data4_T, n_clusters=i)
    index2.append(i)
    silhouette.append(silhouette_score(data4_T, model[1]))
    
print(silhouette)

plt.plot(index2, silhouette, '-o')
plt.show()


