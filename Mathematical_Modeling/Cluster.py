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

output_file1 = '东海 K-Means Cluster.xls'
output_file2 = '杭州湾 K-Means Cluster.xls'
output_file3 = '南海 K-Means Cluster.xls'
output_file4 = '台湾海峡 K-Means Cluster.xls'

output_file1_1 = '东海 Clustering.xls'
output_file2_1 = '杭州湾 Clustering.xls'
output_file3_1 = '南海 Clustering.xls'
output_file4_1 = '台湾海峡 Clustering.xls'


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
model_1 = k_means(data1_T, n_clusters=3)

result1 = model_1[1]
result1_df = pd.DataFrame(result1)


result1_df.to_excel(output_file1, sheet_name='sheet1', index=False)
# result1_df.to_excel('东海 K=6.xls', sheet_name='sheet1', index=False)


# 聚类结果
model_1[1]
# 聚类效果值
model_1[2]

# 用TSNE进行数据降维并展示聚类结果
from sklearn.manifold import TSNE
tsne = TSNE()
tsne.fit_transform(data1_T)
tsne = pd.DataFrame(tsne.embedding_, index = data1_T.index)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

d1 = tsne[model_1[1] == 0]
plt.plot(d1[0], d1[1], 'r.')
d2 = tsne[model_1[1] == 1]
plt.plot(d2[0], d2[1], 'go')
d3 = tsne[model_1[1] == 2]
plt.plot(d3[0], d3[1], 'b*')
plt.show()
len(d1), len(d2), len(d3)


''' 时间序列聚类模式图 '''
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
c1 = c2 = c3 = 0
# 第1类模式图
ax1 = plt.subplot(311)
for i in range(0,len(data1.columns)):
    if (model_1[1][i] == 0):
        data1.ix[:,i].plot()   
        c1 += 1
ax1.set_title('第1类'+str(c1)+'个观测点')
# 第2类模式图
ax2 = plt.subplot(312)
for i in range(0,len(data1.columns)):
    if (model_1[1][i] == 1):
        data1.ix[:,i].plot()
        c2 += 1
ax2.set_title('第2类'+str(c2)+'个观测点')
# 第3类模式图   
ax3 = plt.subplot(313)
for i in range(0,len(data1.columns)):
    if (model_1[1][i] == 2):
        data1.ix[:,i].plot()
        c3 += 1
ax3.set_title('第3类'+str(c3)+'个观测点')

plt.tight_layout()
plt.show()
(c1,c2,c3)


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
# K-Means
M1 = cluster.KMeans(n_clusters=3)
m1 = M1.fit(data1_T)
m1.labels_

# MiniBatchKMeans
M2 = cluster.MiniBatchKMeans(n_clusters=3)
m2 = M2.fit(data1_T)
m2.labels_

# AffinityPropagation
M3 = cluster.AffinityPropagation()
m3 = M3.fit(data1_T)
m3.labels_

# MeanShift
M4 = cluster.MeanShift()
m4 = M4.fit(data1_T)
m4.labels_

# SpectralClustering
M5 = cluster.SpectralClustering(n_clusters=3)
m5 = M5.fit(data1_T)
m5.labels_

# AgglomerativeClustering
M6 = cluster.AgglomerativeClustering(n_clusters=3)
m6 = M6.fit(data1_T)
m6.labels_

# Birch
M7 = cluster.Birch(n_clusters=3)
m7 = M7.fit(data1_T)
m7.labels_

# DBSCAN
M8 = cluster.DBSCAN()
m8 = M8.fit(data1_T)
m8.labels_


# 写入Excel
from xlwt import Workbook
table1_wt = Workbook()
sheet1_wt = table1_wt.add_sheet('sheet1')

sheet1_wt.write(0,0,"K-Means")
for i in range(0,len(m1.labels_)):
    sheet1_wt.write(i+1,0,int(m1.labels_[i]))

sheet1_wt.write(0,1,"MiniBatchKMeans")
for i in range(0,len(m2.labels_)):
    sheet1_wt.write(i+1,1,int(m2.labels_[i]))

sheet1_wt.write(0,2,"AffinityPropagation")
for i in range(0,len(m3.labels_)):
    sheet1_wt.write(i+1,2,int(m3.labels_[i]))

sheet1_wt.write(0,3,"MeanShift")
for i in range(0,len(m4.labels_)):
    sheet1_wt.write(i+1,3,int(m4.labels_[i]))

sheet1_wt.write(0,4,"SpectralClustering")
for i in range(0,len(m5.labels_)):
    sheet1_wt.write(i+1,4,int(m5.labels_[i]))

sheet1_wt.write(0,5,"AgglomerativeClustering")
for i in range(0,len(m6.labels_)):
    sheet1_wt.write(i+1,5,int(m6.labels_[i]))

sheet1_wt.write(0,6,"Brich")
for i in range(0,len(m7.labels_)):
    sheet1_wt.write(i+1,6,int(m7.labels_[i]))

sheet1_wt.write(0,7,"DBSCAN")
for i in range(0,len(m8.labels_)):
    sheet1_wt.write(i+1,7,int(m8.labels_[i]))

table1_wt.save(output_file1_1)



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


''' 时间序列聚类模式图 '''
c1 = c2 = 0
# 第1类模式图
ax1 = plt.subplot(211)
for i in range(0,len(data2.columns)):
    if (model_2[1][i] == 0):
        data2.ix[:,i].plot()   
        c1 += 1
ax1.set_title('第1类'+str(c1)+'个观测点')
# 第2类模式图
ax2 = plt.subplot(212)
for i in range(0,len(data2.columns)):
    if (model_2[1][i] == 1):
        data2.ix[:,i].plot()
        c2 += 1
ax2.set_title('第2类'+str(c2)+'个观测点')

plt.tight_layout()
plt.show()
(c1,c2)


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


''' 其他聚类模型 '''
from sklearn import cluster
# K-Means
M1 = cluster.KMeans(n_clusters=2)
m1 = M1.fit(data2_T)
m1.labels_

# MiniBatchKMeans
M2 = cluster.MiniBatchKMeans(n_clusters=2)
m2 = M2.fit(data2_T)
m2.labels_

# AffinityPropagation
M3 = cluster.AffinityPropagation()
m3 = M3.fit(data2_T)
m3.labels_

# MeanShift
M4 = cluster.MeanShift()
m4 = M4.fit(data2_T)
m4.labels_

# SpectralClustering
M5 = cluster.SpectralClustering(n_clusters=2)
m5 = M5.fit(data2_T)
m5.labels_

# AgglomerativeClustering
M6 = cluster.AgglomerativeClustering(n_clusters=2)
m6 = M6.fit(data2_T)
m6.labels_

# Birch
M7 = cluster.Birch(n_clusters=2)
m7 = M7.fit(data2_T)
m7.labels_

# DBSCAN
M8 = cluster.DBSCAN()
m8 = M8.fit(data2_T)
m8.labels_


# 写入Excel
from xlwt import Workbook
table1_wt = Workbook()
sheet1_wt = table1_wt.add_sheet('sheet1')

sheet1_wt.write(0,0,"K-Means")
for i in range(0,len(m1.labels_)):
    sheet1_wt.write(i+1,0,int(m1.labels_[i]))

sheet1_wt.write(0,1,"MiniBatchKMeans")
for i in range(0,len(m2.labels_)):
    sheet1_wt.write(i+1,1,int(m2.labels_[i]))

sheet1_wt.write(0,2,"AffinityPropagation")
for i in range(0,len(m3.labels_)):
    sheet1_wt.write(i+1,2,int(m3.labels_[i]))

sheet1_wt.write(0,3,"MeanShift")
for i in range(0,len(m4.labels_)):
    sheet1_wt.write(i+1,3,int(m4.labels_[i]))

sheet1_wt.write(0,4,"SpectralClustering")
for i in range(0,len(m5.labels_)):
    sheet1_wt.write(i+1,4,int(m5.labels_[i]))

sheet1_wt.write(0,5,"AgglomerativeClustering")
for i in range(0,len(m6.labels_)):
    sheet1_wt.write(i+1,5,int(m6.labels_[i]))

sheet1_wt.write(0,6,"Brich")
for i in range(0,len(m7.labels_)):
    sheet1_wt.write(i+1,6,int(m7.labels_[i]))

sheet1_wt.write(0,7,"DBSCAN")
for i in range(0,len(m8.labels_)):
    sheet1_wt.write(i+1,7,int(m8.labels_[i]))

table1_wt.save(output_file2_1)




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


''' 时间序列聚类模式图 '''
c1 = c2 = c3 = 0
# 第1类模式图
ax1 = plt.subplot(311)
for i in range(0,len(data3.columns)):
    if (model_3[1][i] == 0):
        data3.ix[:,i].plot()   
        c1 += 1
ax1.set_title('第1类'+str(c1)+'个观测点')
# 第2类模式图
ax2 = plt.subplot(312)
for i in range(0,len(data3.columns)):
    if (model_3[1][i] == 1):
        data3.ix[:,i].plot()
        c2 += 1
ax2.set_title('第2类'+str(c2)+'个观测点')
# 第3类模式图   
ax3 = plt.subplot(313)
for i in range(0,len(data3.columns)):
    if (model_3[1][i] == 2):
        data3.ix[:,i].plot()
        c3 += 1
ax3.set_title('第3类'+str(c3)+'个观测点')
plt.tight_layout()
plt.show()
(c1,c2,c3)


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


''' 其他聚类模型 '''
from sklearn import cluster
# K-Means
M1 = cluster.KMeans(n_clusters=3)
m1 = M1.fit(data3_T)
m1.labels_

# MiniBatchKMeans
M2 = cluster.MiniBatchKMeans(n_clusters=3)
m2 = M2.fit(data3_T)
m2.labels_

# AffinityPropagation
M3 = cluster.AffinityPropagation()
m3 = M3.fit(data3_T)
m3.labels_

# MeanShift
M4 = cluster.MeanShift()
m4 = M4.fit(data3_T)
m4.labels_

# SpectralClustering
M5 = cluster.SpectralClustering(n_clusters=3)
m5 = M5.fit(data3_T)
m5.labels_

# AgglomerativeClustering
M6 = cluster.AgglomerativeClustering(n_clusters=3)
m6 = M6.fit(data3_T)
m6.labels_

# Birch
M7 = cluster.Birch(n_clusters=3)
m7 = M7.fit(data3_T)
m7.labels_

# DBSCAN
M8 = cluster.DBSCAN()
m8 = M8.fit(data3_T)
m8.labels_


# 写入Excel
from xlwt import Workbook
table1_wt = Workbook()
sheet1_wt = table1_wt.add_sheet('sheet1')

sheet1_wt.write(0,0,"K-Means")
for i in range(0,len(m1.labels_)):
    sheet1_wt.write(i+1,0,int(m1.labels_[i]))

sheet1_wt.write(0,1,"MiniBatchKMeans")
for i in range(0,len(m2.labels_)):
    sheet1_wt.write(i+1,1,int(m2.labels_[i]))

sheet1_wt.write(0,2,"AffinityPropagation")
for i in range(0,len(m3.labels_)):
    sheet1_wt.write(i+1,2,int(m3.labels_[i]))

sheet1_wt.write(0,3,"MeanShift")
for i in range(0,len(m4.labels_)):
    sheet1_wt.write(i+1,3,int(m4.labels_[i]))

sheet1_wt.write(0,4,"SpectralClustering")
for i in range(0,len(m5.labels_)):
    sheet1_wt.write(i+1,4,int(m5.labels_[i]))

sheet1_wt.write(0,5,"AgglomerativeClustering")
for i in range(0,len(m6.labels_)):
    sheet1_wt.write(i+1,5,int(m6.labels_[i]))

sheet1_wt.write(0,6,"Brich")
for i in range(0,len(m7.labels_)):
    sheet1_wt.write(i+1,6,int(m7.labels_[i]))

sheet1_wt.write(0,7,"DBSCAN")
for i in range(0,len(m8.labels_)):
    sheet1_wt.write(i+1,7,int(m8.labels_[i]))

table1_wt.save(output_file3_1)





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


''' 时间序列聚类模式图 '''
c1 = c2 = 0
# 第1类模式图
ax1 = plt.subplot(211)
for i in range(0,len(data4.columns)):
    if (model_4[1][i] == 0):
        data4.ix[:,i].plot()   
        c1 += 1
ax1.set_title('第1类'+str(c1)+'个观测点')
# 第2类模式图
ax2 = plt.subplot(212)
for i in range(0,len(data4.columns)):
    if (model_4[1][i] == 1):
        data4.ix[:,i].plot()
        c2 += 1
ax2.set_title('第2类'+str(c2)+'个观测点')

plt.tight_layout()
plt.show()
(c1,c2)


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



''' 其他聚类模型 '''
from sklearn import cluster
# K-Means
M1 = cluster.KMeans(n_clusters=2)
m1 = M1.fit(data4_T)
m1.labels_

# MiniBatchKMeans
M2 = cluster.MiniBatchKMeans(n_clusters=2)
m2 = M2.fit(data4_T)
m2.labels_

# AffinityPropagation
M3 = cluster.AffinityPropagation()
m3 = M3.fit(data4_T)
m3.labels_

# MeanShift
M4 = cluster.MeanShift()
m4 = M4.fit(data4_T)
m4.labels_

# SpectralClustering
M5 = cluster.SpectralClustering(n_clusters=2)
m5 = M5.fit(data4_T)
m5.labels_

# AgglomerativeClustering
M6 = cluster.AgglomerativeClustering(n_clusters=2)
m6 = M6.fit(data4_T)
m6.labels_

# Birch
M7 = cluster.Birch(n_clusters=2)
m7 = M7.fit(data4_T)
m7.labels_

# DBSCAN
M8 = cluster.DBSCAN()
m8 = M8.fit(data4_T)
m8.labels_


# 写入Excel
from xlwt import Workbook
table1_wt = Workbook()
sheet1_wt = table1_wt.add_sheet('sheet1')

sheet1_wt.write(0,0,"K-Means")
for i in range(0,len(m1.labels_)):
    sheet1_wt.write(i+1,0,int(m1.labels_[i]))

sheet1_wt.write(0,1,"MiniBatchKMeans")
for i in range(0,len(m2.labels_)):
    sheet1_wt.write(i+1,1,int(m2.labels_[i]))

sheet1_wt.write(0,2,"AffinityPropagation")
for i in range(0,len(m3.labels_)):
    sheet1_wt.write(i+1,2,int(m3.labels_[i]))

sheet1_wt.write(0,3,"MeanShift")
for i in range(0,len(m4.labels_)):
    sheet1_wt.write(i+1,3,int(m4.labels_[i]))

sheet1_wt.write(0,4,"SpectralClustering")
for i in range(0,len(m5.labels_)):
    sheet1_wt.write(i+1,4,int(m5.labels_[i]))

sheet1_wt.write(0,5,"AgglomerativeClustering")
for i in range(0,len(m6.labels_)):
    sheet1_wt.write(i+1,5,int(m6.labels_[i]))

sheet1_wt.write(0,6,"Brich")
for i in range(0,len(m7.labels_)):
    sheet1_wt.write(i+1,6,int(m7.labels_[i]))

sheet1_wt.write(0,7,"DBSCAN")
for i in range(0,len(m8.labels_)):
    sheet1_wt.write(i+1,7,int(m8.labels_[i]))

table1_wt.save(output_file4_1)

