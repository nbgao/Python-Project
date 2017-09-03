
# coding: utf-8

# # 高级API绘图库Seaborn使用

# ## 1. 对原图样式进行快速优化

# In[1]:

import seaborn as sns
# 使用set()方法1步设置默认样式
sns.set()


# sns.set()采用了默认参数

# In[9]:

sns.set(context='notebook', style='darkgrid', palette='deep',
        font='sans-serif', font_scale=1, color_codes=False, rc=None)


# 其中：
# 
# - context='' 参数控制着默认的画幅大小，分别有 {paper, notebook, talk, poster} 四个值。其中，poster > talk > notebook > paper。
# - style='' 参数控制默认样式，分别有 {darkgrid, whitegrid, dark, white, ticks}，你可以自行更改查看它们之间的不同。
# - palette='' 参数为预设的调色板。分别有 {deep, muted, bright, pastel, dark, colorblind} 等，你可以自行更改查看它们之间的不同。
# - font='' 用于设置字体
# - font_scale= 设置字体大小
# - color_codes= 不使用调色板而采用先前的 'r' 等色彩缩写。

# In[10]:

import matplotlib.pyplot as plt
import seaborn as sns

x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
y_bar = [3, 4, 6, 8, 9, 10, 9, 11, 7, 8]
y_line = [2, 3, 5, 7, 8, 9, 8, 10, 6, 7]

sns.set()
plt.bar(x, y_bar)
plt.plot(x, y_line, '-o', color='c')

plt.show()


# ## 2. seabron.lmplot()

# In[13]:

import seaborn as sns

iris_data = sns.load_dataset('iris')  # 导入iris数据集
print(iris_data.head(10))
print(iris_data.tail(10))


# In[14]:

# 加载数据
iris_data = sns.load_dataset('iris')

# 绘图
sns.lmplot(x='sepal_length', y='sepal_width', hue='species', data=iris_data)
plt.show()


# sns.lmplot() 里的 x, y 分别代表横纵坐标的列名。
# 
# hue= 代表按照 species，即花的类别分类显示。
# 
# data= 自然就是关联到数据集了。
# 
# 由于 Seaborn 对 Pandas 的 DataFrame 数据格式高度兼容，所以一切变得异常简单。绘制出来的图也自动带有图例，并进行了线型回归拟合，还给出了置信区间。

# ## 3. seaborn.PairGrid

# In[15]:

sns.PairGrid(data=iris_data).map(plt.scatter)
plt.show()


# ### 用颜色区分类别

# In[16]:

sns.PairGrid(data=iris_data, hue='species').map(plt.scatter)
plt.show()


# ## 4. seaborn.JointGrid

# In[17]:

sns.JointGrid(data=iris_data, x='sepal_length', y='sepal_width')
plt.show()


# ### 双变量回归拟合散点图以及单变量分布图

# In[18]:

sns.JointGrid(data=iris_data, x='sepal_length', y='sepal_width').plot(sns.regplot, sns.distplot)
plt.show()


# ## seaborn.kdeplot

# ### 单变量核密度估计曲线

# In[19]:

sns.kdeplot(data=iris_data['sepal_length'])
plt.show()


# 可以通过 kernel= 参数设置核函数，有 {'gau' | 'cos' | 'biw' | 'epa' | 'tri' | 'triw' }等，默认为 kernel='gau'。

# In[51]:

plt.figure(figsize=(12,9))

ax1 = plt.subplot(3,2,1)
sns.kdeplot(data=iris_data['sepal_length'], kernel='gau', shade=True, color='c')
ax1.set_title('kernel=gau')

ax2 = plt.subplot(3,2,2)
sns.kdeplot(data=iris_data['sepal_length'], kernel='cos', shade=True, color='m')
ax2.set_title('kernel=cos')

ax3 = plt.subplot(3,2,3)
sns.kdeplot(data=iris_data['sepal_length'], kernel='biw', shade=True, color='g')
ax3.set_title('kernel=biw')

ax4 = plt.subplot(3,2,4)
sns.kdeplot(data=iris_data['sepal_length'], kernel='epa', shade=True, color='r')
ax4.set_title('kernel=epa')

ax5 = plt.subplot(3,2,5)
sns.kdeplot(data=iris_data['sepal_length'], kernel='tri', shade=True, color='b')
ax5.set_title('kernel=tri')

ax6 = plt.subplot(3,2,6)
sns.kdeplot(data=iris_data['sepal_length'], kernel='triw', shade=True, color='y')
ax6.set_title('kernel=triw')

plt.show()


# In[96]:

plt.figure(figsize=(12,16))

ax1 = plt.subplot(3,2,1)
sns.kdeplot(data=iris_data['sepal_length'], data2=iris_data['sepal_width'], kernel='gau', shade=True, cmap='Greens')
ax1.set_title('kernel=gau')

ax2 = plt.subplot(3,2,2)
sns.kdeplot(data=iris_data['sepal_length'], data2=iris_data['sepal_width'], kernel='cos', shade=True, cmap='Oranges')
ax2.set_title('kernel=cos')

ax3 = plt.subplot(3,2,3)
sns.kdeplot(data=iris_data['sepal_length'], data2=iris_data['sepal_width'], kernel='biw', shade=True, cmap='Blues')
ax3.set_title('kernel=biw')

ax4 = plt.subplot(3,2,4)
sns.kdeplot(data=iris_data['sepal_length'], data2=iris_data['sepal_width'], kernel='epa', shade=True, cmap='Reds')
ax4.set_title('kernel=epa')

ax5 = plt.subplot(3,2,5)
sns.kdeplot(data=iris_data['sepal_length'], data2=iris_data['sepal_width'], kernel='tri', shade=True, cmap='Purples')
ax5.set_title('kernel=tri')

ax6 = plt.subplot(3,2,6)
sns.kdeplot(data=iris_data['sepal_length'], data2=iris_data['sepal_width'], kernel='triw', shade=True, cmap='RdPu')
ax6.set_title('kernel=triw')

plt.tight_layout()
plt.show()


# In[76]:

import numpy as np

matrix_data = np.random.rand(10, 10)

plt.figure(figsize=(12,10))
sns.heatmap(data=matrix_data, cmap='Blues')
plt.show()


# ## 7. seaborn.clustermap

# In[105]:

sns.set(color_codes=True)

iris_data = sns.load_dataset('iris')

iris_data.pop("species")

sns.clustermap(iris_data, standard_scale=1, cmap='Blues_r')
plt.show()


# In[ ]:



