
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


# In[18]:

sns.JointGrid(data=iris_data, x='sepal_length', y='sepal_width').plot(sns.regplot, sns.distplot)
plt.show()


# ## seaborn.kdeplot

# In[19]:

sns.kdeplot(data=iris_data['sepal_length'])
plt.show()


# In[20]:

sns.kdeplot(data=iris_data['sepal_length'], shade=True, color='y')
plt.show()


# In[22]:

sns.kdeplot(data=iris_data['sepal_length'], data2=iris_data['sepal_width'], shade=True)
plt.show()


# In[25]:

import numpy as np

matrix_data = np.random.rand(10, 10)

sns.heatmap(data=matrix_data)
plt.show()


# ## 7. seaborn.clustermap

# In[31]:

iris_data = sns.load_dataset('iris')

iris_data.pop("species")

sns.clustermap(iris_data)
plt.show()


# In[ ]:



