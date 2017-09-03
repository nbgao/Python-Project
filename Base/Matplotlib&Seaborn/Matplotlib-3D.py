
# coding: utf-8

# # 3D 绘图

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.matplotlib_fname()
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# ## 1. mplot3d 绘图模块介绍

# mplot3d 模块下主要包含 4 个大类，分别是：
# 
# - mpl_toolkits.mplot3d.axes3d()
# - mpl_toolkits.mplot3d.axis3d()
# - mpl_toolkits.mplot3d.art3d()
# - mpl_toolkits.mplot3d.proj3d()
# 
# 其中，axes3d() 下面主要包含了各种实现绘图的类和方法。axis3d() 主要是包含了和坐标轴相关的类和方法。art3d() 包含了一些可将 2D 图像转换并用于 3D 绘制的类和方法。proj3d() 中包含一些零碎的类和方法，例如计算三维向量长度等。
# 
# 一般情况下，我们用到最多的就是 mpl_toolkits.mplot3d.axes3d() 下面的 mpl_toolkits.mplot3d.axes3d.Axes3D() 类，而 Axes3D() 下面又存在绘制不同类型 3D 图的方法。你可以通过下面的方式导入 Axes3D()。

# ## 2. 三维散点图 scatter

# In[5]:

import numpy as np

# x, y, z均为0到1之间的100个随机数
x = np.random.normal(0,1,100)
y = np.random.normal(0,1,100)
z = np.random.normal(0,1,100)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 创建3D图形对象
fig = plt.figure(figsize=(8,6))
ax = Axes3D(fig)

ax.scatter(x, y, z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# ## 2. 三维线型图 plot

# In[11]:

from mpl_toolkits.mplot3d import Axes3D

x= np.linspace(-6*np.pi, 6*np.pi, 1000)
y = np.sin(x)
z = np.cos(x)

# 创建3D图形对象
fig = plt.figure(figsize=(8,6))
ax = Axes3D(fig)

# 绘制线型图
ax.plot(x, y, z)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# ## 4. 三维柱状图 bar

# In[20]:

from mpl_toolkits.mplot3d import Axes3D

# 创建3D图形对象
fig = plt.figure(figsize=(8,6))
ax = Axes3D(fig)

x = np.arange(0,7)
for i in x:
    y = np.arange(0,10)
    z = abs(np.random.normal(1,10,10))
    ax.bar(y, z, i, zdir='y', color=['r', 'g', 'b', 'y'])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
plt.show();


# ## 5. 三维曲面图 plot_surface

# In[48]:

from mpl_toolkits.mplot3d import Axes3D

# 创建3D图形对象
fig = plt.figure(figsize=(8,6))
ax = Axes3D(fig)

X = np.arange(-2,2,0.1)
Y = np.arange(-2,2,0.1)
X, Y = np.meshgrid(X,Y)
Z = np.sqrt(X**2+Y**2)

# 绘制曲面图，并使用cmap着色
ax.plot_surface(X, Y, Z, cmap=plt.cm.winter)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# ## 6. 混合图绘制

# In[69]:

from mpl_toolkits.mplot3d import Axes3D

# 创建3D图形对象
fig = plt.figure(figsize=(8,6))
ax = Axes3D(fig)

# 生成数据并绘制图1
x1 = np.linspace(-3*np.pi, 3*np.pi, 500)
y1 = np.sin(x1)
ax.plot(x1, y1, zs=0, c='red')

# 生成数据并绘制图2
x2 = np.random.normal(0,1,100)
y2 = np.random.normal(0,1,100)
z2 = np.random.normal(0,1,100)
ax.scatter(x2, y2, z2)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# ## 7. 子图绘制

# In[83]:

from mpl_toolkits.mplot3d import Axes3D

# 创建第1张画布
fig = plt.figure(figsize=(12,5))

# 向画布添加子图1
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

x = np.linspace(-6*np.pi,6*np.pi, 1000)
y = np.sin(x)
z = np.cos(x)

ax1.plot(x, y, z)

# 向画布添加子图2
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = X**2-Y**2

ax2.plot_surface(X, Y, Z, cmap=plt.cm.autumn)

plt.show()


# In[ ]:



