
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt

plt.plot([1,2,3,2,1,2,3,4,5,6,5,4,3,2,1])
plt.show()


# ## matplotlib.pyplot模块
# 
# |__方法__|__含义__|
# |:------:|:-------:|
# |matplotlib.pyplot.angle_spectrum|电子波普图|
# |matplotlib.pyplot.bar|柱状图|
# |matplotlib.pyplot.barh|直方图|
# |matplotlib.pyplot.broken_barh|水平直方图|
# |matplotlib.pyplot.contourf|等高线图|
# |matplotlib.pyplot.errorbar|误差线图|
# |matplotlib.pyplot.hexbin|六边形图案|
# |matplotlib.pyplot.hist|柱形图|
# |matplotlib.pyplot.hist2d|水平柱形图|
# |matplotlib.pyplot.imshow|图像显示|
# |matplotlib.pyplot.pie|饼状图|
# |matplotlib.pyplot.quiver|量场图|
# |matplotlib.pyplot.scatter|散点图|
# |matplotlib.pyplot.spcgram|光谱图|
# |matplotlib.pyplot.subplot|子图|

# |__方法__|__含义__|
# |:--:|:--:|
# |matplotlib.pyplot.annotate|绘制图形标注|
# |matplotlib.pyplot.axhspan|绘制垂直或水平色块|
# |matplotlib.pyplot.clabel|标注轮廓线|
# |matplotlib.pyplot.fill|填充区域|

# # 2D图像绘制

# ## 1. 线形图 plot

# In[3]:

import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.sin(X)

plt.plot(X,y)
plt.show()


# ## 2. 柱形图 bar

# In[19]:

X = np.linspace(-2*np.pi, 2*np.pi, 15)
y = np.cos(X)
plt.bar(X,y)
plt.show()


# ## 3. 散点图 scatter

# In[23]:

X = np.random.normal(0,1,1000)
y = np.random.normal(0,1,1000)

plt.figure(figsize=(6,6))
plt.scatter(X,y)
plt.show()


# ## 4. 饼状图 pie

# In[37]:

Z = [1,2,3,4,5]

plt.figure(figsize=(5,5))
plt.pie(Z)
plt.show()


# ## 5. 量场图 quiver

# In[42]:

X, y =np.mgrid[0:10, 0:10]

plt.figure(figsize=(6,6))
plt.quiver(X, y)
plt.show()


# ## 6. 等高线图 contourf

# In[49]:

x = np.linspace(-5,5,500)
y = np.linspace(-5,5,500)

X, Y = np.meshgrid(x,y)
Z = (1-X/2+X**3+Y**4) * np.exp(-X**2-Y**2)

plt.figure(figsize=(8,8))
plt.contourf(X,Y,Z)
plt.show()


# In[ ]:



