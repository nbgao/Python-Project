
# coding: utf-8

# # 2D 绘图进阶

# ## 1. 线形图样式

# |__参数__|__含义__|
# |:-:|:-:|
# |alpha=|设置线型的透明度，从0.0到1.0|
# |color=|设置线型的颜色|
# |fillstyle=|设置线型的填充样式|
# |linestyle=|设置线型的样式|
# |linewidth=|设置线型的宽度|
# |marker=|设置标记点的样式|

# ### 颜色
# 
# |__color=参数__|__含义__|
# |:-:|:-:|
# |b|蓝色|
# |g|绿色|
# |r|红色|
# |w|白色|
# |m|洋红色|
# |y|黄色|
# |k|黑色|

# ### 线型
# 
# |__linestyle=参数__|__含义__|
# |:-:|:-:|
# |'-'|默认实线|
# |'--'|虚线|
# |'-.'|间断线|
# |':'|点状线|

# ### 点标记样式
# 
# |__marker=参数__|__含义__|
# |:-:|:-:|
# |'.'|实心点|
# |','|像素点|
# |'o'|空心点|
# |'p'|五角形点|
# |'x'|X形点|
# |'+'|+形点|

# In[6]:

import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(-2*np.pi, 2*np.pi, 1000)
y1 = np.sin(X)
y2 = np.cos(X)

plt.plot(X, y1, color='r', linestyle='--', linewidth=2, alpha=0.8)
plt.plot(X, y2, color='b', linestyle='-.', linewidth=2)
plt.show()


# ## 2. 散点图样式

# |__参数__|__含义__|
# |:-:|:-:|
# |s=|散点大小|
# |c=|散点颜色|
# |marker=|散点样式|
# |cmap=|定义多类别散点的颜色|
# |alpha=|点的透明度|
# |edgecolors=|散点边缘颜色|

# In[10]:

x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)
size = np.random.normal(10,40,100)

plt.figure(figsize=(6,6))
plt.scatter(x, y, s=size, c=colors)
plt.show()


# ## 3. 饼状图样式

# In[21]:

labels = ['Cat', 'Dog', 'Cattle', 'Sheep', 'Horse']  # 各类别标签
colors = ['r', 'g', 'b', 'y', 'c']  # 各类别颜色
size = [1, 2, 3, 4, 5]  # 各类别占比
explode = (0.0, 0.0, 0.0, 0.0, 0.2)  # 各类别便宜半径

plt.figure(figsize=(6,6))
plt.pie(size, colors=colors, explode=explode, labels=labels, shadow=True, autopct='%1.2f%%')
plt.axis('equal')
plt.show()


# ## 4. 组合图

# In[111]:

x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
y_bar = [3, 4, 6, 8, 9, 10, 9, 11, 7, 8]
y_line = [2, 3, 5, 8, 7, 9, 8, 10, 6, 7]

plt.figure(figsize=(12,6))
plt.bar(x, y_bar, color='r', alpha=0.75)
plt.plot(x, y_line, '-o', color='y')
plt.show()


# ## 5. 子图绘制

# In[48]:

x = np.linspace(-2*np.pi, 2*np.pi, 100)

y1 = np.sin(x)
y2 = np.tanh(x)
y3 = np.exp(-(x**2)/2)
y4 = 1/(1+np.exp(-x))

plt.figure(figsize=(12,8))

# 子图1
plt.subplot(2,2,1)
plt.plot(x, y1, 'r')

# 子图2
plt.subplot(2,2,2)
plt.plot(x, y2, 'c')

# 子图3
plt.subplot(2,2,3)
plt.plot(x, y3, 'b')

# 子图4
plt.subplot(2,2,4)
plt.plot(x, y4, 'g')

plt.show()


# In[86]:

x = np.linspace(-2*np.pi, 2*np.pi, 100)

y1 = np.log(x**2+1)
y2 = np.exp(abs(x*2)/20)

plt.figure(figsize=(8,4))
# 大图
plt.axes([.0, .0, .8, .8])
plt.plot(x, y1, 'm')

# 小图
plt.axes([.23, .5, .34, .3])  # 左边距，上边距，长度，高度
plt.plot(x, y2, 'r')

plt.show()


# ## 6. 绘制图例

# In[92]:

X = np.linspace(-2*np.pi, 2*np.pi, 100)
y1 = np.sin(X)
y2 = np.cos(X)

plt.figure(figsize=(8,4))

plt.plot(X, y1, color='r', linestyle='--', linewidth=2, label='$y=sin(x)$')
plt.plot(X, y2, color='b', linestyle='-.', linewidth=2, label='$y=cos(x)$')

plt.legend(loc='upper left')
plt.show()


# ## 7. 图像标注

# In[123]:

x_bar = [10, 20, 30, 40, 50]
y_bar = [0.5, 0.6, 0.7, 0.4, 0.6]

plt.figure(figsize=(6,4))
bars = plt.bar(x_bar, y_bar, color='rygcb', label=x_bar, width=3)
plt.ylim(.0, .8)

for i, rect in enumerate(bars):
    x_test = rect.get_x()
    y_test = rect.get_height() + 0.01
    plt.text(x_test, y_test, '%.1f' % y_bar[i])  # 标注文字
    
plt.show()


# In[153]:

x_bar = [10, 20, 30, 40, 50]
y_bar = [0.5, 0.6, 0.7, 0.4, 0.6]

plt.figure(figsize=(9,4))
bars = plt.bar(x_bar, y_bar, color='rygcb', label=x_bar, width=3)
plt.ylim(.0, .8)

for i, rect in enumerate(bars):
    x_test = rect.get_x() + 0.7
    y_test = rect.get_height() + 0.01
    plt.text(x_test, y_test, '%.1f' % y_bar[i])  # 标注文字
    
    # 增加箭头标注4
    plt.annotate('Max', xy=(32,0.68), xytext=(36,0.69), arrowprops=dict(facecolor='black', width=1.5, headwidth=7))
    
plt.show()


# - xy=() 表示标注终点坐标
# - xytext=() 表示标注起点坐标
# - arrowprops=() 用于设置箭头样式
# - facecolor= 设置颜色
# - width= 设置箭尾宽度
# - headwidth= 设置箭头宽度
# 
# 
# - arrowstyle= 用于改变箭头的样式
# - connectionstyle= 的参数可以用于更改箭头连接的样式

# ## 8. 动态图绘制

# In[173]:

import matplotlib.animation as animation

# 生成数据并建立绘制类型的图像
fig, ax = plt.subplots(figsize=(9,6))

x = np.arange(-np.pi, np.pi, 0.01)
line, = plt.plot(x, np.sin(x))

# 更新函数
def update(i):
    line.set_ydata(np.sin(x + i/10.0))
    return line,

# 绘制动图
animation = animation.FuncAnimation(fig, update)

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.show()


# In[ ]:



