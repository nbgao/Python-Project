
# coding: utf-8

# # 1. Class MATLAB API

# In[2]:

from pylab import *


# In[4]:

from numpy import *
x = linspace(0,5,10)
y = x**2

figure()
plot(x,y,'r')
xlabel('x')
ylabel('y')
title('title')
show()


# In[6]:

subplot(1,2,1)
plot(x,y,'r--')
subplot(1,2,2)
plot(y,x,'g*-');


# # 2. matplotlib oriented object API

# In[7]:

import matplotlib.pyplot as plt


# In[9]:

fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])  # insert axes

# main figure
axes1.plot(x,y,'r')
axes1.set_xlabel('x')
axes1.set_ylabel('y')
axes1.set_title('title')

# insert
axes2.plot(y,x,'g')
axes2.set_xlabel('y')
axes2.set_ylabel('x')
axes2.set_title('insert title')


# In[11]:

fig, axes = plt.subplots()

axes.plot(x,y,'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');


# In[17]:

fig, axes = plt.subplots(nrows=1, ncols=2)

for ax in axes:
    ax.plot(x,y,'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')

# tight_layout解决标签重叠问题
fig.tight_layout()


# # 2.1 Figure size, length/width and DPI

# In[18]:

fig = plt.figure(figsize=(8,4), dpi=100)


# In[21]:

fig, axes = plt.subplots(figsize=(12,5))

axes.plot(x,y,'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');


# # 2.2 Save figure

# In[26]:

fig.savefig("Figure/filename1.png")


# In[27]:

fig.savefig("Figure/filename2.png", dpi=200)


# In[31]:

fig.savefig("Figure/filename3.pgf")


# # 2.3 Title, axis and legend

# In[29]:

ax.set_title("title")


# In[30]:

ax.set_xlabel('x')
ax.set_ylabel('y')


# In[32]:

ax.legend(["curve1", "curve2", "curve3"]);


# In[39]:

ax.plot(x, x**2, label="curve1")
ax.plot(x, x**3, label="curve2")
ax.legend()


# In[41]:

# let matplotlib decide the optimal location
ax.legend(loc=0)
# upper right corner
ax.legend(loc=1)
# upper left corner
ax.legend(loc=2)
# lower left corner
ax.legend(loc=3)
# lower right corner
ax.legend(loc=4)


# # 2.4 Text, LaTex, font size, font style

# In[51]:

from matplotlib import rcParams


# In[52]:

# Update the matplotlib configuration parameters
rcParams.update({'font.size':18, 'font.family':'serif'})

fig, ax = plt.subplots()

# LaTex
ax.plot(x, x**2, label=r"$y = \alpha^2$")
ax.plot(x, x**3, label=r"$y = \alpha^3$")
ax.legend(loc=2)  # upper left corner

# Font size
ax.set_xlabel(r"$\alpha$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_title('title');


# ## STIX font style

# In[54]:

# Update the matplotlib configuration parameters
rcParams.update({'font.size':18, 'font.family':'STIXGeneral', 'mathtext.fontset':'stix'})

fig, ax = plt.subplots()

# LaTex
ax.plot(x, x**2, label=r"$y = \alpha^2$")
ax.plot(x, x**3, label=r"$y = \alpha^3$")
ax.legend(loc=2)  # upper left corner

# Font size
ax.set_xlabel(r"$\alpha$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_title('title');


# ### LaTex render

# In[55]:

rcParams.update({'font.size':18, 'text.usetex':True})

fig, ax = plt.subplots()

# LaTex
ax.plot(x, x**2, label=r"$y = \alpha^2$")
ax.plot(x, x**3, label=r"$y = \alpha^3$")
ax.legend(loc=2)  # upper left corner

# Font size
ax.set_xlabel(r"$\alpha$", fontsize=18)
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_title('title');


# In[57]:

# Reset
rcParams.update({'font.size':12, 'font.family':'sans', 'text.usetex':False})


# # 2.5 Set color, line width and line style

# ## color

# In[58]:

ax.plot(x, x**2, 'b.-')
ax.plot(x, x**3, 'g--')

fig


# In[59]:

fig, ax = plt.subplots()

ax.plot(x, x+1, color="red", alpha=0.5)
ax.plot(x, x+2, color="#1155dd")
ax.plot(x, x+3, color="#15cc55")


# ## line and point style

# In[68]:

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(x, x+1, color="blue", linewidth=0.25)
ax.plot(x, x+2, color="blue", linewidth=0.50)
ax.plot(x, x+3, color="blue", linewidth=1.00)
ax.plot(x, x+4, color="blue", linewidth=2.00)

# possible linestype options ‘-‘, ‘–’, ‘-.’, ‘:’, ‘steps’
ax.plot(x, x+5, color="red", lw=2, linestyle='-')
ax.plot(x, x+6, color="red", lw=2, ls='-.')
ax.plot(x, x+7, color="red", lw=2, ls=':')

# custom dash
line, = ax.plot(x, x+8, color="black", lw=1.50)
line.set_dashes([5, 10, 15, 10]) # format: line length, space length, ...

# possible marker symbols: marker = '+', 'o', '*', 's', ',', '.', '1', '2', '3', '4', ...
ax.plot(x, x+ 9, color="green", lw=2, ls='', marker='+')
ax.plot(x, x+10, color="green", lw=2, ls='', marker='o')
ax.plot(x, x+11, color="green", lw=2, ls='', marker='s')
ax.plot(x, x+12, color="green", lw=2, ls='', marker='1')

# marker size and color
ax.plot(x, x+13, color="purple", lw=1, ls='-', marker='o', markersize=2)
ax.plot(x, x+14, color="purple", lw=1, ls='-', marker='o', markersize=4)
ax.plot(x, x+15, color="purple", lw=1, ls='-', marker='o', markersize=8, markerfacecolor="red")
ax.plot(x, x+16, color="purple", lw=1, ls='-', marker='s', markersize=8, markerfacecolor="yellow", markeredgewidth=2, markeredgecolor="blue")


# # 2.6 Control axis style

# In[71]:

fig, axes = plt.subplots(1,3,figsize=(12,4))

axes[0].plot(x,x**2,x,x**3)
axes[0].set_title("default axes ranges")

axes[1].plot(x,x**2,x,x**3)
axes[1].axis('tight')
axes[1].set_title("tight axes")

axes[2].plot(x,x**2,x,x**3)
axes[2].set_ylim([0,60])
axes[2].set_xlim([2,5])
axes[2].set_title("custom axes range");


# ## log scale

# In[72]:

fig, axes = plt.subplots(1,2,figsize=(10,4))

axes[0].plot(x,x**2,x,exp(x))
axes[0].set_title("Normal scale")

axes[1].plot(x,x**2,x,exp(x))
axes[1].set_yscale("log")
axes[1].set_title("Logarithmic scale (y)");


# # 2.7 Custom position and symbol

# In[78]:

fig, ax = plt.subplots(figsize=(10,4))

ax.plot(x,x**2,x,x**3,lw=2)
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels([r'$\alpha$', r'$\beta$', r'$\gamma$', r'$\delta$', r'$\epsilon$'], fontsize=18)

yticks = [0,50,100,150]
ax.set_yticks(yticks)
ax.set_yticklabels(["$%.1f$" % y for y in yticks], fontsize=18);


# ## Notation scientifique

# In[79]:

fig, ax= plt.subplots(1,1)

ax.plot(x,x**2,x,exp(x))
ax.set_title("scientific notation")
ax.set_yticks([0,50,100,150])

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
ax.yaxis.set_major_formatter(formatter)


# # 2.8 Axis numbers and label to axis spacing

# In[83]:

rcParams['xtick.major.pad'] = 2
rcParams['ytick.major.pad'] = 2

fig, ax = plt.subplots(1,1)
ax.plot(x,x**2,x,exp(x))
ax.set_yticks([0,50,100,150])
ax.set_title("label and axis spacing")

# padding between axis label and axis numbers
ax.xaxis.labelpad = 2
ax.yaxis.labelpad = 2

ax.set_xlabel('x')
ax.set_ylabel('y');


# ## adjust axis position

# In[84]:

fig, ax = plt.subplots(1,1)

ax.plot(x,x**2,x,exp(x))
ax.set_yticks([0,50,100,150])

ax.set_title('title')
ax.set_xlabel('x')
ax.set_ylabel('y')

fig.subplots_adjust(left=0.15, right=0.9,bottom=0.1, top=0.9);


# # 2.9 Axis grid

# In[96]:

fig, axes = plt.subplots(1,2,figsize=(10,3))

# default grid apperarance
axes[0].plot(x,x**2,x,x**3,lw=2)
axes[0].grid(True)

# custom grid appearance
axes[1].plot(x,x**2,x,x**3,lw=2)
axes[1].grid(color='b', alpha=0.6, linestyle='dashed', linewidth=0.6)


# # 2.10 Axis

# In[97]:

fig, ax = plt.subplots(figsize=(6,2))

ax.spines['bottom'].set_color('blue')
ax.spines['top'].set_color('blue')

ax.spines['left'].set_color('red')
ax.spines['left'].set_linewidth(2)

# turn off axis spine to the right
ax.spines['right'].set_color('none')
ax.yaxis.tick_left()  # only ticks on the left side


# # 2.11 Double axis

# In[99]:

fig, ax1 = plt.subplots()

ax1.plot(x,x**2,lw=2,color="blue")
ax1.set_ylabel(r"area $(m^2)$", fontsize=18, color="blue")
for label in ax1.get_yticklabels():
    label.set_color("blue")

ax2 = ax1.twinx()
ax2.plot(x,x**3,lw=2,color="red")
ax2.set_ylabel(r"volume $(m^3)$", fontsize=18, color="red")

for label in ax.get_yticklabels():
    label.set_color("red")


# # 2.12 Set origin point at (0,0)

# In[102]:

fig, ax = plt.subplots()

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))  # set position of x spine to x=0

ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))  # set position of y spine to y=0

xx = np.linspace(-0.75,1,100)
ax.plot(xx, xx**3);


# # 2.13 Other 2D figure style

# In[138]:

n = array([0,1,2,3,4,5])

fig, axes = plt.subplots(2,2,figsize=(12,12))

axes[0,0] = plt.subplot(221)
axes[0,0].set_title("scatter")
axes[0,0].scatter(xx,xx+0.25*randn(len(xx)))


axes[0,1] = plt.subplot(222)
axes[0,1].set_title("step")
axes[0,1].step(n,n**2,lw=2)


axes[1,0] = plt.subplot(223)
axes[1,0].set_title("bar")
axes[1,0].bar(n,n**2, align="center", width=0.5, alpha=0.5)


axes[1,1] = plt.subplot(224)
axes[1,1].set_title("fill_between")
axes[1,1].fill_between(x,x**2,x**3, color="green", alpha=0.5);


# In[130]:

# polar plot using add_axes and polar projection
fig = plt.figure()
ax = fig.add_axes([0.0,0.0,0.6,0.6], polar=True)
t = linspace(0,2*pi,100)
ax.plot(t,t,lw=3)


# In[140]:

# histogram
n = np.random.randn(100000)

fig, axes = plt.subplots(1,2,figsize=(12,4))

axes[0].hist(n)
axes[0].set_title("Default histogram")
axes[0].set_xlim((min(n), max(n)))

axes[1].hist(n, cumulative=True, bins=50)
axes[1].set_title("Cumulative detailed histogram")
axes[1].set_xlim((min(n), max(n)));


# # 2.14 Text comments

# In[141]:

fig, ax = plt.subplots()

ax.plot(xx, xx**2, xx, xx**3)

ax.text(0.15, 0.2, r'$y=x^2$', fontsize=20, color='blue')
ax.text(0.65, 0.1, r'$y=x^3$', fontsize=20, color='green');


# # 2.15 Multiple subfigure & insert figure

# ## subplots

# In[152]:

fig, ax = plt.subplots(2, 3, figsize=(9,6))
fig.tight_layout()


# ## subplot2grid

# In[151]:

fig = plt.figure(figsize=(6,6))

ax1 = plt.subplot2grid((3,3),(0,0), colspan=3)
ax2 = plt.subplot2grid((3,3),(1,0), colspan=2)
ax3 = plt.subplot2grid((3,3),(1,2), rowspan=2)
ax4 = plt.subplot2grid((3,3),(2,0))
ax5 = plt.subplot2grid((3,3),(2,1))

fig.tight_layout()


# ## add_axes

# In[154]:

fig, ax = plt.subplots()

ax.plot(xx, xx**2, xx, xx**3)
fig.tight_layout()

# inset
inset_ax = fig.add_axes([0.2, 0.55, 0.35, 0.35])  # X,Y,width,height

inset_ax.plot(xx, xx**2, xx, xx**3)
inset_ax.set_title('zoom near origin')

# set axis range
inset_ax.set_xlim(-0.2,0.2)
inset_ax.set_ylim(-0.005,0.01)

# set axis tick locations
inset_ax.set_yticks([0,0.005,0.01])
inset_ax.set_xticks([-0.1,0,0.1]);


# # 2.16 Color mapping & contour figure

# In[155]:

alpha = 0.7
phi_ext = 2*pi*0.5


# In[156]:

def flux_qubit_potential(phi_m, phi_p):
    return 2+alpha-2*cos(phi_p)*cos(phi_m) - alpha*cos(phi_ext-2*phi_p)


# In[161]:

phi_m = linspace(0,2*pi,200)
phi_p = linspace(0,2*pi,200)
X,Y = meshgrid(phi_p, phi_m)
Z = flux_qubit_potential(X,Y).T


# ## pcolor

# In[163]:

fig, ax = plt.subplots(figsize=(8,6))
p = ax.pcolor(X/(2*pi), Y/(2*pi), Z, cmap=plt.cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max())
cb = fig.colorbar(p, ax=ax)


# ## imshow

# In[176]:

fig, ax = plt.subplots(figsize=(8,6))

im = ax.imshow(Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0,1,0,1])
im.set_interpolation('bilinear')

cb = fig.colorbar(im, ax=ax)


# ## contour

# In[179]:

fig, ax = plt.subplots(figsize=(6,5))
cnt = ax.contour(Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0,1,0,1])


# In[ ]:



