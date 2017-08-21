
# coding: utf-8

# In[2]:

import numpy as np


# # 1. Create numpy array
# ## array

# In[5]:

v = np.array([1,2,3,4])
v


# In[22]:

M = np.array([[1,2], [3,4]])
M


# In[11]:

type(v), type(M)


# In[12]:

v.shape, M.shape


# In[14]:

v.size, M.size


# In[16]:

np.shape(v), np.shape(M)


# In[19]:

M.dtype


# ## arange

# In[23]:

M = np.array([[1,2],[3,4]], dtype=complex)
M


# In[24]:

x = np.arange(0, 10, 1)
x


# In[25]:

x = np.arange(-1,1,0.1)
x


# ## linspace & logspace

# In[27]:

np.linspace(0,10,5)


# In[29]:

np.logspace(0,10,10,base=np.e)


# ## mgrid

# In[33]:

x, y = np.mgrid[0:5, 0:5]
x


# In[34]:

y


# ## random data

# In[35]:

from numpy import random
# uniform random numbers in [0,1]
random.rand(5,5)


# In[37]:

# standard normal distributed random numbers
random.randn(5,5)


# ## diag

# In[39]:

np.diag([1,2,3])


# In[40]:

# diagonal with offset from the main diagonal
np.diag([1,2,3], k=1)


# ## zeros & ones

# In[41]:

np.zeros((3,3))


# In[43]:

np.ones((3,3))


# # 2. File I/O create array
# ## CSV

# ### numpy.genfromtxt read data file

# In[64]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
data = np.genfromtxt('Data/stockholm_td_adj.dat')
data.shape


# In[54]:

fig, ax = plt.subplots(figsize = (14, 4))
ax.plot(data[:,0] + data[:,1]/12.0 + data[:,2]/365, data[:,5])
ax.axis('tight')
ax.set_title('temperatures in Stockholm')
ax.set_xlabel('year')
ax.set_ylabel('temperature(C)')
fig


# **使用np.savetxt可以将numpy数组保存到CSV文件中**

# In[55]:

M  = random.rand(3,3)
M


# In[65]:

np.savetxt("Data/random-matrix.csv", M)


# In[66]:

np.savetxt("Data/random-matrix-2.csv", M, fmt='%.5f')


# ## numpy native file type

# In[68]:

np.save("Data/random-matrix.npy", M)


# In[70]:

np.load("Data/random-matrix.npy")


# # 3. numpy array common property

# In[71]:

M.itemsize


# In[73]:

M.nbytes


# In[74]:

M.ndim


# # 4. Operator array

# ## Index

# In[75]:

v[0]


# In[76]:

M[1,1]


# In[77]:

M


# In[78]:

M[1]


# In[80]:

M[1,:]


# In[82]:

M[:,1]


# In[83]:

M[0,0] = 1
M


# In[86]:

M[1,:] = 0
M[:,2] = -1
M


# ## Split index

# In[87]:

A = np.array([1,2,3,4,5])
A


# In[88]:

A[1:3]


# In[89]:

A[1:3] = [-2, -3]
A


# **lower, upper, step all take the default values**

# In[90]:

A[::] 


# **step is 2**

# In[92]:

A[::2]


# **first three elements**

# In[93]:

A[:3]


# In[94]:

A[3:]


# **reverse index**

# In[95]:

A[-1]


# In[96]:

A[-3:]


# In[98]:

A = np.array([[n+m*10 for n in range(5)] for m in range(5)])
A


# In[99]:

A[1:4, 1:4]


# In[100]:

A[::2, ::2]


# ## Fancy indexing

# In[102]:

row_indices = [1,2,3]
A[row_indices]


# In[104]:

col_indices = [1,2,-1]
A[row_indices, col_indices]


# In[105]:

B = np.array([n for n in range(5)])
B


# In[106]:

row_mask = np.array([True, False, True, False, False])
B[row_mask]


# In[108]:

row_mask = np.array([1,0,1,0,0], dtype=bool)
B[row_mask]


# In[110]:

x = np.arange(0,10,0.5)
x


# In[111]:

mask = (5<x) * (x<7.5)
mask


# In[112]:

x[mask]


# # 5. Operate numpy array functions

# where

# In[115]:

indices = np.where(mask)
indices


# In[116]:

x[indices]


# ## diag

# In[118]:

np.diag(A)


# In[120]:

np.diag(A, -1)


# ## take

# In[122]:

v2 = np.arange(-3,3)
v2


# In[123]:

row_indices = [1,3,5]
v2[row_indices]


# In[124]:

v2.take(row_indices)


# In[126]:

np.take([-3,-2,-1,0,1,2], row_indices)


# ## choose

# In[129]:

which = [1,0,1,0]
choices = [[-2,-2,-2,-2], [5,5,5,5]]
np.choose(which, choices)


# # 6. Linear Algebra

# In[130]:

v1 = np.arange(0,5)
v1 * 2


# In[131]:

v1 + 1


# In[132]:

A * 2, A + 2


# ## Element-wise *

# In[133]:

A * A


# In[134]:

v1 * v1


# In[135]:

A.shape, v1.shape


# In[136]:

A * v1


# ## Matrix Calculas

# In[138]:

np.dot(A, A)


# In[139]:

np.dot(A, v1)


# In[143]:

np.dot(v1, v1)


# ### array mapping to matrix

# In[146]:

M = np.matrix(A)
v = np.matrix(v1).T
v


# In[147]:

M * M


# In[148]:

M * v


# In[149]:

v.T * v


# In[150]:

v + M * v


# In[154]:

v = np.matrix([1,2,3,4,5,6]).T
np.shape(M), np.shape(v)


# ## array/matrix transpose

# In[157]:

C = np.matrix([[1j, 2j], [3j, 3j]])
C


# In[158]:

np.conjugate(C)


# In[159]:

C.H


# In[161]:

np.real(C)


# In[162]:

np.imag(C)


# In[163]:

np.angle(C+1)


# In[164]:

abs(C)


# ## Matrix calculas

# In[165]:

from scipy.linalg import *
inv(C)


# In[166]:

C.I * C


# In[169]:

np.linalg.det(C)


# In[170]:

np.linalg.det(C.I)


# ## Data process

# In[172]:

np.shape(data)


# ### mean

# In[173]:

np.mean(data[:,3])


# ### std & var

# In[175]:

np.std(data[:,3]), np.var(data[:,3])


# ### minimum & maximum

# In[176]:

data[:,3].min()


# In[177]:

data[:,3].max()


# In[180]:

d = np.arange(0,10)
d


# ### sum, prod, trace

# In[181]:

sum(d)


# In[183]:

np.prod(d+1)


# In[185]:

np.cumsum(d)


# In[187]:

np.cumprod(d+1)


# In[188]:

np.trace(A)


# ## Process for subarray

# In[189]:

np.unique(data[:,1])


# In[192]:

mask_feb = data[:,1] == 2
np.mean(data[mask_feb,3])


# In[199]:

bmonths = np.arange(1,13)
monthly_mean = [np.mean(data[data[:,1] == month, 3]) for month in months]
fig, ax = plt.subplots()
ax.bar(months, monthly_mean)
ax.set_xlabel("Month")
ax.set_ylabel("Monthly avg. temp.")


# ## operate high dimensional

# In[209]:

m = random.rand(3,3)
m


# In[211]:

m.max()


# In[212]:

m.max(axis=0)


# In[213]:

m.max(axis=1)


# # Change shape & size

# In[200]:

A


# In[204]:

n, m = A.shape
B = A.reshape((1, n*m))
B


# In[215]:

B[0,0:5] = 5
B


# In[216]:

A


# **flatten function create a high level array vector version**

# In[218]:

B = A.flatten()
B


# In[219]:

B[0:5] = 10
B


# In[220]:

A


# # 7. Create a new dimensional

# In[223]:

v = np.array([1,2,3])
np.shape(v)


# In[225]:

v[:,np.newaxis]


# In[227]:

v[:,np.newaxis].shape


# In[229]:

v[np.newaxis,:].shape


# In[ ]:



