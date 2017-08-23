
# coding: utf-8

# # 1. Load dataset

# ## Iris

# In[2]:

from sklearn import datasets
iris = datasets.load_iris()
iris.data.shape


# In[3]:

iris.DESCR


# In[4]:

iris.feature_names


# In[5]:

iris.target_names


# In[7]:

iris.target.shape


# In[8]:

import numpy as np
np.unique(iris.target)


# ## MINST

# In[10]:

digits = datasets.load_digits()
digits.DESCR


# In[11]:

print(digits.data)


# In[12]:

digits.target


# In[13]:

digits.data.shape


# # 2. Learn & predict

# In[14]:

from sklearn import svm
clf = svm.SVC(gamma = 0.001, C = 100)


# In[15]:

clf.fit(digits.data[:-1], digits.target[:-1])


# In[16]:

clf.predict(digits.data[-1])


# # 3. Regression

# In[18]:

from sklearn import linear_model
clf = linear_model.LinearRegression()
clf.fit([[0,0], [1,1], [2,2]], [0,1,2])
clf.coef_


# # 4. Classification

# In[20]:

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target)
knn.predict([[0.1,0.2,0.3,0.4]])


# # 5. Cluster

# In[23]:

from sklearn import cluster, datasets
iris = datasets.load_iris()
k_means = cluster.KMeans(n_clusters = 3)
k_means.fit(iris.data)


# In[24]:

print(k_means.labels_[::10])


# In[25]:

print(iris.target[::10])


# ## Image compression

# In[65]:

from scipy import misc
import scipy as sp


# In[53]:

image = misc.ascent().astype(np.float32)
image.shape


# In[54]:

import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()


# In[55]:

X = image.reshape((-1,1))


# In[56]:

k_means = cluster.KMeans(n_clusters = 5)
k_means.fit(X)


# In[57]:

values = k_means.cluster_centers_.squeeze()


# In[60]:

labels = k_means.labels_
labels


# In[63]:

image_compressed = np.choose(labels, values)


# In[70]:

image_compressed.shape = image.shape
image_compressed.shape


# In[71]:

plt.imshow(image_compressed)
plt.show()


# # 6. Decompostion

# In[73]:

from sklearn import decomposition
pca = decomposition.PCA(n_components = 2)
pca.fit(iris.data)


# In[77]:

import pylab as pl
X = pca.transform(iris.data)
X.shape


# In[79]:

plt.scatter(X[:,0], X[:,1], c = iris.target)
plt.show()


# In[ ]:



