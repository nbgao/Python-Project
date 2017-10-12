
# coding: utf-8

# # 使用NetworkX进行图分析

# ## 1. 利用联通子图发现社区

# In[10]:

from __future__ import print_function
import networkx as nx
from networkx import read_edgelist

G = read_edgelist('./hartford_drug.edgelist')
print(G.number_of_nodes())
print(G.number_of_edges())

import matplotlib.pyplot as plt
nx.draw(G)
plt.show()


# In[14]:

from networkx.algorithms import number_connected_components, connected_components
print(number_connected_components(G))
for subG in connected_components(G):
    print(subG)


# In[15]:

from networkx.algorithms import connected_component_subgraphs

for i, subG in enumerate(connected_component_subgraphs(G)):
    print('G%s' % i, subG.number_of_nodes(), subG.number_of_edges())


# ## 2. 通过三角计算强化社区发现

# In[16]:

from networkx.algorithms import triangles, transitivity, average_clustering

print(triangles(G))
print(transitivity(G))
print(average_clustering(G))


# ## 3. 利用PageRank发现影响力中心

# In[19]:

from collections import Counter
from networkx.algorithms import pagerank

pr = pagerank(G)
for p in Counter(pr).most_common():
    print(p)

