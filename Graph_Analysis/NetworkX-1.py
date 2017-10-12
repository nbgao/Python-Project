
# coding: utf-8

# ## 1. 基本操作

# In[1]:

from __future__ import print_function
import networkx as nx
import matplotlib.pyplot as plt


# In[2]:

G = nx.Graph()
edge = [(1,2), (1,3), (1,4), (2,3), (3,4), (5,4), (7,4)]

for e in edge:
    G.add_edge(*e)

nx.draw(G)
plt.show()


# In[3]:

print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())
print('Nodes:', G.nodes())
print('Edges:', G.edges())
print('Neighbors of Node 1:', G.neighbors(1))


# In[4]:

G.remove_node(7)
print(G.nodes())
print(G.edges())


# In[5]:

G.remove_edge(1,3)
print(G.nodes())
print(G.edges())


# In[6]:

nx.draw(G)
plt.show()


# ## 2. 为图中的元素添加属性

# In[9]:

G.graph['day'] = 'Monday'
print(G.graph)
G.node[1]['name'] = 'jilu'
print(G.nodes(data=True))


# In[10]:

G.add_edge(7, 8, weight=4.7)
G.add_edges_from([(3,8), (4,5)], color='red')
G.add_edges_from([(9,1, {'color': 'blue'}), (8,3,{'weight':8})])
G[1][2]['weight'] = 4.0
G.edge[1][2]['weight'] = 4
print(G.edges(data=True))


# ## 3. 有向图及节点的度数

# In[11]:

DG = nx.DiGraph()
DG.add_weighted_edges_from([(1,2,0.5), (3,1,0.75)])

print(DG.degree(1))
print(DG.out_degree(1))
print(DG.in_degree(1))


# ## 4. 构建图及图操作

# In[13]:

import networkx as nx
G = nx.Graph()
for e in [(1,2), (1,3), (2,3), (3,4), (5,4), (7,4)]:
    G.add_edge(*e)
nx.write_edgelist(G, "./graph_edges")
G1 = nx.read_edgelist("./graph_edges")
print(G1.edges())

