# -*- coding: utf-8 -*

from matplotlib import pyplot as plt
from sklearn.cluster import k_means
import pandas as pd
from sklearn.metrics import silhouette_score

file = pd.read_csv("cluster_data.csv", header=0)

index = []
silhouette = []

for i in range(8):
    model = k_means(file, n_clusters=i + 2)
    index.append(i + 2)
    silhouette.append(silhouette_score(file, model[1]))

plt.plot(index, silhouette, "-o")
plt.show()