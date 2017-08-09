# -*- coding: utf-8 -*

from matplotlib import pyplot as plt
from sklearn.cluster import k_means
import pandas as pd

file = pd.read_csv("cluster_data.csv", header=0)

index = []
inertia = []

for i in range(9):
    model = k_means(file, n_clusters=i + 1)
    index.append(i + 1)
    inertia.append(model[2])

plt.plot(index, inertia, "-o")
plt.show()