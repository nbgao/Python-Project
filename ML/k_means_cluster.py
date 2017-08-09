# -*- coding: utf-8 -*

from matplotlib import pyplot as plt
from sklearn.cluster import k_means
import pandas as pd
from sklearn.metrics import silhouette_score

file = pd.read_csv("cluster_data.csv", header=0)
X = file['x']
y = file['y']


def k_number():
    index = []
    inertia = []
    silhouette = []
    for i in range(20):
        model = k_means(file, n_clusters=i + 2)
        inertia.append(model[2])
        index.append(i + 2)
        silhouette.append(silhouette_score(file, model[1]))
    print silhouette 
    plt.plot(index, silhouette, "-o")
    plt.plot(index, inertia, "-o")
    plt.show()


def k_means_iris(n_cluster):
    model = k_means(file, n_clusters=n_cluster)
    cluster_centers = model[0]
    cluster_labels = model[1]
    cluster_inertia = model[2]
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.scatter(X, y, c="grey")
    ax2.scatter(X, y, c=cluster_labels)

    for center in cluster_centers:
        ax2.scatter(center[0], center[1], marker="p", edgecolors="red")
    print "cluster_inertia: %s" % cluster_inertia

    plt.show()


if __name__ == '__main__':
    k_number()
    k_means_iris(int(input("Input clusters: ")))