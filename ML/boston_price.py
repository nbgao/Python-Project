# -*- coding: utf-8 -*

from sklearn import datasets
from sklearn.svm import LinearSVR
from matplotlib import pyplot as plt
from sklearn.cross_validation import cross_val_predict

boston = datasets.load_boston()

feature = boston.data
target = boston.target

model = LinearSVR()

predictions = cross_val_predict(model, feature, target, cv=10)

plt.scatter(target, predictions)
plt.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=4)
plt.xlabel("true_target")
plt.ylabel("prediction")
plt.show()
