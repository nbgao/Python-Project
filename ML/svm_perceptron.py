# -*- coding: utf-8 -*

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import train_test_split

df = pd.read_csv("data.csv", header=0)

x = df[["x", "y"]]
y = df["class"]

train_feature, test_feature, train_target, test_target = train_test_split(x, y, train_size=0.77, random_state=56
                                                                          )

model = Perceptron()
model.fit(train_feature, train_target)
results = model.score(test_feature,test_target)

model2 = LinearSVC()
model2.fit(train_feature, train_target)
results2 = model2.score(test_feature,test_target)

print(results)
print(results2)
