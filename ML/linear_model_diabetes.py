# -*- coding: utf-8 -*

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn import cross_validation

diabetes = datasets.load_diabetes()

diabetes_feature = diabetes.data[:, np.newaxis, 2]

diabetes_target = diabetes.target

train_feature, test_feature, train_target, test_target = cross_validation.train_test_split(diabetes_feature, diabetes_target, test_size=0.33, random_state=56)

model = linear_model.LinearRegression()
model.fit(train_feature, train_target)

plt.scatter(train_feature, train_target,  color='black')
plt.scatter(test_feature, test_target,  color='red')
plt.plot(test_feature, model.predict(test_feature), color='blue',
         linewidth=3)

plt.legend(('Fit line', 'Train Set', 'Test Set'), loc='lower right')
plt.title('LinearRegression Example')

plt.xticks(())
plt.yticks(())

plt.show()

