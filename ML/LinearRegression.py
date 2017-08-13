# -*- coding: utf-8 -*-

from sklearn import linear_model
# 调用最小二乘回归方法
model = linear_model.LinearRegression()
# 模型拟合
model.fit([[0,0], [1,1], [2,2]], [0,1,2])
print(model.coef_)
print(model.intercept_)
print(model.predict([3,3]))


from sklearn import datasets
from sklearn import cross_validation
import numpy as np
diabetes = datasets.load_diabetes()
diabetes_feature = diabetes.data[:,np.newaxis,2]
diabetes_target = diabetes.target

train_feature, test_feature, train_target, test_target = cross_validation.train_test_split(diabetes_feature, diabetes_target, test_size=0.33, random_state=56)

from  sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(train_feature, train_target)

import matplotlib.pyplot as plt
plt.scatter(train_feature, train_target, color='c')
plt.scatter(test_feature, test_target, color='b')
plt.plot(test_feature, model.predict(test_feature), color='r', lw=2)
# 绘制图例
plt.title('Linear Regression Examplt')
plt.legend(('Fit line', 'Train Set', 'Test Set'), loc='lower right')
plt.xticks(())
plt.yticks(())
plt.show()


''' 
贝叶斯岭回归 sklearn.linear_model.BayesianRidge
Lassor回归 sklearn.linear_model.Lasso()
岭回归 sklearn.linear_model.Ridge()
随机梯度下降回归sklearn.linear_model.SGDRegressor()
鲁棒回归 sklearn.linead_model.HuberRegressor()
'''

from sklearn.datasets import make_classification
X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)

import matplotlib.pyplot as plt
plt.scatter(X1[:,0], X1[:,1], marker='o', c=Y1)
plt.show()

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Perceptron

train_feature, test_feature, train_target, test_target = train_test_split(X1, Y1, train_size=0.77, random_state=56)
model = Perceptron()
model.fit(train_feature, train_target)
results = model.predict(test_feature)
plt.scatter(test_feature[:,0], test_feature[:,1], marker=',')

# 将预测结果用标签样式标注在测试数据左上方
for i, txt in enumerate(results):
    plt.annotate(txt, (test_feature[:,0][i], test_feature[:,1][i]))
 
plt.show()