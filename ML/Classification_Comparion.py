# -*- coding: utf-8 -*-
import pandas as pd
# 读取csv文件，并将第一行设为表头
data = pd.read_csv("class_data.csv",header=0)
print(data)

import matplotlib.pyplot as plt
# 绘制散点图
plt.scatter(data["X"], data["Y"], c=data['CLASS'])
plt.show()



''' scikit-learn 常用分类器对比试验 '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 加载色彩模块
from matplotlib.colors import ListedColormap
# 导入数据切分模块
from sklearn.cross_validation import train_test_split
# 导入准确度评估模块
from sklearn.metrics import accuracy_score

# 集成方法分类器
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# 高斯过程分类器
from sklearn.gaussian_process import GaussianProcessClassifier

# 广义线性分类器
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier

# KNN(K近邻)分类器
from sklearn.neighbors import KNeighborsClassifier

# 朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB

# 神经网络分类器
from sklearn.neural_network import MLPClassifier

# 决策树分类器
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# 支持向量机分类器
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


# 建立模型
model = [
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    GaussianProcessClassifier(),
    PassiveAggressiveClassifier(),
    RidgeClassifier(),
    SGDClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    MLPClassifier(),
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    SVC(),
    LinearSVC()
]

# 模型命名
classifier_Names = ['AdaBoost', 'Bagging', 'ExtraTrees', 'GradientBoosting', 'RandomForest','GaussianProcess', 'PassiveAggressive', 'Ridge', 'SGD', 'KNeighbors', 'GaussianNB', 'MLP', 'DecisionTree', 'ExtraTree', 'SVC', 'LinearSVC']

# 读取数据并划分
data = pd.read_csv("class_data.csv", header=0)

# 指定特征变量
feature = data[['X', 'Y']]
# 指定分类变量
target = data['CLASS']
# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3)

# 迭代模型
for name, clf in zip(classifier_Names,model):
    # 训练模型
    clf.fit(X_train, y_train)
    # 模型预测
    pre_labels = clf.predict(X_test)
    # 计算预测准确度
    score = accuracy_score(y_test, pre_labels)
    
print('%s:%.2f' % (name, score))



# 绘制数据集
i = 1
# 绘制热力图选择的样式
cm = plt.cm.Reds
# 为绘制训练集和测试集选择样式
cm_color = ListedColormap(['red', 'yellow'])

# 栅格化
x_min, x_max = data['X'].min() - 0.5, data['X'].max() + 0.5
y_min, y_max = data['Y'].min() - 0.5, data['Y'].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))


# 模型迭代
for name, clf in zip(classifier_Names, model):
    # 绘制子图
    ax = plt.subplot(4, 4, i);
    # 模型训练
    clf.fit(X_train, y_train)
    # 模型预测
    pre_labels = clf.predict(X_test)
    # 模型准确度
    score = accuracy_score(y_test, pre_labels)
    
    # 决策边界判断
    if hasattr(clf, "decisition_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        print("decision_function: ", clf)
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
        print("predict_proba: ", clf)
        
    # 绘制决策边界热力图
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.6)
    
    # 绘制训练集和测试集
    ax.scatter(X_train['X'], X_train['Y'], c=y_train, cmap=cm_color)
    ax.scatter(X_test['X'], X_test['Y'], c=y_test, cmap=cm_color, edgecolor='black')
    
    # 图形样式设定
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title('%s | %.2f' % (name, score))
    
    i += 1
    
# 显示图
plt.show()


    
    
    

