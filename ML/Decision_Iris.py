# -*- coding: utf-8 -*-

# 导入数据集
from sklearn import datasets

iris = datasets.load_iris()     #加载iris数据集
iris_feature = iris.data        #特征数据
iris_target = iris.target       #分类数据


# 导入交叉验证库
from sklearn.cross_validation import train_test_split

feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size = 0.33, random_state = 42)
'''
参数说明
feature_train 训练集特征
feature_test  测试集特征
target_train  训练集目标值
target_test   验证集目标值
test_size     划分到测试集数据占全部数据的占比
random_state  乱序程度
'''

print(target_train)


# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 所有参数均置为默认状态
data_model = DecisionTreeClassifier() 
data_model
'''
DecisionTreeClassifier函数参数说明
criterion = gini/entropy 选择基尼指数或熵来做损失函数
max_depth = int 用来控制决策树的最大深度，防止模型出现过拟合
min_samples_leaf = int 用来设置叶节点上的最少样本数量，用于对树进行修剪
splitter = best/random 每个节点的分类策略 “最佳”或“随机”
'''
# 使用训练集训练模型
data_model.fit(feature_train, target_train)
#使用模型对测试集进行预测
predict_results  = data_model.predict(feature_test)

# 将预测结果和验证集目标值输出，对照比较
print(predict_results, target_test)

from sklearn.metrics import accuracy_score
# accuracy_score预测结果的准确度
accuracy_score(predict_results, target_test)
# DTC自带score方法
scores = data_model.score(feature_test, target_test)
scores


# 调整DTC模型参数
data_model_1 = DecisionTreeClassifier('entropy','random')
data_model_1.fit(feature_train, target_train)
scores1 = data_model_1.score(feature_test, target_test)
scores1