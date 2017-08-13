# -*- coding: utf-8 -*-
''' 线性分类器 '''
import pandas as pd

# 读取csv数据文件
df = pd.read_csv("data.csv", header=0)
df.head(8)

# 导入数据集划分模块
from sklearn.cross_validation import train_test_split
# 读取特征值及目标值
feature = df[['x', 'y']]
target = df['class']

# 对数据集进行分割
train_feature, test_feature, train_target, test_target = train_test_split(feature, target, train_size=0.77)


# 导入感知分类器
from sklearn.linear_model import Perceptron
# 导入线性支持向量机分类器
from sklearn.svm import LinearSVC

# 构造感知机预测模型
model = Perceptron()
model.fit(train_feature, train_target)

# 构造线性支持向量机分类模型
model2 = LinearSVC()
model2.fit(train_feature, train_target)

# 感知机分类预测准确度
print(model.score(test_feature, test_target))
# 支持向量机分类准确度
print(model2.score(test_feature, test_target))


''' 非线性支持向量机 '''
# 导入数据集模块
from sklearn import datasets
import matplotlib.pyplot as plt

# 载入数据集
digits = datasets.load_digits()

for index, image in enumerate(digits.images[:5]):
    plt.subplot(2,5,index+1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    
plt.show()

print(digits.target[:5])
print(digits.images[1])



# 导入交叉验证模块
from sklearn import cross_validation
# 导入非线性支持向量机分类器
from sklearn.svm import SVC
# 导入评估模块
from sklearn.metrics import accuracy_score

# 指定特征
feature = digits.data
# 指定目标值
target = digits.target

# 划分数据集，将其中70%划为训练集，另30%作为测试集
train_feature, test_feature, train_target, test_target = cross_validation.train_test_split(feature, target, test_size=0.33)

# 建立模型
model = SVC()
# 模型训练
model.fit(train_feature, train_target)
# 模型预测
results = model.predict(test_feature)

# 评估预测精确度
scores = accuracy_score(test_target, results)
print(scores)

model2 = SVC(gamma=0.001)
model2.fit(train_feature, train_target)
results2 = model2.predict(test_feature)
scores2 = accuracy_score(test_target, results2)
print(scores2)



''' 支持向量机回归 '''
# 导入数据集模块
from sklearn import datasets
# 导入波士顿房产数据集
boston = datasets.load_boston()
# 输出数据集介绍文档
print(boston.DESCR)

# 导入线性支持向量机回归模块
from sklearn.svm import LinearSVR
# 导入交叉验证模块
from sklearn.cross_validation import cross_val_predict

feature = boston.data
target = boston.target

# 建立线性支持向量机回归模型
model = LinearSVR()
# 交叉验证,数据集等分为10份
predictions = cross_val_predict(model, feature, target, cv=10)


import matplotlib.pyplot as plt
# 绘制散点图
plt.scatter(target, predictions)
# 绘制45°参考线
plt.plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=2)
# 设置坐标轴名称
plt.xlabel('true_target')
plt.ylabel('prediction')

plt.show()
