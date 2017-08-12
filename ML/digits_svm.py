# -*- coding: utf-8 -*

from sklearn import datasets
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

digits = datasets.load_digits()

# 绘制数据集前 5 个手写数字的灰度图
for index, image in enumerate(digits.images[:5]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()

# 训练手写数字识别模型
feature = digits.data
target = digits.target
images = digits.images

train_feature, test_feature, train_target, test_target, train_images, test_images = cross_validation.train_test_split(
    feature, target, images,
    test_size=0.33)

model = SVC(gamma=0.001)
model.fit(train_feature, train_target)
results = model.predict(test_feature)
scores = accuracy_score(test_target, results)

print scores

# 可视化查看前 4 项预测结果

images_labels_and_prediction = list(zip(test_images, test_target, results))

for index, (image, true_label, prediction_label) in enumerate(images_labels_and_prediction[:4]):
    plt.subplot(2, 4, index + 5)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xlabel("prediction:%i" % prediction_label)
    plt.title("True:%i" % true_label)

plt.show()
