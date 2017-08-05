from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_feature = iris.data
iris_target = iris.target

feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33,
                                                                          random_state=56)
dt_model = DecisionTreeClassifier()
dt_model.fit(feature_train, target_train)
predict_results = dt_model.predict(feature_test)
scores = dt_model.score(feature_test, target_test)

print predict_results
print target_test

print accuracy_score(predict_results, target_test)
print scores
