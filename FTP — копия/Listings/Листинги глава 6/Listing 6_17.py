# Модуль SciLearn0
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                   iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
z = knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])
pr = knn.predict(X_new)
print("Метка вида цветка: {}".format(pr))
print("Вид цветка: {}".format(iris_dataset['target_names'][pr]))
