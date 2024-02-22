# создание и обучение классификатора
# Это промежуточный код, он не является рабочим
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# Практическое использование классификатора
X_new = np.array([[5, 2.9, 1, 0.2]])
pr = knn.predict(X_new)
print("Вид цветка: {}".format(iris_dataset['target_names'][pr]))
