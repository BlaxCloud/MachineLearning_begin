# Listing 4.21_1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Описание класса Perceptron
class Perceptron(object):
    '''
    Классификатор на основе персептрона.
    Параметры
    eta:float  - Темп обучения (между 0.0 и 1.0)
    n_iter:int - Проходы по тренировочному набору данных.
    Атрибуты
    w_: 1-мерный массив - Весовые коэффициенты после подгонки.
    errors_: список - Число случаев ошибочной классификации в каждой эпохе.
    '''

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    '''
    Выполнить подгонку модели под тренировочные данные.
    Параметры
    X : массивоподобный, форма = [n_sam ples, n_features] тренировочные векторы, где 
                                    n_samples - число образцов и
                                    n _features - число признаков, 
    у : массивоподобный, форма = [n_samples] Целевые значения.
    Возвращает
    self: object
    '''

    def fit(self, x, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    '''Рассчитать чистый вход'''

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    '''Вернуть метку класса после единичного скачка'''

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# Загрузка  из сети интернет масива - 150 элеентов, загрузка их в объект DataFrame  и печать
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print('Массив')
print(df.to_string())

# выборка из объекта DF 100 элементов (столбец 4 название цветков) и загрузка его в одномерный массив Y и печать
y = df.iloc[0:100, 4].values
print('Значение четвертого столбца Y - 100')
print(y)

# Преобразование названий цветков (столбец 4) в одномерный массив (вектор) из -1 и 1
y = np.where(y == 'Iris-setosa', -1, 1)
print('Значение названий цветков  в виде -1 и 1, Y - 100')
print(y)

# выборка из объекта DF массива 100 элементов (столбец 0 и столбец 2), загрузка его в массив X (иатрица) и печать
X = df.iloc[0:100, [0, 2]].values
print('Значение X - 100')
print(X)
print('Конец X')

# Формирование параметров значений для вывода на график
# Первые 50 элементов (Строки 0-50, столбцы 0,1)
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='щетинистый')
# Следующие 50 элементов (Строки 50-100, столбцы 0,1)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный')

# Формировние названий осей и вывод графика на экран
plt.xlabel('длина чашелистика')
plt.ylabel('длина лепестка')
plt.legend(loc='upper left')
plt.show()

# Тренировка
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
# число ошибочно классифицированных случаев во время обновлений
plt.ylabel('Число случаев ошибочной классификации')
plt.show()

i1 = [5.5, 1.6]
i2 = [6.4, 4.5]
R1 = ppn.predict(i1)
R2 = ppn.predict(i2)
print('R1=', R1, '  R2=', R2)

if R1 == 1:
    print('R1= Вид Iris setosa')
else:
    print('R1= Вид Iris versicolor')
