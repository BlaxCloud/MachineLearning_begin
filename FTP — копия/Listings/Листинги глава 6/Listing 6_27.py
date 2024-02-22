# Listing 6.27
from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('Тренировочный набор данных', X_train.shape)
print('Метки тренировочного набора данных', y_train.shape)
print('Тестовый набор данных', X_test.shape)
print('Метки тестового набора данных', y_test.shape)

plt.imshow(X_train[1])
plt.show()
