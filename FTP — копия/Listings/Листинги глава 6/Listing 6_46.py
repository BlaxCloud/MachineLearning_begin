# Listing 6.46
import matplotlib.pyplot as plt
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

Tr_Im = train_images.shape
Tr_label = len(train_labels)
Labels = train_labels
print('Тренировочный массив изображений', Tr_Im)
print('Тренировочный массив меток', Tr_label)
print('Метки изображений', Labels)

Test_Im = test_images.shape
Test_label = len(test_labels)
print('Тестовый массив изображений', Test_Im)
print('Тестовый массив меток', Test_label)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
