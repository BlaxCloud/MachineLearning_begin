# Listing 6.49
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

train_images = train_images / 255.0
test_images = test_images / 255.0

# Модель нейронной сети
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10, activation='softmax')
])

# Компиляция модели нейронной сети
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Тренировка (обучение) модели
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nТочность на проверочных данных:', test_acc)


