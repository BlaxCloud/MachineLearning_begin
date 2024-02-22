# Listing 6.41
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np

model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(learning_rate=0.1))
print(model.summary())

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
model.fit(X, y, batch_size=1, epochs=1000, verbose=0)

print("Проверка работы обученной сети:")
print("XOR(0,0):", model.predict(np.array([[0, 0]])))
print("XOR(0,1):", model.predict(np.array([[0, 1]])))
print("XOR(1,0):", model.predict(np.array([[1, 0]])))
print("XOR(1,1):", model.predict(np.array([[1, 1]])))
