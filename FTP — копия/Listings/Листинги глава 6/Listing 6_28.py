from keras.datasets import mnist
import matplotlib.pyplot as plt
(X_train, y_train), (X_test, y_test) = mnist.load_data()
plt.figure()
plt.imshow(X_train[1])
plt.colorbar()
plt.grid(False)
plt.show()
