# Listing 5.7
# Это промежуточный код, он не является рабочим
# Делаем предсказания
emily = np.array([-7, -3])  # 128 фунтов, 63 дюйма
frank = np.array([20, 2])   # 155 фунтов, 68 дюймов
print("Emily: %.3f" % network.feedforward(emily))
print("Frank: %.3f" % network.feedforward(frank))
