# Listing 4.13
# Это промежуточный код, он не являтся рабочим
# Тренировка сети
for i in range(100000):
    # Получить случайную X координату точки
    x = random.choice(list(data.keys()))

    # Получить соответствующую Y координату точки
    true_result = data[x]

    # Получить ответ сети
    out = proceed(x)

    # Считаем ошибку сети
    delta = true_result - out

    # Меняем вес при x в соответствии с дельта-правилом
    k += delta*rate*x

    # Меняем вес при постоянном входе в соответствии с дельта-правилом
    c += delta*rate
