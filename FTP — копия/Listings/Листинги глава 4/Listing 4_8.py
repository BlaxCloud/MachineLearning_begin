# Listing 4.8
# Это промежуточный код, он не является рабочим
# Тренировка сети
n = 1000  # количество уроков
for i in range(n):
    j = random.randint(0, 9)  # Генерируем случайное число j от 0 до 9
    r = perceptron(nums[j])  # Результат обращения к сумматору (ответ - Да или НЕТ)

    if j != tema:  # Если генератор выдал случайное число j не равное 5
        if r:  # Если сумматор сказал True (ДА-это пятерка), а j это не пятерка
            decrease(nums[j])  # Ошибка первого типа, уменьшаем значимые веса

    else:  # Если генератор выдал случайное число j равное 5
        if not r:  # Если сумматор сказал False (НЕТ-это не пятерка), а на самом деле j=5
            increase(nums[tema])  # Ошибка второго типа, увеличиваем значимые веса
