# Listing 4.9
# Модуль Urok2
import random

# Обучающая выборка (идеальное изображение цифр от 0 до 9)
num0 = list('111101101101111')
num1 = list('001001001001001')
num2 = list('111001111100111')
num3 = list('111001111001111')
num4 = list('101101111001001')
num5 = list('111100111001111')
num6 = list('111100111101111')
num7 = list('111001001001001')
num8 = list('111101111101111')
num9 = list('111101111001111')
# Список всех цифр от 0 до 9 в едином массиве
nums = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]

tema = 5       # какой цифре обучаем
n_sensor = 15  # количество сенсоров
weights = [0 for i in range(n_sensor)]  # ОБнуление весов


# Функция определения - является ли полученное изображение числом 5
# возвращает Да, если признано, что это 5. Нет, если отвергнуто, что это 5
def perceptron(Sensor):
    b = 7  # Порог функции активации
    s = 0  # Начальное значение суммы
    for i in range(n_sensor):  # цикл суммирования сигналов от сенсоров
        s += int(Sensor[i]) * weights[i]
    if s >= b:
        return True  # Сумма превысила порог
    else:
        return False  # Сумма меньше порога


# Уменьшение значений весов
# Если сеть ошиблась и выдала Да при входной цифре, отличной от пятерки
def decrease(number):
    for i in range(n_sensor):
        if int(number[i]) == 1:  # Если вход возбужден
            weights[i] -= 1  # Уменьшаем связанный с входом вес на единицу


# Увеличение значений весов
# Если сеть ошиблась и выдала Нет при поданной на вход цифре 5
def increase(number):
    for i in range(n_sensor):
        if int(number[i]) == 1:  # Если вход возбужден
            weights[i] += 1  # Увеличиваем связанный с входом вес на единицу

# Тренировка сети
n = 100  # количество уроков
for i in range(n):
    j = random.randint(0, 9)  # Генерируем случайное число j от 0 до 9
    r = perceptron(nums[j])  # Результат обращения к сумматору (ответ - Да или НЕТ)

    if j != tema:  # Если генератор выдал случайное число j не равное 5
        if r:  # Если сумматор сказал True (ДА-это пятерка), а j это не пятерка
            decrease(nums[j])  # Ошибка первого типа, уменьшаем значимые веса

    else:  # Если генератор выдал случайное число j равное 5
        if not r:  # Если сумматор сказал False (НЕТ-это не пятерка), а на самом деле j=5
            increase(nums[tema])  # Ошибка второго типа, увеличиваем значимые веса

print(j)
print(weights)  # Вывод значений весов


'''
# проверка работы программы на обучающей выборке
print("0 это 5? ", perceptron(num0))
print("1 это 5? ", perceptron(num1))
print("2 это 5? ", perceptron(num2))
print("3 это 5? ", perceptron(num3))
print("4 это 5? ", perceptron(num4))
print("5 это 5? ", perceptron(num5))
print("6 это 5? ", perceptron(num6))
print("7 это 5? ", perceptron(num7))
print("8 это 5? ", perceptron(num8))
print("9 это 5? ", perceptron(num9))


# Тестовая выборка (различные варианты изображения цифры 5)
num51 = list('111100111000111')
num52 = list('111100010001111')
num53 = list('111100011001111')
num54 = list('110100111001111')
num55 = list('110100111001011')
num56 = list('111100101001111')

print("+++++++++++++")
# Прогон по тестовой выборке
# print("Узнал 5 в 5? ", perceptron(num5))
print("Узнал 5 в 51? ", perceptron(num51))
print("Узнал 5 в 52? ", perceptron(num52))
print("Узнал 5 в 53? ", perceptron(num53))
print("Узнал 5 в 54? ", perceptron(num54))
print("Узнал 5 в 55? ", perceptron(num55))
print("Узнал 5 в 56? ", perceptron(num56))
'''