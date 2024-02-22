class Car(object):
    # Наименование класса
    Name_class = "Автомобиль"

    def __init__(self, brand, weight, power):
        self.brand = brand    # Марка, модель автомобиля
        self.weight = weight  # Вес автомобиля
        self.power = power    # Мощность двигателя

    # Метод двигаться прямо
    def drive(self):
        # Здесь команды двигаться прямо
        print("Поехали, двигаемся прямо!")

    # Метод повернуть направо
    def righ(self):
        # Здесь команды повернуть руль направо
        print("Едем, поворачиваем руль направо!")

    # Метод повернуть налево
    def left(self):
        # Здесь команды повернуть руль налево
        print("Едем, поворачиваем руль налево!")

    # Метод тормозить
    def brake(self):
        # Здесь команды нажатия на педаль тормоза
        print("Стоп, активируем тормоз")

    # Метод подать звуковой сигнал
    def beep(self):
        # Здесь команды подачи звукового сигнала
        print("Подан звуковой сигнал")
