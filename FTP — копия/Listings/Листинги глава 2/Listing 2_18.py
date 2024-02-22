class Cat:
    Name_Class = "Кошки"

    # Действия, которые надо выполнять при создании объекта "Кошка"
    def __init__(self, wool_color, eyes_color, name):
        self.wool_color = wool_color
        self.eyes_color = eyes_color
        self.name = name

    # Мурлыкать
    def purr(self):
        print("Муррр!")

    # Шипеть
    def hiss(self):
        print("Шшшш!")

    # Царапаться
    def scrabble(self):
        print("Цап-царап!")
