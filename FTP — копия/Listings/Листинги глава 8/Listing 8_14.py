# Listing 8_14
# Это промежуточный модуль, он не является рабочим
# Путь к фотографиям
path = './YaleFace/yalefaces/'
# Получаем лица и соответствующие им номера
images, labels = get_images(path)
cv2.destroyAllWindows()

# Обучаем программу распознавать лица
recognizer.train(images, np.array(labels))
# Сохраняем результат тренировки
recognizer.write('./YaleFace/Yale_face2.yml')
print('Обучение закончено')
