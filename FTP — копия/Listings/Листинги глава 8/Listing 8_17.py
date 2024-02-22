# Listing 8_17
import cv2
import os
import numpy as np
from PIL import Image

# Загрузка каскадов Хаара для поиска лиц
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Формирование локального бинарного шаблона
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./YaleFace/Yale_face2.yml')

path = './YaleFace/yalefaces/'
print(path)
# Создаем список фотографий для распознавания
image_paths = [os.path.join(path, f)
               for f in os.listdir(path) if f.endswith('.happy')]

for image_path in image_paths:  # Ищем лица на фотографиях
    gray = Image.open(image_path).convert('L')
    image = np.array(gray, 'uint8')
    faces = faceCascade.detectMultiScale(image,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Если лица найдены, пытаемся распознать их.
        # Функция recognizer.predict в случае успешного распознавания
        # возвращает номер и параметр confidence.
        # Этот параметр указывает на уверенность алгоритма,
        # что это именно тот человек. Чем он меньше,
        # тем больше уверенность
        number_predicted, conf = recognizer.predict(
            image[y: y + h, x: x + w])

        # Извлекаем настоящий номер человека на фото и сравниваем с тем,
        # что выдал алгоритм
        number_actual = int(os.path.split(image_path)[1].split(".")[0].
                            replace("subject", ""))

        if number_actual == number_predicted:
            print("{} is Correctly Recognized with confidence {}".
                  format(number_actual, conf))
        else:
            print("{} is Incorrect Recognized as {}".
                  format(number_actual, number_predicted))

        cv2.imshow("Recognizing Face", image[y: y + h, x: x + w])
        cv2.waitKey(1000)

cv2.destroyAllWindows()
