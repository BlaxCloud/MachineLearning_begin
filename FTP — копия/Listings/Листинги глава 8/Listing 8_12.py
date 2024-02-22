# Listing 8_12
# Это промежуточный модуль, он не является рабочим
import cv2
import os
import numpy as np
from PIL import Image

# Загрузка каскадов Хаара для поиска лиц
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Формирование локального бинарного шаблона
recognizer = cv2.face.LBPHFaceRecognizer_create(1, 8, 8, 8, 123)
