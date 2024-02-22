# Listing 8_1
import cv2

# загрузка изображения
img = cv2.imread('./Images/Test_Face.jpg')
# показать загруженной изображение
cv2.imshow('Input photo', img)

# загрузка каскада Хаара
classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# выполнение распознавания лиц
bboxes = classifier.detectMultiScale(img)
# формирование прямоугольника вокруг каждого обнаруженного лица
for box in bboxes:
    # формирование координат
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # рисование прямоугольников
    cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 1)

cv2.imshow('Window with face detection', img)  # показать обработанное изображение
cv2.imwrite('./Images/Test_Face_det.jpg', img)  # сохранить обработанное изображение

cv2.waitKey(0)  # держать окна с изображениями открытым
cv2.destroyAllWindows()  # закрыть все окна