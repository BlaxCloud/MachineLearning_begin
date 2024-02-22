# Listing 8_5
import cv2

# Загрузка изображения
img = cv2.imread('./Images/Test_Numer.jpg')
# показать загруженной изображение
cv2.imshow('Input photo', img)
# загрузка каскада Хаара
classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
# Выполнение распознавания объектов
bboxes = classifier.detectMultiScale(img)
# формирование прямоугольника вокруг каждого обнаруженного объекта
for box in bboxes:
    # формирование координат
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # рисование прямоугольников
    cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 3)

cv2.imshow('Window with object detection', img)  # показать изображение
cv2.imwrite('./Images/Test_Numer_det.jpg', img)  # сохранить изображение

cv2.waitKey(0)  # держать окно с изображениями открытым
cv2.destroyAllWindows()   # закрыть все окна
