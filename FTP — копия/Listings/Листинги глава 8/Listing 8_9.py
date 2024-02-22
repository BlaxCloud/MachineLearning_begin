# Listing 8_9
import cv2

# загрузка фотографии
img = cv2.imread('./Images/Test4.jpg')
cv2.imshow('Input photo', img)
# загрузка предварительно обученной модели
classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")
# classifier = cv2.CascadeClassifier(
    # cv2.data.haarcascades + "haarcascade_upperbody.xml")
# classifier = cv2.CascadeClassifier(
    # cv2.data.haarcascades + "haarcascade_lowerbody.xml")
# classifier = cv2.CascadeClassifier(
    # cv2.data.haarcascades + "haarcascade_righteye_2splits.xml")
# classifier = cv2.CascadeClassifier(
    # cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml")

# выполнение распознавания объектов
bboxes = classifier.detectMultiScale(img)
# формирование прямоугольника вокруг каждого обнаруженного объекта
for box in bboxes:
    # формирование координат
    x, y, width, height = box
    x2, y2 = x + width, y + height
    # рисование прямоугольников
    cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 1)

cv2.imshow('Window with object detection', img)  # показать
cv2.imwrite('./Images/Test4_det.jpg', img)      # сохранить

cv2.waitKey(0)    # держать окно с изображением открытым
cv2.destroyAllWindows()  # закрыть все окна
