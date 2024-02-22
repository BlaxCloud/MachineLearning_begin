# Listing 8_8
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # активация камеры
car_cascade = cv2.CascadeClassifier('./XML/cars.xml')  # загрузка классификатора

# цикл обработки кадров
while True:
    ret, frames = cap.read()  # читает кадры из видео
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)    # оттенки серого
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)  # обнаруживает автомобили

    # Нарисовать прямоугольник в найденном авто
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('video', frames)  # отображать обработанные кадры в окне
    if cv2.waitKey(33) == 27:    # завершить цикл, если нажата клавиша Esc
        break

cv2.destroyAllWindows()  # закрыть все окна
