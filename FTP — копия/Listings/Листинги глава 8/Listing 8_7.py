# Listing 8_7
import cv2

classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Активация камеры
# video_capture = cv2.VideoCapture(0)  # Активация камеры

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Получаем серую картинку
    plaques = classifier.detectMultiScale(gray, 1.3, 5)
    for i, (x, y, w, h) in enumerate(plaques):
        roi_color = frame[y:y + h, x:x + w]
        cv2.putText(frame,
                    str(x) + " " + str(y) + " " + str(w) + " " + str(h),
                    (480, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255))
        r = 400.0 / roi_color.shape[1]
        dim = (400, int(roi_color.shape[0] * r))
        resized = cv2.resize(roi_color, dim,
                             interpolation=cv2.INTER_AREA)
        w_resized = resized.shape[0]
        h_resized = resized.shape[1]
        # Собираем в основную картинку
        frame[100:100 + w_resized, 100:100 + h_resized] = resized
        cv2.rectangle(roi_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite('Images/Video_Znak.jpg', resized)  # сохранить только знак
    # Отображение результирующего кадра
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite('Images/Video_Znak_det.jpg', frame)  # сохранить автомобиль и знак

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

# деактивировать камеру, закрыть все окна
video_capture.release()
cv2.destroyAllWindows()
