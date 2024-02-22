# Listing 7_10
from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()
# Путь к файлу с моделью сети
model_path = execution_path + "\\Model\\yolo.h5"
# Путь для записи обработанного видео файла
video_path_out = execution_path + "\\Video\\camera_detected_video"

# для встроенной камеры ноутбука
# camera = cv2.VideoCapture(0)

# для подключенной внешней web камеры
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()

video_path = detector.detectObjectsFromVideo(camera_input=camera,
    output_file_path=video_path_out,
    frames_per_second=20,
    log_progress=True,
    minimum_percentage_probability=30)

print(video_path)
