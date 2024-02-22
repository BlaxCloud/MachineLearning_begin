# Listing 7_9
from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()
# Путь к файлу с моделью сети
model_path = execution_path + "\\Model\\yolo-tiny.h5"
# Путь к файлу с видео
vide_path_in = execution_path + "\\Video\\traffic.mp4"
vide_path_out = execution_path + "\\Video\\traffic_detected"

detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()

video_path = detector.detectObjectsFromVideo(
    input_file_path=vide_path_in,
    output_file_path=vide_path_out,
    frames_per_second=20,
    log_progress=True)
print(video_path)