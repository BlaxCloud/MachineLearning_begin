# Listing 7_13
from imageai.Detection import VideoObjectDetection
import os


def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("Секунда : ", second_number)
    print("Массив выходных данных каждого кадра ", output_arrays)
    print("Массив подсчета уникальных объектов в каждом кадре: ", count_arrays)
    print("Среднее количество уникальных объектов в последнюю секунду: ",
          average_output_count)
    print("------------КОНЕЦ ДАННЫХ В ЭТОЙ СЕКУНДЕ --------------")


execution_path = os.getcwd()
# Путь к файлу с моделью сети
model_path = execution_path + "\\Model\\yolo.h5"
# Путь к файлам с видео
video_path_in = execution_path + "\\Video\\traffic.mp4"
video_path_out = execution_path + "\\Video\\video_second_analysis"

video_detector = VideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(model_path)
video_detector.loadModel()
video_detector.detectObjectsFromVideo(
    input_file_path=video_path_in,
    output_file_path=video_path_out,
    frames_per_second=20,
    per_second_function=forSeconds,
    minimum_percentage_probability=90)
