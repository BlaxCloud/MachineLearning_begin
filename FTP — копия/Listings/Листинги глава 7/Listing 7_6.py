# Listing 7_6
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
# Путь к файлу с моделью сети
model_path = execution_path + "\\Model\\yolo.h5"
# Путь к файлу с изображением
img_path_in = execution_path + "\\Images\\image5.jpg"
img_path_out = execution_path + "\\Images\\image_out5.jpg"

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()
custom = detector.CustomObjects(person=True, dog=True)
detections = detector.detectObjectsFromImage(
    custom_objects=custom,
    input_image=img_path_in,
    output_image_path=img_path_out,
    minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"],
          " : ", eachObject["box_points"])
    print("--------------------------------")