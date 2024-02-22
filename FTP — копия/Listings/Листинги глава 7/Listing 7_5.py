# Listing 7_5
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
# Путь к файлу с моделью сети
model_path = execution_path + "\\Model\\resnet50_coco_best_v2.1.0.h5"
# Путь к файлу с изображением
img_path_in = execution_path + "\\Images\\image5.jpg"
img_path_out = execution_path + "\\Images\\image_out5.jpg"

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(model_path)
detector.loadModel()
detections = detector.detectObjectsFromImage(
    input_image=img_path_in,
    output_image_path=img_path_out,
    minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"],
          " : ", eachObject["box_points"])
    print("--------------------------------")