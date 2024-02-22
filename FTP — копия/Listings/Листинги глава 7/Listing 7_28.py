# # Listing 7.28
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("./Im_Yolo_Trening/models/detection_model-ex-002--loss-0039.096.h5")
detector.setJsonPath("./Im_Yolo_Trening/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(
    input_image="./Images/Znak_Zebra.jpg",
    output_image_path="./Images/Znak_Zebra_New.jpg",)
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"],
          " : ", detection["box_points"])
