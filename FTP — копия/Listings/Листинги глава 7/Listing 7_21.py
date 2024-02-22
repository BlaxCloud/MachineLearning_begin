# Listing 7_21
from imageai.Detection.Custom import CustomObjectDetection
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("./hololens/models/hololens-ex-60--loss-2.76.h5")
detector.setJsonPath("./hololens/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(
    input_image="./Images/holo1.jpg",
    output_image_path="./Images/holo1-detected.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"],
                             " : ", detection["box_points"])
