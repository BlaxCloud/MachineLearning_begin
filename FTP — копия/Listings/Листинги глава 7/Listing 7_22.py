# Listing 7_22
from imageai.Detection.Custom import CustomVideoObjectDetection

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("./hololens/models/hololens-ex-60--loss-2.76.h5")
video_detector.setJsonPath("./hololens/json/detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(
    input_file_path="./Video/holo.mp4",
    output_file_path="./Video/holo1-detected",
    frames_per_second=20,
    minimum_percentage_probability=40, log_progress=True)
