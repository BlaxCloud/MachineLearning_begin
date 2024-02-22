# Listing 7_19
from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hololens")
trainer.setTrainConfig(object_names_array=["hololens"],
                       batch_size=4,
                       num_experiments=200,
                       train_from_pretrained_model="./Model/pretrained-yolov3.h5")
trainer.trainModel()
