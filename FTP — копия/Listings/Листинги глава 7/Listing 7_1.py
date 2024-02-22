# Listing 7_1
from imageai.Classification import ImageClassification
import os

# Текущая директория
execution_path = os.getcwd()
# Путь к файлу с моделью сети
model_path = execution_path + "\\Model\\resnet50_imagenet_tf.2.0.h5"
# Путь к файлу с изображением
img_path = execution_path + "\\Images\\image1.jpg"

prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(model_path)
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(img_path, result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
