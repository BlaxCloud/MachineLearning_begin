# Listing 7_2
from imageai.Classification import ImageClassification
import os

# Текущая директория
execution_path = os.getcwd()
# Путь к файлу с моделью сети
model_path = execution_path + "\\Model\\inception_v3_weights_tf_dim_ordering_tf_kernels.h5"
# Путь к файлу с изображением
img_path = execution_path + "\\Images\\image1.jpg"
# img_path = execution_path + "\\Images\\image4.jpg"

prediction = ImageClassification()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath(model_path)
prediction.loadModel()

predictions, probabilities = prediction.classifyImage(img_path, result_count=5)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)