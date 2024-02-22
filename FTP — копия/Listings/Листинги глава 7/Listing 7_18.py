# Listing 7_18
from imageai.Classification.Custom import CustomImageClassification
import os

execution_path = os.getcwd()
# Путь к файлу с моделью сети
model_path = execution_path +\
             "\\Im_Trening\\models\\model_ex-004_acc-1.000000.h5"
# Путь к файлу json
json_path = execution_path + "\\Im_Trening\\json\\model_class.json"
# Путь к файлам с изображением
img_path = execution_path + "\\Images\\Znak3.jpg"
# img_path = execution_path + "\\Images\\Znak_Stop.jpg"

print(model_path)
print(json_path)
print(img_path)

prediction = CustomImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath(model_path)
prediction.setJsonPath(json_path)
prediction.loadModel(num_objects=2)

predictions, probabilities = prediction.classifyImage(img_path,
                                                      result_count=2)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction, " : ", eachProbability)
