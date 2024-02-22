# Listing 7_3
from imageai.Classification import ImageClassification
import os

# Текущая директория
execution_path = os.getcwd()
# Путь к файлу с моделью сети
model_path = execution_path + "\\Model\\resnet50_imagenet_tf.2.0.h5"

prediction = ImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath(os.path.join(execution_path, model_path))
prediction.loadModel()

# массив с файлами рисунков
all_images_array = []
all_files = os.listdir('.\\Images\\')  # формирование массива со всеми файлами
for each_file in all_files:            # выборка только рисунков
    if each_file.endswith(".jpg") or each_file.endswith(".png"):
        all_images_array.append(execution_path + "\\Images\\" + each_file)

for img_path in all_images_array:
    predictions, probabilities = prediction.classifyImage(
        os.path.join(execution_path, img_path), result_count=5)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction, " : ", eachProbability)
    print('-------------------------------')