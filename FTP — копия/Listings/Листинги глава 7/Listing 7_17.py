# Listing 7_17
from imageai.Classification.Custom import ClassificationModelTrainer
import os

execution_path = os.getcwd()
# Путь к обучающей выборке
data_set = execution_path + "\\Im_Trening\\"

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsMobileNetV2()
model_trainer.setDataDirectory(data_set)
model_trainer.trainModel(num_objects=2,
                         num_experiments=15,
                         enhance_data=True,
                         show_network_summary=True)
