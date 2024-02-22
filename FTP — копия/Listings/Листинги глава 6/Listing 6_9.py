# Listing 6.9
# Модуль PyBr_Trener_Ful
import pickle
import matplotlib.pylab as plt
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer


net = buildNetwork(2, 3, 1)
y = net.activate([2, 1])
print('Y=', y)

ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))
print(ds)

trainer = BackpropTrainer(net)
trnerr, valerr = trainer.trainUntilConvergence(dataset=ds, maxEpochs=100)
plt.plot(trnerr, 'b', valerr, 'r')
plt.show()


# Проверка работы сети после обучения
y = net.activate([1, 1])
print('Y1=', y)

# запись сети в файл txt
fileObject = open('MyNet.txt', 'wb')
pickle.dump(net, fileObject)
fileObject.close()

# чтение сети из файла txt
fileObject = open('MyNet.txt', 'rb')
net2 = pickle.load(fileObject)

# Проверка работы загруженной из файла сети
y = net2.activate([1, 1])
print('Y2=', y)

