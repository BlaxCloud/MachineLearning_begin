# Listing 6.3.1
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised import BackpropTrainer

net = buildNetwork(2, 3, 1)
y = net.activate([2, 1])

ds = SupervisedDataSet(2, 1)

xorModel = [
   [(0, 0), (0,)],
   [(0, 1), (1,)],
   [(1, 0), (1,)],
   [(1, 1), (0,)],
]

for input, target in xorModel:
    ds.addSample(input, target)

trainer = BackpropTrainer(net, ds)
print(trainer.train())

