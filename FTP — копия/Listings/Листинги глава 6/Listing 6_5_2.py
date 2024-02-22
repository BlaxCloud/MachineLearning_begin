# Listing 6_5_2
import matplotlib.pylab as plt
from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer

net = buildNetwork(2, 3, 1)
y = net.activate([2, 1])

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

for mod in net.modules:
    print("Module:", mod.name)
    if mod.paramdim > 0:
        print("--parameters:", mod.params)
    for conn in net.connections[mod]:
        print("-connection to", conn.outmod.name)
        if conn.paramdim > 0:
            print("- parameters", conn.params)
    if hasattr(net, "recurrentConns"):
        print("Recurrent connections")
        for conn in net.recurrentConns:
            print("-", conn.inmod.name, " to", conn.outmod.name)
            if conn.paramdim > 0:
                print("- parameters", conn.params)

y = net.activate([1, 1])
print('Y=', y)
