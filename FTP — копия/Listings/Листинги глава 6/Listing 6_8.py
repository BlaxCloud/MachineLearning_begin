# Listing 6.8
# Это промежуточный код, он не является рабочим
from pybrain3.tools.xml.networkwriter import NetworkWriter
from pybrain3.tools.xml.networkreader import NetworkReader
NetworkWriter.writeToFile(net, 'filename.xml')
net = NetworkReader.readFrom('filename.xml')
