from perceptron import Perceptron
import numpy

inputNodeCount = 2
hiddenNodeCount = 50
outputNodeCount = 5
layerCount = 3
learningRate = 0.3

perc = Perceptron(inputNodeCount,
                  hiddenNodeCount,
                  outputNodeCount,
                  layerCount,
                  learningRate)

l1 = [1,2,3,4]
l2 = [6,5,4,3]
inputs1 = numpy.array(l1, ndmin=2).T
inputs2 = numpy.array(l2, ndmin=2).T
print(inputs1 * inputs2)
