import matplotlib.pyplot as pyplot
import random
import numpy
import ast

class SOM:
    # Конструктор
    def __init__(self, neurons, varCount, learningRate):        
        self.neurons = neurons        
        self.varCount = varCount      
        self.learningRate = learningRate
        
    def findClosest(self, point):
        minDistance = 10
        minIndex = -1
        for i in range(len(self.neurons)):
            #print(point, '; ', self.neurons[i])
            distance = self.findDistance(point, self.neurons[i])
            if (distance < minDistance):
                minDistance = distance
                minIndex = i
        return minIndex
    
    def findDistance(self, point1, point2):
        distance = 0.0
        for i in range(len(point1)):
            distance += (point2[i] - point1[i])**2
        return numpy.sqrt(distance)
    
    def train(self, inputs, sigma0, tau):
        print('START TRAINING')
        for i in range(len(inputs)):
            affectedNeurons = []
            distances = []
            closestIndex = self.findClosest(inputs[i])
            radiusI = sigma0 * numpy.exp(- (i / tau))
            print('radiusI: ', radiusI)
            for j in range(len(self.neurons)):
                distance = self.findDistance(self.neurons[j], self.neurons[closestIndex])
                if (distance < radiusI):
                    affectedNeurons.append(j)
                    distances.append(distance)
            for j in range(len(affectedNeurons)):
                for k in range(self.varCount):
                    self.neurons[affectedNeurons[j]][k] += self.learningRate * (inputs[i][k] - self.neurons[affectedNeurons[j]][k]) * (((radiusI - distances[j]) / radiusI)**2)
            print('TRAINED ', i)
        print('FINISHED TRAINING')