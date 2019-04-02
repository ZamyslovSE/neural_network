import matplotlib.pyplot as pyplot
import random
import numpy
import ast

class Neuron:
    # Конструктор
    def __init__(self, W, D):        
        self.W = W         
        self.D = D   
    
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
            distance = self.findDistance(point, self.neurons[i].W)
            if (distance < minDistance):
                minDistance = distance
                minIndex = i
        return minIndex
    
    def findDistance(self, point1, point2):
        distance = 0.0
        for i in range(len(point1)):
            distance += (point2[i] - point1[i])**2
        return numpy.sqrt(distance)
    
    def train(self, inputs, sigma0, searchRadius, tau1, tau2):
        print('START TRAINING')
        for i in range(len(inputs)):
            affectedNeurons = []
            distances = []
            closestIndex = self.findClosest(inputs[i])
            radiusI = searchRadius * numpy.exp(- (i / tau1))
            learningRateI = self.learningRate * numpy.exp(- (i / tau2))
            sigma = sigma0 * numpy.exp(- (i / tau1))
            #print('radiusI: ', radiusI)
            #print('learningRateI: ', learningRateI)
            #print('sigma: ', sigma)
            for j in range(len(self.neurons)):
                distance = self.findDistance(self.neurons[j].W, self.neurons[closestIndex].W)
                if (distance < radiusI):
                    affectedNeurons.append(j)
                    distances.append(distance)
            for j in range(len(affectedNeurons)):
                hJ = numpy.exp(-((self.findDistance(self.neurons[closestIndex].D, self.neurons[affectedNeurons[j]].D)**2) / (2 * sigma**2)))
                #print('hJ: ', hJ)
                for k in range(self.varCount):
                    self.neurons[affectedNeurons[j]].W[k] += learningRateI * hJ * (inputs[i][k] - self.neurons[affectedNeurons[j]].W[k]) #* (((radiusI - distances[j]) / radiusI)**2)
            print('TRAINED ', i)
        print('FINISHED TRAINING')